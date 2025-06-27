use crate::asth_cell::AsthCellColumn;
use crate::constants::{M2_PER_KM2, SECONDS_PER_YEAR, SIGMA_KM2_YEAR};
use crate::energy_mass::{EnergyMass, StandardEnergyMass};
use crate::material::{MaterialProfile, MaterialType};
use crate::sim::Simulation;
use crate::sim::sim_op::{SimOp, SimOpHandle};

/// Thermal Diffusion Operator
///
/// Implements realistic thermal diffusion between layers with:
/// - Energy cascading when layers reach thermal capacity limits
/// - Material-based thermal conductivity
/// - Temperature-driven energy flow (hot to cold)
/// - Proper thermal equilibration between adjacent layers
#[derive(Debug, Clone)]
pub struct ThermalDiffusionOp {
    pub name: String,
    pub diffusion_rate: f64,           // Base diffusion rate (0.0 to 1.0)
    pub max_temp_change_per_step: f64, // Maximum temperature change per step (K)
}

struct Transfer {
    from_index: usize,
    to_index: usize,
    energy: f64,
}

enum LayerType {
    Lith,
    Asth,
}
struct LayerPointer {
    pub layer_type: LayerType,
    pub index: usize,
    pub energy_mass: StandardEnergyMass,
    pub height_km: f64,
}

impl ThermalDiffusionOp {
    /// Create a new thermal diffusion operator
    pub fn new(diffusion_rate: f64, max_temp_change_per_step: f64) -> Self {
        Self {
            name: "ThermalDiffusionOp".to_string(),
            diffusion_rate,
            max_temp_change_per_step,
        }
    }

    /// Create a handle for the thermal diffusion operator
    pub fn handle(diffusion_rate: f64, max_temp_change_per_step: f64) -> SimOpHandle {
        SimOpHandle::new(Box::new(Self::new(
            diffusion_rate,
            max_temp_change_per_step,
        )))
    }

    /// Process thermal diffusion for a single cell column
    fn process_cell_thermal_diffusion(&self, column: &mut AsthCellColumn, years: f64) {
        // First, radiate energy from the top layer to space
        self.radiate_top_layer_to_space(column, years);

        // Create a simplified thermal diffusion between adjacent asthenosphere layers
        // This avoids complex borrowing issues while still providing realistic thermal mixing

        let mut columns: Vec<LayerPointer> = Vec::new();
        let mut asth_columns: Vec<LayerPointer> = Vec::new();
        for index in 0..column.lithospheres.iter().len() {
            if let lith = column
                .lithospheres
                .get(index)
                .expect("cannot get lithosphere")
            {
                columns.push(LayerPointer {
                    index,
                    layer_type: LayerType::Lith,
                    energy_mass: StandardEnergyMass::new_with_material_energy(lith.material_type(), lith.energy_joules(), lith.volume_km3()),
                    height_km: lith.height_km,
                })
            }
        }
        for index in 0..column.asth_layers.iter().len() {
            if let layer = column
                .asth_layers
                .get(index)
                .expect("cannot get lithosphere")
            {
                asth_columns.push(LayerPointer {
                    index,
                    layer_type: LayerType::Asth,
                    energy_mass: StandardEnergyMass::new_with_material_energy(layer.material_type(), layer.energy_joules(), layer.volume_km3()),
                    height_km: column.layer_height_km,
                })
            }
        }

        columns.extend(asth_columns);

        for (index, pointer) in columns.iter().enumerate() {
            if (index > 0) {
                let prev_pointer = columns.get(index - 1).unwrap();
                self.transfer_energy(years, column, pointer, prev_pointer)
            }
        }
    }

    fn transfer_energy(
        &self,
        years: f64,
        column: &mut AsthCellColumn,
        from_pointer: &LayerPointer,
        to_pointer: &LayerPointer,
    ) {
        // Use the energy_mass from LayerPointer
        let from_mass = &from_pointer.energy_mass;
        let to_mass = &to_pointer.energy_mass;

        // Delegate calculation
        let base_energy_transfer =
            from_mass.calculate_thermal_transfer(to_mass, self.diffusion_rate, years);

        let (source_height_km, source_energy) = if base_energy_transfer > 0.0 {
            (from_pointer.height_km, from_pointer.energy_mass.energy()) // from is hotter
        } else {
            (to_pointer.height_km, to_pointer.energy_mass.energy()) // to is hotter
        };
        let height_scale = (source_height_km * 2.0).sqrt();

        let energy_transfer = base_energy_transfer * height_scale;
        let max_transfer_rate = energy_transfer.abs().min(source_energy * ENERGY_THROTTLE);
        match from_pointer.layer_type {
            LayerType::Lith => {
                let (_, from_lith, _) = column.lithosphere(from_pointer.index);

                if energy_transfer > 0.0 {
                    from_lith.remove_energy(max_transfer_rate);
                } else {
                    from_lith.add_energy(-max_transfer_rate);
                }
            }
            LayerType::Asth => {
                let (_, from_asth) = column.layer(from_pointer.index);

                if energy_transfer > 0.0 {
                    from_asth.remove_energy(max_transfer_rate);
                } else {
                    from_asth.add_energy(-max_transfer_rate);
                }
            }
        }

        match to_pointer.layer_type {
            LayerType::Lith => {
                let (_, to_lith, _) = column.lithosphere(to_pointer.index);

                if energy_transfer > 0.0 {
                    to_lith.add_energy(max_transfer_rate);
                } else {
                    to_lith.remove_energy(-max_transfer_rate);
                }
            }
            LayerType::Asth => {
                let (_, to_asth) = column.layer(to_pointer.index);

                if energy_transfer > 0.0 {
                    to_asth.add_energy(max_transfer_rate);
                } else {
                    to_asth.remove_energy(-max_transfer_rate);
                }
            }
        }
    }

    /// Radiate energy from the top layer to space using Stefan-Boltzmann law
    fn radiate_top_layer_to_space(&self, column: &mut AsthCellColumn, years: f64) {
        // Determine which layer is the surface and remove energy from it
        if !column.lithospheres_next.is_empty()
            && column.lithospheres_next.last().unwrap().height_km > 10.0
        {
            // Radiate from top lithosphere layer
            let top_lithosphere = column.lithospheres_next.last().unwrap();
            let surface_temp = top_lithosphere.kelvin();

            // Calculate radiated energy per km²
            let radiated_energy_per_km2 = SIGMA_KM2_YEAR * surface_temp.powi(4) * years;
            let total_radiated_energy = radiated_energy_per_km2 * column.area();

            // Remove from top lithosphere layer
            let (_, top_lithosphere, _) = column.lithosphere(0);
            let current_energy = top_lithosphere.energy_joules();
            let energy_to_remove = total_radiated_energy.min(current_energy * 0.9); // Max 90% per step
            top_lithosphere.remove_energy(energy_to_remove);
        } else {
            // Radiate from top asthenosphere layer
            let surface_temp = column.asth_layers_next[0].kelvin();

            // Calculate radiated energy per km²
            let radiated_energy_per_km2 = SIGMA_KM2_YEAR * surface_temp.powi(4) * years;
            let total_radiated_energy = radiated_energy_per_km2 * column.area();

            // Remove from top asthenosphere layer
            let (_, top_layer_next) = column.layer(0);
            let current_energy = top_layer_next.energy_joules();
            let energy_to_remove = total_radiated_energy.min(current_energy * ENERGY_THROTTLE); // Max 20% per step
            top_layer_next.remove_energy(energy_to_remove);
        }
    }
}

const ENERGY_THROTTLE: f64 = 0.2;

impl SimOp for ThermalDiffusionOp {
    fn name(&self) -> &str {
        &self.name
    }

    fn update_sim(&mut self, sim: &mut Simulation) {
        let years = sim.years_per_step as f64;

        for column in sim.cells.values_mut() {
            self.process_cell_thermal_diffusion(column, years);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::EARTH_RADIUS_KM;
    use crate::planet::Planet;
    use crate::sim::{SimProps, Simulation};
    use h3o::Resolution;

    #[test]
    fn test_thermal_diffusion_creation() {
        let diffusion_op = ThermalDiffusionOp::new(0.1, 50.0);
        assert_eq!(diffusion_op.diffusion_rate, 0.1);
        assert_eq!(diffusion_op.max_temp_change_per_step, 50.0);
    }

    #[test]
    fn test_thermal_diffusion_equilibration() {
        let mut sim = Simulation::new(SimProps {
            name: "thermal_diffusion_test",
            planet: Planet {
                radius_km: EARTH_RADIUS_KM as f64,
                resolution: Resolution::Zero,
            },
            ops: vec![],
            res: Resolution::Two,
            layer_count: 3,
            layer_height_km: 50.0,
            sim_steps: 1,
            years_per_step: 1000,
            debug: false,
            alert_freq: 1,
            starting_surface_temp_k: 1000.0,
        });

        // Create temperature gradient: hot bottom, cool top
        {
            let cell = sim.cells.values_mut().next().unwrap();
            cell.asth_layers_next[0].set_temp_kelvin(1000.0); // Cool surface
            cell.asth_layers_next[1].set_temp_kelvin(1500.0); // Medium
            cell.asth_layers_next[2].set_temp_kelvin(2000.0); // Hot bottom
        }

        let initial_temps: Vec<f64> = {
            let cell = sim.cells.values().next().unwrap();
            cell.asth_layers_next.iter().map(|l| l.kelvin()).collect()
        };

        // Apply thermal diffusion
        let mut diffusion_op = ThermalDiffusionOp::new(0.1, 100.0);
        diffusion_op.update_sim(&mut sim);

        let final_temps: Vec<f64> = {
            let cell = sim.cells.values().next().unwrap();
            cell.asth_layers_next.iter().map(|l| l.kelvin()).collect()
        };

        // Temperatures should move toward equilibrium
        // Top layer should get warmer, bottom layer should get cooler
        assert!(
            final_temps[0] > initial_temps[0],
            "Top layer should warm up"
        );
        assert!(
            final_temps[2] < initial_temps[2],
            "Bottom layer should cool down"
        );

        println!("Initial temps: {:?}", initial_temps);
        println!("Final temps: {:?}", final_temps);
    }
}
