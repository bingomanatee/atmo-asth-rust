use crate::asth_cell::AsthCellColumn;
use crate::energy_mass::{EnergyMass, StandardEnergyMass};
use crate::sim::sim_op::{SimOp, SimOpHandle};
use crate::sim::simulation::Simulation;

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
        for index in 0..column.lith_layers_t.iter().len() {
            if let (lith, _) = column
                .lith_layers_t
                .get(index)
                .expect("cannot get lithosphere")
            {
                columns.push(LayerPointer {
                    index,
                    layer_type: LayerType::Lith,
                    energy_mass: lith.energy_mass.clone(),
                    height_km: lith.height_km,
                })
            }
        }
        for index in 0..column.asth_layers_t.iter().len() {
            if let (layer, _) = column
                .asth_layers_t
                .get(index)
                .expect("cannot get asthenosphere")
            {
                asth_columns.push(LayerPointer {
                    index,
                    layer_type: LayerType::Asth,
                    energy_mass: layer.energy_mass.clone(),
                    height_km: column.layer_height_km,
                })
            }
        }

        columns.extend(asth_columns);
        println!("--------- columns ---------");

        for (index, pointer) in columns.iter().enumerate() {
            if (index > 0) {
                let prev_pointer = columns.get(index - 1).unwrap();
                self.transfer_energy(years, column, pointer, prev_pointer)
            }
        }
        
        for c in columns {
            println!("type: {}, energy: {} J, volume: {} km3, temp: {} K", match c.layer_type {
                LayerType::Lith => "Lithosphere",
                LayerType::Asth => "Atheosphere"
            }, c.energy_mass.energy(), c.energy_mass.volume(), c.energy_mass.kelvin())
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
        if from_pointer.index
            >= match from_pointer.layer_type {
                LayerType::Lith => column.lith_layers_t.len(),
                LayerType::Asth => column.asth_layers_t.len(),
            }
        {
            return;
        }
        // Delegate calculation
        let base_energy_transfer =
            from_mass.calculate_thermal_transfer(to_mass, self.diffusion_rate, years);

        let source_energy = if base_energy_transfer > 0.0 {
            from_pointer.energy_mass.energy() // from is hotter
        } else {
            to_pointer.energy_mass.energy() // to is hotter
        };

        let energy_transfer = base_energy_transfer;
        let max_transfer_rate = energy_transfer.abs().min(source_energy * ENERGY_THROTTLE);
        match from_pointer.layer_type {
            LayerType::Lith => {
                let (_, from_lith) = &mut column.lithosphere_mut(from_pointer.index);

                if energy_transfer > 0.0 {
                    from_lith.remove_energy(max_transfer_rate);
                } else {
                    from_lith.add_energy(-max_transfer_rate);
                }
            }
            LayerType::Asth => {
                let (_, from_asth) = &mut column.layer_mut(from_pointer.index);

                if energy_transfer > 0.0 {
                    from_asth.remove_energy(max_transfer_rate);
                } else {
                    from_asth.add_energy(-max_transfer_rate);
                }
            }
        }

        match to_pointer.layer_type {
            LayerType::Lith => {
                let (_, to_lith) = &mut column.lithosphere_mut(to_pointer.index);

                if energy_transfer > 0.0 {
                    to_lith.add_energy(max_transfer_rate);
                } else {
                    to_lith.remove_energy(-max_transfer_rate);
                }
            }
            LayerType::Asth => {
                let (_, to_asth) = &mut column.layer_mut(to_pointer.index);

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
        let area = column.area();
        // Determine which layer is the surface and remove energy from it
        if !column.lith_layers_t.is_empty()
            && column.lith_layers_t.last().unwrap().1.height_km > 10.0
        {
            // Radiate from top lithosphere layer
            let (_, top_lithosphere) = &mut column.lithosphere_mut(0);
            top_lithosphere
                .energy_mass_mut()
                .radiate_to_space(area, years);
        } else {
            let (_, top_layer_next) = &mut column.layer_mut(0);
            top_layer_next
                .energy_mass_mut()
                .radiate_to_space(area, years);
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
    use crate::sim::simulation::{SimProps, Simulation};
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
            asth_layer_height_km: 50.0,
            lith_layer_height_km: 25.0,
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
