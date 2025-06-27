use crate::asth_cell::AsthCellColumn;
use crate::sim::sim_op::{SimOp, SimOpHandle};
use crate::sim::Simulation;
use crate::energy_mass::EnergyMass;

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
    pub diffusion_rate: f64,  // Base diffusion rate (0.0 to 1.0)
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
        SimOpHandle::new(Box::new(Self::new(diffusion_rate, max_temp_change_per_step)))
    }


    /// Process thermal diffusion for a single cell column
    fn process_cell_thermal_diffusion(&self, column: &mut AsthCellColumn, years: f64) {
        // Create a simplified thermal diffusion between adjacent asthenosphere layers
        // This avoids complex borrowing issues while still providing realistic thermal mixing

        let mut lith_columns: Vec<LayerPointer> = Vec::new();
        let mut asth_columns: Vec<LayerPointer> = Vec::new();
        for index in 0..column.lithospheres_next.iter().len() {
            lith_columns.push(LayerPointer {
                index, layer_type: LayerType::Lith
            })
        }
        for index in 0..column.layers_next.iter().len() {
            asth_columns.push(LayerPointer {
                index, layer_type: LayerType::Asth
            })
        }
        
        let mut columns: Vec<LayerPointer> = lith_columns.iter().rev().collect() as Vec<LayerPointer>;
        columns.extend(asth_columns);

        // Calculate temperature differences and energy transfers between adjacent layers
        let mut energy_transfers: Vec<Transfer> = vec![];

        for i in 0..columns.iter().len() {
            
            let upper_temp = column.layers_next[i].kelvin();
            let lower_temp = column.layers_next[i + 1].kelvin();
            let temp_diff = lower_temp - upper_temp; // Positive if lower is hotter

            if temp_diff.abs() < 0.1 {
                continue; // No significant temperature difference
            }

            // Calculate thermal conductivity interface
            let upper_conductivity = column.layers_next[i].thermal_conductivity();
            let lower_conductivity = column.layers_next[i + 1].thermal_conductivity();
            let interface_conductivity = 2.0 * upper_conductivity * lower_conductivity /
                (upper_conductivity + lower_conductivity);

            // Calculate energy transfer rate
            let base_transfer_rate = interface_conductivity * self.diffusion_rate * years * 1e6; // Scaling factor
            let energy_transfer = temp_diff * base_transfer_rate;

            // Limit energy transfer to prevent extreme temperature changes
            let upper_energy = column.layers_next[i].energy_joules();
            let lower_energy = column.layers_next[i + 1].energy_joules();

            let max_change = (upper_energy.min(lower_energy)) * 0.1; // Max 10% energy change per step
            let limited_transfer = if energy_transfer > 0.0 {
                energy_transfer.min(max_change)
            } else {
                energy_transfer.max(-max_change)
            };

            energy_transfers.push(Transfer {
                from_index: i + 1,
                to_index: i,
                energy: limited_transfer,
            });
        }

        // Apply energy transfers
        for transfer in energy_transfers {
            let from_cell =
                if (transfer.energy > 0.0) {}
        }
    }
}

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
            cell.layers_next[0].set_temp_kelvin(1000.0); // Cool surface
            cell.layers_next[1].set_temp_kelvin(1500.0); // Medium
            cell.layers_next[2].set_temp_kelvin(2000.0); // Hot bottom
        }

        let initial_temps: Vec<f64> = {
            let cell = sim.cells.values().next().unwrap();
            cell.layers_next.iter().map(|l| l.kelvin()).collect()
        };

        // Apply thermal diffusion
        let mut diffusion_op = ThermalDiffusionOp::new(0.1, 100.0);
        diffusion_op.update_sim(&mut sim);

        let final_temps: Vec<f64> = {
            let cell = sim.cells.values().next().unwrap();
            cell.layers_next.iter().map(|l| l.kelvin()).collect()
        };

        // Temperatures should move toward equilibrium
        // Top layer should get warmer, bottom layer should get cooler
        assert!(final_temps[0] > initial_temps[0], "Top layer should warm up");
        assert!(final_temps[2] < initial_temps[2], "Bottom layer should cool down");

        println!("Initial temps: {:?}", initial_temps);
        println!("Final temps: {:?}", final_temps);
    }
}
