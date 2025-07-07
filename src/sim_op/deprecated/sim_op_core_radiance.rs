// Core Radiance Operator
// Adds Earth's core radiance energy influx to the bottom asthenosphere layer

use crate::sim::simulation::Simulation;
use crate::sim_op::{SimOp, SimOpHandle};
use crate::energy_mass::EnergyMass;

#[derive(Debug, Clone)]
pub struct CoreRadianceOp {
    pub name: String,
    pub core_radiance_j_per_km2_per_year: f64,
}

impl CoreRadianceOp {
    pub fn handle(core_radiance_j_per_km2_per_year: f64) -> SimOpHandle {
        SimOpHandle::new(Box::new(CoreRadianceOp {
            name: "CoreRadianceOp".to_string(),
            core_radiance_j_per_km2_per_year,
        }))
    }

    pub fn handle_earth() -> SimOpHandle {
        // Earth's core radiance: 2.52e12 J per km² per year
        SimOpHandle::new(Box::new(CoreRadianceOp {
            name: "CoreRadianceOp".to_string(),
            core_radiance_j_per_km2_per_year: 2.52e12,
        }))
    }
}

impl SimOp for CoreRadianceOp {
    fn name(&self) -> &str {
        &self.name
    }

    fn update_sim(&mut self, sim: &mut Simulation) {
        // Apply core radiance to the bottom layer of each cell
        for column in sim.cells.values_mut() {
            // Get area first to avoid borrowing conflicts
            let area_km2 = column.area();
            let years = sim.years_per_step as f64;

            // Apply core radiance to bottom layer - MODIFY NEXT ARRAYS!
            if let Some((_, bottom_layer)) = column.asth_layers_t.last_mut() {
                // Calculate energy influx for this time step
                let energy_influx = self.core_radiance_j_per_km2_per_year * area_km2 * years;

                // Add energy to bottom layer (temperature will increase)
                bottom_layer.add_energy(energy_influx);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planet::Planet;
    use crate::constants::EARTH_RADIUS_KM;
    use h3o::Resolution;
    use crate::sim::simulation::{SimProps, Simulation};

    #[test]
    fn test_core_radiance_op_adds_energy_to_next_arrays() {
        // Create test simulation
        let mut sim = Simulation::new(SimProps {
            name: "test_core_radiance",
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
            starting_surface_temp_k: 1500.0,
        });

        // Record initial bottom layer energy using layer() method
        let cell = sim.cells.values_mut().next().unwrap();
        let bottom_layer_index = cell.asth_layers_t.len() - 1;
        let (current_bottom, _) = &cell.layer(bottom_layer_index);
        let initial_bottom_energy = current_bottom.energy_joules();
        let initial_bottom_temp = current_bottom.kelvin();

        // Apply CoreRadianceOp
        let mut core_op = CoreRadianceOp {
            name: "TestCore".to_string(),
            core_radiance_j_per_km2_per_year: 2.52e12, // Earth's core heat
        };

        sim.step_with_ops(&mut [&mut core_op]);

        // Verify energy was added to bottom layer using layer() method
        let cell = sim.cells.values_mut().next().unwrap();
        let (final_current_bottom, _) = &cell.layer(bottom_layer_index);
        let final_bottom_energy = final_current_bottom.energy_joules();
        let final_bottom_temp = final_current_bottom.kelvin();

        let energy_added = final_bottom_energy - initial_bottom_energy;
        let temp_increase = final_bottom_temp - initial_bottom_temp;

        assert!(energy_added > 0.0, "CoreRadianceOp should add energy! Added: {:.2e} J", energy_added);
        assert!(temp_increase > 0.0, "CoreRadianceOp should increase temperature! Increase: {:.1}K", temp_increase);

        println!("✅ CoreRadianceOp test passed - Energy added: {:.2e} J, Temp increase: {:.1}K", energy_added, temp_increase);
    }
}


