use crate::sim::sim_op::{SimOp, SimOpHandle};
use crate::sim::Simulation;
use crate::temp_utils::{cooling_per_cell_per_year, radiance_per_cell_per_year};

/// this is a VERY SIMPLE model adding radiance  from the area below the earth to the upper layer. 
/// will be replaced by simulated radiance later
pub struct RadianceOp;

impl RadianceOp {
    pub fn new() -> RadianceOp {
        RadianceOp
    }

    pub fn handle() -> SimOpHandle {
        SimOpHandle::new(Box::new(RadianceOp::new()))
    }
}

impl SimOp for RadianceOp {
    fn name(&self) -> &str {
        "RadianceOp"
    }

    fn update_sim(&mut self, sim: &mut Simulation) {
        for column in sim.cells.values_mut() {
            let lithosphere_height = column.total_lithosphere_height();
            let (_, next_layer) = column.layer(0);

            let radiance_per_year = radiance_per_cell_per_year(sim.resolution, sim.planet.radius_km, lithosphere_height);
            let radiance = radiance_per_year * sim.years_per_step as f64;
            next_layer.add_energy(radiance);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::RadianceOp;
    use approx::assert_abs_diff_eq;
    use crate::constants::{ASTHENOSPHERE_SURFACE_START_TEMP_K, EARTH_RADIUS_KM};
    use crate::planet::Planet;
    use crate::sim::sim_op::SimOpHandle;
    use crate::sim::{SimProps, Simulation};
    use h3o::Resolution;
    #[test]
    fn radiance() {
        let mut sim = Simulation::new(SimProps {
            name: "radiance_op_test",
            planet: Planet {
                radius_km: EARTH_RADIUS_KM as f64,
                resolution: Resolution::One,
            },
            ops: vec![RadianceOp::handle()],
            res: Resolution::Two,
            layer_count: 4,
            layer_height_km: 10.0,
            sim_steps: 10,
            years_per_step: 100_000,
            debug: true,
            alert_freq: 1,
            starting_surface_temp_k: ASTHENOSPHERE_SURFACE_START_TEMP_K,
        });

        for (_id, cell) in sim.cells.clone() {
            if let Some(layer) = cell.layers.first() {
                assert_abs_diff_eq!(layer.energy_joules(),  6.04e23, epsilon= 5.0e22);
            }
        }

        sim.simulate();

        for (_id, cell) in sim.cells {
            if let Some(layer) = cell.layers.first() {
                // Updated expected value to match corrected operator behavior
                // The operators now correctly modify next arrays, resulting in different energy distribution
                assert_abs_diff_eq!(layer.energy_joules(), 6.70e23, epsilon= 5.0e21);
            }
        }

    }

    #[test]
    fn test_radiance_op_modifies_next_arrays() {
        use crate::sim::{Simulation, SimProps};
        use crate::planet::Planet;
        use crate::constants::EARTH_RADIUS_KM;
        use h3o::Resolution;

        // Create test simulation with temperature gradient
        let mut sim = Simulation::new(SimProps {
            name: "test_radiance",
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
            starting_surface_temp_k: 1500.0,
        });

        // Create temperature gradient: cool surface, hot bottom
        let cell = sim.cells.values_mut().next().unwrap();
        cell.layers_next[0].set_energy_joules(1.0e20); // Cool surface
        cell.layers_next[1].set_energy_joules(2.0e20); // Medium
        cell.layers_next[2].set_energy_joules(3.0e20); // Hot bottom

        // Record initial temperatures
        let cell = sim.cells.values_mut().next().unwrap();
        let initial_temps: Vec<f64> = (0..3).map(|i| {
            let (current, _) = cell.layer(i);
            current.kelvin()
        }).collect();
        let initial_gradient = initial_temps[0] - initial_temps[initial_temps.len() - 1];

        // Apply RadianceOp
        let mut radiance_op = RadianceOp::new();
        sim.step_with_ops(&mut [&mut radiance_op]);

        // Verify thermal mixing occurred using layer() method
        let cell = sim.cells.values_mut().next().unwrap();
        let final_temps: Vec<f64> = (0..3).map(|i| {
            let (current, _) = cell.layer(i);
            current.kelvin()
        }).collect();
        let final_gradient = final_temps[0] - final_temps[final_temps.len() - 1];

        let gradient_change = (initial_gradient - final_gradient).abs();
        let temps_changed = initial_temps != final_temps;

        assert!(temps_changed, "RadianceOp should change temperatures!");
        assert!(gradient_change > 0.01, "RadianceOp should change temperature gradient! Change: {:.3}K", gradient_change);

        println!("✅ RadianceOp test passed - Gradient change: {:.3}K", gradient_change);
    }
}
