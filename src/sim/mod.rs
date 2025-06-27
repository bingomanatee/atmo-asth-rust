pub mod sim_op;
pub mod simulation;
use crate::sim::sim_op::SimOp;

#[cfg(test)]
mod tests {
    use crate::asth_cell::asth_cell_column::AsthCellColumn;
    use crate::constants::{ASTHENOSPHERE_SURFACE_START_TEMP_K, EARTH_RADIUS_KM};
    use crate::planet::Planet;
    use crate::sim::sim_op::SimOp;
    use approx::assert_abs_diff_eq;
    use h3o::{CellIndex, Resolution};
    use crate::sim::simulation::{SimProps, Simulation};

    #[test]
    fn creation() {
        let sim = Simulation::new(SimProps {
            name: "creation_test",
            planet: Planet {
                radius_km: EARTH_RADIUS_KM as f64,
                resolution: Resolution::One,
            },
            ops: vec![],
            res: Resolution::Two,
            layer_count: 4,
            asth_layer_height_km: 10.0,
            lith_layer_height_km: 5.0,
            sim_steps: 500,
            years_per_step: 1_000_000,
            debug: true,
            alert_freq: 100,
            starting_surface_temp_k: ASTHENOSPHERE_SURFACE_START_TEMP_K
        });

        assert_eq!(sim.cells.into_iter().len(), 5882);
    }

    #[test]
    fn simple_sim() {
        /// Example operation that demonstrates all three lifecycle methods
        pub struct ExampleOp {
            pub intensity: f64,
            pub init_count: usize,
            pub update_count: usize,
        }

        impl ExampleOp {
            fn new(intensity: f64) -> ExampleOp {
                ExampleOp {
                    intensity,
                    init_count: 0,
                    update_count: 0,
                }
            }

            fn handle(intensity: f64) -> SimOpHandle {
                SimOpHandle::new(Box::new(ExampleOp::new(intensity)))
            }
        }

        impl SimOp for ExampleOp {
            fn name(&self) -> &str {
                "ExampleOp"
            }

            fn init_sim(&mut self, sim: &mut Simulation) {
                self.init_count += 1;
                println!("Initializing simulation '{}' with {} cells", sim.name, sim.cells.len());
            }

            fn update_sim(&mut self, sim: &mut Simulation) {
                self.update_count += 1;
                for column in sim.cells.values_mut() {
                    let (_, next_layer) = column.asth_layer(0);
                    let current_energy = next_layer.energy_joules();
                    next_layer.set_energy_joules(current_energy * self.intensity);
                }
            }

            fn after_sim(&mut self, sim: &mut Simulation) {
                println!("Simulation '{}' completed after {} updates", sim.name, self.update_count);
            }
        }

        use crate::sim::sim_op::SimOpHandle;

        let mut sim = Simulation::new(SimProps {
            name: "simple_cooling_sim",
            planet: Planet {
                radius_km: EARTH_RADIUS_KM as f64,
                resolution: Resolution::One,
            },
            ops: vec![ExampleOp::handle(0.99)],
            res: Resolution::Two,
            layer_count: 4,
            asth_layer_height_km: 100.0,
            lith_layer_height_km: 50.0,
            sim_steps: 500,
            years_per_step: 1_000_000,
            debug: true,
            alert_freq: 100,
            starting_surface_temp_k: ASTHENOSPHERE_SURFACE_START_TEMP_K
        });

        for (id, cell) in sim.cells.clone() {
            if let Some(layer) = cell.asth_layers.first() {
                assert_abs_diff_eq!(layer.energy_joules(), 6.04e23, epsilon = 5.0e22);
            }
        }

        sim.simulate();

        for (id, cell) in sim.cells {
            if let Some(layer) = cell.asth_layers.first() {
                assert_abs_diff_eq!(layer.energy_joules(), 3.5e21, epsilon = 5.0e20);
            }
        }
    }
}
