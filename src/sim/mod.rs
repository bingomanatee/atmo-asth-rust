pub mod sim_op;

use crate::constants::{ASTHENOSPHERE_SURFACE_START_TEMP_K, SPECIFIC_HEAT_CAPACITY_MANTLE_J_PER_KG_K};
use crate::h3_utils::H3Utils;
use crate::planet::Planet;
use crate::sim::sim_op::{SimOp, SimOpHandle};
use crate::temp_utils::volume_kelvin_to_joules;
use h3o::{CellIndex, Resolution};
use std::collections::HashMap;
use crate::asth_cell::{AsthCellColumn, AsthCellParams};

pub struct Simulation {
    pub planet: Planet,
    pub resolution: Resolution,
    ops: Vec<Box<dyn SimOp>>,
    pub cells: HashMap<CellIndex, AsthCellColumn>,
    layer_count: usize,
    layer_height_km: f64,
    step: i32,
    sim_steps: i32,
    pub years_per_step: u32,
    name: String,
    debug: bool,
    alert_freq: usize,
    starting_surface_temp_k: f64
}

pub struct SimProps {
    pub planet: Planet,
    pub name: &'static str,
    pub ops: Vec<SimOpHandle>,
    pub res: Resolution,
    pub layer_count: usize,
    pub layer_height_km: f64,
    pub sim_steps: i32,
    pub years_per_step: u32,
    pub alert_freq: usize,
    pub starting_surface_temp_k: f64,
    pub debug: bool,
}

impl Simulation {
    pub fn new(props: SimProps) -> Simulation {
        let ops = props.ops.into_iter().map(|handle| handle.op).collect();
        let mut sim = Simulation {
            planet: props.planet,
            ops,
            resolution: props.res,
            cells: HashMap::new(),
            layer_count: props.layer_count,
            layer_height_km: props.layer_height_km,
            step: -1,
            sim_steps: props.sim_steps,
            years_per_step: props.years_per_step,
            name: props.name.to_string(),
            alert_freq: props.alert_freq,
            debug: props.debug,
            starting_surface_temp_k: props.starting_surface_temp_k
        };
        sim.make_cells();
        sim
    }

    pub fn make_cells(&mut self) {
        for (cell_index, _base) in H3Utils::iter_cells_with_base(self.resolution) {
            let volume = H3Utils::cell_area(self.resolution, self.planet.radius_km);
            // Create cells with surface temperature - projection will handle geothermal gradient
            let cell = AsthCellColumn::new(AsthCellParams {
                cell_index,
                volume,
                energy: 0.0, // This will be ignored by the constructor
                layer_count: self.layer_count,
                layer_height_km: self.layer_height_km,
                planet_radius_km: self.planet.radius_km,
                surface_temp_k: self.starting_surface_temp_k
            });
            self.cells.insert(cell_index, cell);
        }
    }

    fn report_step(&self) {
        if !self.debug {
            return;
        }

        match self.alert_freq {
            0 => {
                return;
            }
            1 => {}
            _ => {
                if (self.step % self.alert_freq as i32) > 0  {
                    return;
                }
            }
        }

        let global_energy = self.energy_at_layer(0);
        let cells = self.cells.iter().len();
        println!(
            "{} STEP {}: surface global energy (J) {:.3e}, per cell {:.3e} J",
            self.name,
            self.step,
            global_energy,
            global_energy / cells as f64
        );
    }

    pub fn energy_at_layer(&self, layer: usize) -> f64 {
        self.cells
            .iter()
            .map(|(_index, cell)| match cell.asth_layers.get(layer) {
                None => 0.0,
                Some(layer) => layer.energy_joules(),
            })
            .sum()
    }

    /// Get the current simulation step number
    pub fn current_step(&self) -> i32 {
        self.step
    }

    /// Run a single step with custom operators (for testing)
    pub fn step_with_ops(&mut self, ops: &mut [&mut dyn SimOp]) {
        // Copy current to next arrays
        for column in self.cells.values_mut() {
            for i in 0..column.asth_layers.len() {
                column.asth_layers_next[i] = column.asth_layers[i].clone();
            }
            column.lithospheres_next = column.lithospheres.clone();
        }

        // Run operators on next arrays
        for op in ops {
            op.update_sim(self);
        }

        // Commit next to current arrays
        self.advance();

        self.step += 1;
    }

    pub fn simulate(&mut self) {
        if self.step > -1 {
            panic!("Simulation.simulate can only execute once");
        }
        self.step = 0;
        self.simulate_init();
        loop {
            self.step += 1;

            self.simulate_step();
            self.advance();

            self.report_step();

            if self.step >= self.sim_steps {
                break;
            }
        }
        self.simulate_end();
    }

    fn advance(&mut self) {
        for column in self.cells.values_mut() {
            column.commit_next_layers();
        }
    }

    fn simulate_init(&mut self) {
        let mut ops = std::mem::take(&mut self.ops);

        for op in &mut ops {
            op.init_sim(self);
        }
        self.ops = ops;
    }

    fn simulate_end(&mut self) {
        let mut ops = std::mem::take(&mut self.ops);

        for op in &mut ops {
            op.after_sim(self);
        }
        self.ops = ops;
    }

    fn simulate_step(&mut self) {
        let mut ops = std::mem::take(&mut self.ops);

        for op in &mut ops {
            op.update_sim(self);
        }
        self.ops = ops;
    }
}

#[cfg(test)]
mod tests {
    use crate::asth_cell::asth_cell_column::AsthCellColumn;
    use crate::constants::{ASTHENOSPHERE_SURFACE_START_TEMP_K, EARTH_RADIUS_KM};
    use crate::planet::Planet;
    use crate::sim::sim_op::SimOp;
    use crate::sim::{SimProps, Simulation};
    use approx::assert_abs_diff_eq;
    use h3o::{CellIndex, Resolution};

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
            layer_height_km: 10.0,
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
                    let (_, next_layer) = column.layer(0);
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
            layer_height_km: 10.0,
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
