use crate::global_thermal::global_h3_cell::{GlobalH3Cell, GlobalH3CellConfig};
use crate::h3_utils::H3Utils;
use crate::planet::Planet;
use crate::sim::sim_op::{SimOp, SimOpHandle};
use h3o::{CellIndex, Resolution};
use std::collections::HashMap;
use std::rc::Rc;


pub struct Simulation {
    pub planet: Planet,
    pub resolution: Resolution,
    ops: Vec<Box<dyn SimOp>>,
    pub cells: HashMap<CellIndex, GlobalH3Cell>,
    pub layer_count: usize,
    pub step: i32,
    pub sim_steps: i32,
    pub years_per_step: u32,
    pub name: String,
    pub debug: bool,
}

pub struct SimProps {
    pub planet: Planet,
    pub name: &'static str,
    pub ops: Vec<SimOpHandle>,
    pub res: Resolution,
    pub layer_count: usize,
  //  pub asth_layer_height_km: f64,
   // pub lith_layer_height_km: f64, 
    pub sim_steps: i32,
    pub years_per_step: u32,
  //  pub alert_freq: usize,
   // pub starting_surface_temp_k: f64,
    pub debug: bool,
}

impl Simulation {
    pub fn new(props: SimProps) -> Simulation {
        let ops = props.ops.into_iter().map(|handle| handle.op).collect();
        let sim = Simulation {
            planet: props.planet,
            ops,
            resolution: props.res,
            cells: HashMap::new(),
            layer_count: props.layer_count,
            step: -1,
            sim_steps: props.sim_steps,
            years_per_step: props.years_per_step,
            name: props.name.to_string(),
            debug: props.debug,
        };
        sim
    }

    pub fn make_cells<F>(&mut self, config_fn: F)
    where
        F: Fn(CellIndex, Rc<Planet>) -> GlobalH3CellConfig
    {
        // Create shared planet reference
        let planet = Rc::new(self.planet.clone());

        for (cell_index, _base) in H3Utils::iter_cells_with_base(self.resolution) {
            // Use the provided function to create configuration for this cell
            let config = config_fn(cell_index, planet.clone());

            let cell = GlobalH3Cell::new_with_config(config);
            self.cells.insert(cell_index, cell);
        }
    }

    /// Get the current simulation step number
    pub fn current_step(&self) -> i32 {
        self.step
    }

    /// Run a single step with custom operators (for testing)
    pub fn step_with_ops(&mut self, ops: &mut [&mut dyn SimOp]) {
        // Reset next state to current state for all global cells
        for cell in self.cells.values_mut() {
            cell.reset_next_state();
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
    }

    pub fn run(&mut self) {
        self.step = 0;
        self.simulate_init();
        loop {
            self.step += 1;

            self.simulate_step();
            self.advance();

            if self.step >= self.sim_steps {
                break;
            }
        }
        self.simulate_end();
    }

    fn advance(&mut self) {
        for cell in self.cells.values_mut() {
            cell.commit_next_state();
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
