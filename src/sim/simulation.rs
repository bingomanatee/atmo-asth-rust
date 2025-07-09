use crate::global_thermal::global_h3_cell::{GlobalH3Cell, GlobalH3CellConfig};
use crate::h3_utils::H3Utils;
use crate::planet::Planet;
use crate::sim_op::{SimOp, SimOpHandle};
use h3o::{CellIndex, Resolution};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct OpTiming {
    pub op_name: String,
    pub init_time: Duration,
    pub total_update_time: Duration,
    pub update_call_count: u32,
    pub after_time: Duration,
}

impl OpTiming {
    pub fn new(op_name: String) -> Self {
        Self {
            op_name,
            init_time: Duration::ZERO,
            total_update_time: Duration::ZERO,
            update_call_count: 0,
            after_time: Duration::ZERO,
        }
    }
    
    pub fn avg_update_time(&self) -> Duration {
        if self.update_call_count > 0 {
            self.total_update_time / self.update_call_count
        } else {
            Duration::ZERO
        }
    }
    
    pub fn total_time(&self) -> Duration {
        self.init_time + self.total_update_time + self.after_time
    }
}

pub struct Simulation {
    pub planet: Planet,
    pub resolution: Resolution,
    pub ops: Vec<Box<dyn SimOp>>,
    pub cells: HashMap<CellIndex, GlobalH3Cell>,
    pub layer_count: usize,
    pub step: i32,
    pub sim_steps: i32,
    pub years_per_step: u32,
    pub name: String,
    pub debug: bool,
    pub op_timings: Vec<OpTiming>,
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
        let mut sim = Simulation {
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
            op_timings: Vec::new(),
        };
        
        // Initialize timing structs for each op
        for op in &sim.ops {
            sim.op_timings.push(OpTiming::new(op.name().to_string()));
        }
        
        sim
    }

    pub fn make_cells<F>(&mut self, config_fn: F)
    where
        F: Fn(CellIndex, Arc<Planet>) -> GlobalH3CellConfig
    {
        // Create shared planet reference
        let planet = Arc::new(self.planet.clone());

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
        self.print_timing_report();
    }

    fn advance(&mut self) {
        for cell in self.cells.values_mut() {
            cell.commit_next_state();
        }
    }

    fn simulate_init(&mut self) {
        let mut ops = std::mem::take(&mut self.ops);

        for (i, op) in ops.iter_mut().enumerate() {
            let start = Instant::now();
            op.init_sim(self);
            let elapsed = start.elapsed();
            self.op_timings[i].init_time = elapsed;
        }
        self.ops = ops;
    }

    fn simulate_end(&mut self) {
        let mut ops = std::mem::take(&mut self.ops);

        for (i, op) in ops.iter_mut().enumerate() {
            let start = Instant::now();
            op.after_sim(self);
            let elapsed = start.elapsed();
            self.op_timings[i].after_time = elapsed;
        }
        self.ops = ops;
    }

    fn simulate_step(&mut self) {
        let mut ops = std::mem::take(&mut self.ops);

        for (i, op) in ops.iter_mut().enumerate() {
            let start = Instant::now();
            op.update_sim(self);
            let elapsed = start.elapsed();
            self.op_timings[i].total_update_time += elapsed;
            self.op_timings[i].update_call_count += 1;
        }
        self.ops = ops;
    }
    
    pub fn print_timing_report(&self) {
        println!("\n📊 === SIMULATION TIMING REPORT ===");
        println!("🔄 Total steps: {}", self.sim_steps);
        println!("⏱️  Years per step: {}", self.years_per_step);
        println!();
        
        let mut total_time = Duration::ZERO;
        for timing in &self.op_timings {
            total_time += timing.total_time();
        }
        
        println!("📈 PER-OPERATION BREAKDOWN:");
        for timing in &self.op_timings {
            let total_op_time = timing.total_time();
            let percentage = if total_time.as_millis() > 0 {
                (total_op_time.as_millis() as f64 / total_time.as_millis() as f64) * 100.0
            } else {
                0.0
            };
            
            println!("  🔧 {:<25} | Total: {:>8.2}ms | Avg/step: {:>8.2}ms | Init: {:>6.2}ms | After: {:>6.2}ms | Share: {:>5.1}%",
                timing.op_name,
                total_op_time.as_millis(),
                timing.avg_update_time().as_millis(),
                timing.init_time.as_millis(),
                timing.after_time.as_millis(),
                percentage
            );
        }
        
        println!();
        println!("⏱️  TOTAL SIMULATION TIME: {:.2}ms ({:.2}s)", total_time.as_millis(), total_time.as_secs_f64());
        println!("🚀 Average time per step: {:.2}ms", total_time.as_millis() as f64 / self.sim_steps as f64);
        println!("💫 Steps per second: {:.2}", self.sim_steps as f64 / total_time.as_secs_f64());
        println!("📊 === END TIMING REPORT ===\n");
    }
}
