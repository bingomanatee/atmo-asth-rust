use crate::asth_cell::{AsthCellColumn, AsthCellParams};
use crate::h3_utils::H3Utils;
use crate::planet::Planet;
use crate::sim::sim_op::{SimOp, SimOpHandle};
use h3o::{CellIndex, Resolution};
use std::collections::HashMap;

pub struct Simulation {
    pub planet: Planet,
    pub resolution: Resolution,
    ops: Vec<Box<dyn SimOp>>,
    pub cells: HashMap<CellIndex, AsthCellColumn>,
    pub layer_count: usize,
    pub asth_layer_height_km: f64,
    pub lith_layer_height_km: f64,
    pub step: i32,
    pub sim_steps: i32,
    pub years_per_step: u32,
    pub name: String,
    pub debug: bool,
    alert_freq: usize,
    starting_surface_temp_k: f64,
}

pub struct SimProps {
    pub planet: Planet,
    pub name: &'static str,
    pub ops: Vec<SimOpHandle>,
    pub res: Resolution,
    pub layer_count: usize,
    pub asth_layer_height_km: f64,
    pub lith_layer_height_km: f64, 
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
            asth_layer_height_km: props.asth_layer_height_km,
            lith_layer_height_km: props.lith_layer_height_km,
            step: -1,
            sim_steps: props.sim_steps,
            years_per_step: props.years_per_step,
            name: props.name.to_string(),
            alert_freq: props.alert_freq,
            debug: props.debug,
            starting_surface_temp_k: props.starting_surface_temp_k,
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
                layer_height_km: self.asth_layer_height_km,
                planet_radius_km: self.planet.radius_km,
                surface_temp_k: self.starting_surface_temp_k,
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
                if (self.step % self.alert_freq as i32) > 0 {
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
            column.lithospheres_next = column.lith_layers.clone();
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
