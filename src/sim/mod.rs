pub mod sim_op;

use crate::asth_cell::{AsthCellColumn, AsthCellParams};
use crate::constants::ASTHENOSPHERE_SURFACE_START_TEMP_K;
use crate::h3_utils::H3Utils;
use crate::planet::Planet;
use crate::sim::sim_op::{SimOp, SimOpHandle};
use crate::temp_utils::volume_kelvin_to_joules;
use h3o::{CellIndex, Resolution};
use std::collections::HashMap;

pub struct Simulation {
    planet: Planet,
    resolution: Resolution,
    step_ops: Vec<Box<dyn SimOp>>,
    start_ops: Vec<Box<dyn SimOp>>,
    end_ops: Vec<Box<dyn SimOp>>,
    cells: HashMap<CellIndex, AsthCellColumn>,
    layer_count: usize,
    layer_height_km: f64,
    step: i32,
    sim_steps: i32,
    years_per_step: u32,
}

pub struct SimProps {
    planet: Planet,
    step_ops: Vec<SimOpHandle>,
    start_ops: Vec<SimOpHandle>,
    end_ops: Vec<SimOpHandle>,
    res: Resolution,
    layer_count: usize,
    layer_height: f64,
    layer_height_km: f64,
    sim_steps: i32,
    years_per_step: u32,
}

impl Simulation {
    pub fn new(props: SimProps) -> Simulation {
        let step_ops = props.step_ops.into_iter().map(|handle| handle.op).collect();
        let start_ops = props
            .start_ops
            .into_iter()
            .map(|handle| handle.op)
            .collect();
        let end_ops = props.end_ops.into_iter().map(|handle| handle.op).collect();
        let mut sim = Simulation {
            planet: props.planet,
            step_ops,
            start_ops,
            end_ops,
            resolution: props.res,
            cells: HashMap::new(),
            layer_count: props.layer_count,
            layer_height_km: props.layer_height_km,
            step: -1,
            sim_steps: props.sim_steps,
            years_per_step: props.years_per_step
        };
        sim.make_cells();
        sim
    }

    pub fn make_cells(&mut self) {
        for (cell_index, _base) in H3Utils::iter_cells_with_base(self.resolution) {
            let volume = H3Utils::cell_area(self.resolution, self.planet.radius_km);
            let energy = volume_kelvin_to_joules(volume, ASTHENOSPHERE_SURFACE_START_TEMP_K);
            let cell = AsthCellColumn::new(AsthCellParams {
                cell_index,
                volume,
                energy,
                layer_count: self.layer_count,
                layer_height_km: self.layer_height_km,
                planet_radius: self.planet.radius_km,
            });
            self.cells.insert(cell_index, cell);
        }
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
        let mut ops = std::mem::take(&mut self.start_ops);

        for op in &mut ops {
            op.update_sim(self);
        }
        self.start_ops = ops;
    }

    fn simulate_end(&mut self) {
        let mut ops = std::mem::take(&mut self.end_ops);

        for op in &mut ops {
            op.update_sim(self);
        }
        self.end_ops = ops;
    }

    fn simulate_step(&mut self) {
        let mut ops = std::mem::take(&mut self.step_ops);

        for op in &mut ops {
            op.update_sim(self);
        }
        self.step_ops = ops;
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use crate::asth_cell::AsthCellColumn;
    use crate::constants::EARTH_RADIUS_KM;
    use crate::planet::Planet;
    use crate::sim::sim_op::SimOp;
    use crate::sim::{SimProps, Simulation};
    use h3o::{CellIndex, Resolution};

    #[test]
    fn creation() {
        let sim = Simulation::new(SimProps {
            planet: Planet {
                radius_km: EARTH_RADIUS_KM as f64,
                resolution: Resolution::One,
            },
            step_ops: vec![],
            start_ops: vec![],
            end_ops: vec![],
            res: Resolution::Two,
            layer_count: 4,
            layer_height: 10.0,
            layer_height_km: 10.0,
            sim_steps: 500,
            years_per_step: 1_000_000
        });

        assert_eq!(sim.cells.into_iter().len(), 5882);
    }

    #[test]
    fn simple_sim() {
        pub struct CoolingOp {
            pub intensity: f64,
        }

        impl CoolingOp {
            fn new(intensity: f64) -> CoolingOp {
                CoolingOp { intensity }
            }
            
            fn handle(intensity: f64)  -> SimOpHandle{
                SimOpHandle::new(Box::new(CoolingOp::new(intensity)))
            }
        }

        impl SimOp for CoolingOp {
            fn update_sim(&mut self, sim: &mut Simulation) {
                for column in sim.cells.values_mut() {
                    if let Some(cell) = column.layers_next.get_mut(0) {
                        cell.energy_joules *= self.intensity;
                    }
                }
            }
        }

        use crate::sim::sim_op::SimOpHandle;

        let mut sim = Simulation::new(SimProps {
            planet: Planet {
                radius_km: EARTH_RADIUS_KM as f64,
                resolution: Resolution::One,
            },
            step_ops: vec![CoolingOp::handle(0.99)],
            start_ops: vec![],
            end_ops: vec![],
            res: Resolution::Two,
            layer_count: 4,
            layer_height: 10.0,
            layer_height_km: 10.0,
            sim_steps: 500,
            years_per_step: 1_000_000
        });

        for (id, cell) in sim.cells.clone() {
            if let Some(layer) = cell.layers.first() {
                assert_abs_diff_eq!(layer.energy_joules,  5.47e23, epsilon= 5.0e22);
            }
        }

        sim.simulate();

        for (id, cell) in sim.cells {
            if let Some(layer) = cell.layers.first() {
                assert_abs_diff_eq!(layer.energy_joules,  3.5e21, epsilon= 5.0e20);
            }
        }

    }
}
