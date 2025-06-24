use crate::sim::sim_op::{SimOp, SimOpHandle};
use crate::sim::Simulation;
use crate::temp_utils::cooling_per_cell_per_year;

pub struct CoolingOp;

impl CoolingOp {
    pub fn new() -> CoolingOp {
        CoolingOp
    }

    pub fn handle() -> SimOpHandle {
        SimOpHandle::new(Box::new(CoolingOp::new()))
    }
}

impl SimOp for CoolingOp {
    fn update_sim(&mut self, sim: &mut Simulation) {
        for column in sim.cells.values_mut() {
            if let Some(cell) = column.layers_next.get_mut(0) {
                let lithosphere_km = column.lithosphere_next.height_km;
                let cooling_per_year_per_cell = cooling_per_cell_per_year(sim.resolution, sim.planet.radius_km, lithosphere_km);
                let cooling = cooling_per_year_per_cell * sim.years_per_step as f64;

                cell.energy_joules = (cell.energy_joules - cooling).max(0.0);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::CoolingOp;
    use approx::assert_abs_diff_eq;
    use crate::constants::EARTH_RADIUS_KM;
    use crate::planet::Planet;
    use crate::sim::sim_op::SimOpHandle;
    use crate::sim::{SimProps, Simulation};
    use h3o::Resolution;
    #[test]
    fn just_chillin() {
        let mut sim = Simulation::new(SimProps {
            planet: Planet {
                radius_km: EARTH_RADIUS_KM as f64,
                resolution: Resolution::One,
            },
            step_ops: vec![CoolingOp::handle()],
            start_ops: vec![],
            end_ops: vec![],
            res: Resolution::Two,
            layer_count: 4,
            layer_height: 10.0,
            layer_height_km: 10.0,
            sim_steps: 50,
            years_per_step: 100_000,
        });

        for (_id, cell) in sim.cells.clone() {
            if let Some(layer) = cell.layers.first() {
                assert_abs_diff_eq!(layer.energy_joules,  5.47e23, epsilon= 5.0e22);
            }
        }

        sim.simulate();

        for (_id, cell) in sim.cells {
            if let Some(layer) = cell.layers.first() {
                assert_abs_diff_eq!(layer.energy_joules,  3.5e21, epsilon= 5.0e20);
            }
        }

    }
}
