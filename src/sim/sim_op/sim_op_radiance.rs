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
    fn update_sim(&mut self, sim: &mut Simulation) {
        for column in sim.cells.values_mut() {
            let lithosphere_height = column.total_lithosphere_height();
            if let Some(cell) = column.layers_next.get_mut(0) {
                let radiance_per_year = radiance_per_cell_per_year(sim.resolution, sim.planet.radius_km, lithosphere_height);
                let radiance = radiance_per_year * sim.years_per_step as f64;
                cell.energy_joules = (cell.energy_joules + radiance).max(0.0);
            }
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
            layer_height: 10.0,
            layer_height_km: 10.0,
            sim_steps: 10,
            years_per_step: 100_000,
            debug: true,
            alert_freq: 1,
            starting_surface_temp_k: ASTHENOSPHERE_SURFACE_START_TEMP_K,
        });

        for (_id, cell) in sim.cells.clone() {
            if let Some(layer) = cell.layers.first() {
                assert_abs_diff_eq!(layer.energy_joules,  6.04e23, epsilon= 5.0e22);
            }
        }

        sim.simulate();

        for (_id, cell) in sim.cells {
            if let Some(layer) = cell.layers.first() {
                assert_abs_diff_eq!(layer.energy_joules, 8.25e23, epsilon= 5.0e21);
            }
        }

    }
}
