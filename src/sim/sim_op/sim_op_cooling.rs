use crate::sim::sim_op::{SimOp, SimOpHandle};
use crate::sim::Simulation;
use crate::temp_utils::cooling_per_cell_per_year;

/// This op is @deprectaed 
/// cooling happens now in radiance and atmosphere
/// 
pub struct CoolingOp {
    cooling_scale: f64
}

impl CoolingOp {
    pub fn new(cooling_scale: f64) -> CoolingOp {
        CoolingOp {
            cooling_scale
        }
    }

    pub fn handle(cooling_scale: f64) -> SimOpHandle {
        SimOpHandle::new(Box::new(CoolingOp::new(cooling_scale)))
    }
}

impl SimOp for CoolingOp {
    fn name(&self) -> &str {
        "CoolingOp"
    }

    fn update_sim(&mut self, sim: &mut Simulation) {
        for column in sim.cells.values_mut() {
            let lithosphere_km = column.total_lithosphere_height_next();
            let (_, next_layer) = column.layer(0);

            let cooling_per_year_per_cell = cooling_per_cell_per_year(sim.resolution, sim.planet.radius_km, lithosphere_km);
            let cooling = cooling_per_year_per_cell * sim.years_per_step as f64;

            next_layer.remove_energy(cooling * self.cooling_scale);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::CoolingOp;
    use approx::assert_abs_diff_eq;
    use crate::constants::{ASTHENOSPHERE_SURFACE_START_TEMP_K, EARTH_RADIUS_KM};
    use crate::planet::Planet;
    use crate::sim::sim_op::SimOpHandle;
    use crate::sim::{SimProps, Simulation};
    use h3o::Resolution;
    #[test]
    fn just_chillin() {
        let mut sim = Simulation::new(SimProps {
            name: "cooling_op_test",
            planet: Planet {
                radius_km: EARTH_RADIUS_KM as f64,
                resolution: Resolution::One,
            },
            ops: vec![CoolingOp::handle(1.0)],
            res: Resolution::Two,
            layer_count: 4,
            layer_height_km: 10.0,
            sim_steps: 10,
            years_per_step: 100_000,
            debug: true,
            alert_freq: 1,
            starting_surface_temp_k: ASTHENOSPHERE_SURFACE_START_TEMP_K
        });

        for (_id, cell) in sim.cells.clone() {
            if let Some(layer) = cell.asth_layers.first() {
                assert_abs_diff_eq!(layer.energy_joules(),  6.04e23, epsilon= 5.0e22);
            }
        }

        sim.simulate();

        for (_id, cell) in sim.cells {
            if let Some(layer) = cell.asth_layers.first() {
                assert_abs_diff_eq!(layer.energy_joules(),  3.49e23, epsilon= 5.0e21);
            }
        }
    }

    #[test]
    fn test_cooling_varies_with_lithosphere_thickness() {
        use crate::material::MaterialType;
        use crate::asth_cell::AsthCellLithosphere;
        use crate::sim::sim_op::SimOp;

        // Create two simulations - one with no lithosphere, one with thick lithosphere
        let mut sim_no_lithosphere = Simulation::new(SimProps {
            name: "no_lithosphere_test",
            planet: Planet {
                radius_km: EARTH_RADIUS_KM as f64,
                resolution: Resolution::One,
            },
            ops: vec![],
            res: Resolution::Two,
            layer_count: 4,
            layer_height_km: 10.0,
            sim_steps: 1,
            years_per_step: 1000,
            debug: false,
            alert_freq: 1,
            starting_surface_temp_k: ASTHENOSPHERE_SURFACE_START_TEMP_K
        });

        let mut sim_thick_lithosphere = Simulation::new(SimProps {
            name: "thick_lithosphere_test",
            planet: Planet {
                radius_km: EARTH_RADIUS_KM as f64,
                resolution: Resolution::One,
            },
            ops: vec![],
            res: Resolution::Two,
            layer_count: 4,
            layer_height_km: 10.0,
            sim_steps: 1,
            years_per_step: 1000,
            debug: false,
            alert_freq: 1,
            starting_surface_temp_k: ASTHENOSPHERE_SURFACE_START_TEMP_K
        });

        // Add thick lithosphere to second simulation
        for column in sim_thick_lithosphere.cells.values_mut() {
            column.lithospheres_next.clear();
            column.lithospheres_next.push(AsthCellLithosphere::new(
                50.0, // 50 km thick lithosphere
                MaterialType::Silicate,
                column.default_volume(),
            ));
        }

        let mut cooling_op_no_lith = CoolingOp::new(1.0);
        let mut cooling_op_thick_lith = CoolingOp::new(1.0);

        // Record initial energies
        let initial_energy_no_lith: f64 = sim_no_lithosphere.cells.values()
            .map(|column| column.asth_layers[0].energy_joules())
            .sum();
        let initial_energy_thick_lith: f64 = sim_thick_lithosphere.cells.values()
            .map(|column| column.asth_layers[0].energy_joules())
            .sum();

        // Apply cooling to both simulations
        cooling_op_no_lith.update_sim(&mut sim_no_lithosphere);
        cooling_op_thick_lith.update_sim(&mut sim_thick_lithosphere);

        // Calculate energy loss
        let final_energy_no_lith: f64 = sim_no_lithosphere.cells.values()
            .map(|column| column.asth_layers_next[0].energy_joules())
            .sum();
        let final_energy_thick_lith: f64 = sim_thick_lithosphere.cells.values()
            .map(|column| column.asth_layers_next[0].energy_joules())
            .sum();

        let energy_loss_no_lith = initial_energy_no_lith - final_energy_no_lith;
        let energy_loss_thick_lith = initial_energy_thick_lith - final_energy_thick_lith;

        // Simulation with no lithosphere should lose more energy than one with thick lithosphere
        assert!(energy_loss_no_lith > energy_loss_thick_lith,
                "No lithosphere should cool faster than thick lithosphere. No lith loss: {:.2e}, Thick lith loss: {:.2e}",
                energy_loss_no_lith, energy_loss_thick_lith);

        // Both should lose some energy (positive loss)
        assert!(energy_loss_no_lith > 0.0, "Should lose energy when cooling");
        assert!(energy_loss_thick_lith > 0.0, "Should lose some energy even with thick lithosphere");
    }
}
