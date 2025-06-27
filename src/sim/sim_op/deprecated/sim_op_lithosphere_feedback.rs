use crate::sim::sim_op::{SimOp, SimOpHandle};
use crate::sim::Simulation;

/// Lithosphere Feedback Heating Operator
/// 
/// This operator implements a feedback mechanism where accumulated lithosphere
/// triggers increased heat input to prevent runaway lithosphere formation.
/// 
/// The mechanism works by:
/// 1. Tracking total lithosphere thickness across all cells
/// 2. When lithosphere exceeds a threshold, adding heat proportional to the excess
/// 3. This heat addition reduces further lithosphere formation by raising temperatures
pub struct LithosphereFeedbackOp {
    /// Global lithosphere thickness threshold (km) that triggers feedback heating
    pub global_threshold_km: f64,
    
    /// Local lithosphere thickness threshold (km) per cell that triggers feedback heating
    pub local_threshold_km: f64,
    
    /// Heat addition factor - multiplier for excess lithosphere to determine heat added
    /// Units: Joules per km of excess lithosphere per cell per year
    pub heat_factor_j_per_km_per_year: f64,
    
    /// Whether to use global threshold (true) or local threshold (false)
    pub use_global_threshold: bool,
}

impl LithosphereFeedbackOp {
    /// Create a new lithosphere feedback operator
    /// 
    /// # Arguments
    /// * `global_threshold_km` - Global lithosphere threshold in km
    /// * `local_threshold_km` - Local lithosphere threshold in km per cell
    /// * `heat_factor_j_per_km_per_year` - Heat addition per km excess per year
    /// * `use_global_threshold` - Whether to use global (true) or local (false) thresholds
    pub fn new(
        global_threshold_km: f64,
        local_threshold_km: f64,
        heat_factor_j_per_km_per_year: f64,
        use_global_threshold: bool,
    ) -> Self {
        Self {
            global_threshold_km,
            local_threshold_km,
            heat_factor_j_per_km_per_year,
            use_global_threshold,
        }
    }

    /// Create a handle for the lithosphere feedback operator
    pub fn handle(
        global_threshold_km: f64,
        local_threshold_km: f64,
        heat_factor_j_per_km_per_year: f64,
        use_global_threshold: bool,
    ) -> SimOpHandle {
        SimOpHandle::new(Box::new(Self::new(
            global_threshold_km,
            local_threshold_km,
            heat_factor_j_per_km_per_year,
            use_global_threshold,
        )))
    }

    /// Calculate total lithosphere thickness across all cells
    fn calculate_global_lithosphere_thickness(&self, sim: &Simulation) -> f64 {
        sim.cells
            .values()
            .map(|column| column.total_lithosphere_height_next())
            .sum()
    }

    /// Calculate average lithosphere thickness per cell
    fn calculate_average_lithosphere_thickness(&self, sim: &Simulation) -> f64 {
        let total = self.calculate_global_lithosphere_thickness(sim);
        let cell_count = sim.cells.len() as f64;
        if cell_count > 0.0 {
            total / cell_count
        } else {
            0.0
        }
    }
}

impl SimOp for LithosphereFeedbackOp {
    fn name(&self) -> &str {
        "LithosphereFeedbackOp"
    }

    fn update_sim(&mut self, sim: &mut Simulation) {
        if self.use_global_threshold {
            // Global threshold mode: check average lithosphere thickness
            let avg_thickness = self.calculate_average_lithosphere_thickness(sim);
            
            if avg_thickness > self.global_threshold_km {
                let excess_km = avg_thickness - self.global_threshold_km;
                let heat_per_cell_per_year = self.heat_factor_j_per_km_per_year * excess_km;
                let heat_per_cell_per_step = heat_per_cell_per_year * sim.years_per_step as f64;
                
                // Add heat to all cells proportionally
                for column in sim.cells.values_mut() {
                    let (_, next_layer) = column.layer(0);
                    next_layer.add_energy(heat_per_cell_per_step);
                }
            }
        } else {
            // Local threshold mode: check each cell individually
            for column in sim.cells.values_mut() {
                let local_thickness = column.total_lithosphere_height_next();
                
                if local_thickness > self.local_threshold_km {
                    let excess_km = local_thickness - self.local_threshold_km;
                    let heat_per_year = self.heat_factor_j_per_km_per_year * excess_km;
                    let heat_per_step = heat_per_year * sim.years_per_step as f64;
                    
                    // Add heat to this specific cell
                    let (_, next_layer) = column.layer(0);
                    next_layer.add_energy(heat_per_step);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::{ASTHENOSPHERE_SURFACE_START_TEMP_K, EARTH_RADIUS_KM};
    use crate::planet::Planet;
    use crate::sim::{SimProps, Simulation};
    use crate::material::MaterialType;
    use crate::asth_cell::AsthCellLithosphere;
    use h3o::Resolution;
    use approx::assert_abs_diff_eq;

    fn create_test_simulation() -> Simulation {
        Simulation::new(SimProps {
            name: "lithosphere_feedback_test",
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
            starting_surface_temp_k: ASTHENOSPHERE_SURFACE_START_TEMP_K,
        })
    }

    #[test]
    fn test_global_threshold_no_feedback() {
        let mut sim = create_test_simulation();
        let mut op = LithosphereFeedbackOp::new(
            10.0, // global threshold
            5.0,  // local threshold (unused)
            1e20, // heat factor
            true, // use global threshold
        );

        // Record initial energy
        let initial_energy: f64 = sim.cells.values()
            .map(|column| column.asth_layers[0].energy_joules())
            .sum();

        // Run the operator (should not add heat since no lithosphere exists)
        op.update_sim(&mut sim);

        // Energy should remain the same
        let final_energy: f64 = sim.cells.values()
            .map(|column| column.asth_layers_next[0].energy_joules())
            .sum();

        assert_abs_diff_eq!(initial_energy, final_energy, epsilon = 1e10);
    }

    #[test]
    fn test_global_threshold_with_feedback() {
        let mut sim = create_test_simulation();
        let mut op = LithosphereFeedbackOp::new(
            5.0,  // global threshold
            10.0, // local threshold (unused)
            1e20, // heat factor
            true, // use global threshold
        );

        // Add lithosphere to exceed global threshold
        for column in sim.cells.values_mut() {
            column.lithospheres_next.clear();
            column.lithospheres_next.push(AsthCellLithosphere::new(
                10.0, // 10 km height > 5 km threshold
                MaterialType::Silicate,
                column.default_volume(),
            ));
        }

        // Record initial energy
        let initial_energy: f64 = sim.cells.values()
            .map(|column| column.asth_layers[0].energy_joules())
            .sum();

        // Run the operator (should add heat due to excess lithosphere)
        op.update_sim(&mut sim);

        // Energy should increase
        let final_energy: f64 = sim.cells.values()
            .map(|column| column.asth_layers_next[0].energy_joules())
            .sum();

        assert!(final_energy > initial_energy, 
                "Energy should increase due to feedback heating");
    }

    #[test]
    fn test_local_threshold_with_feedback() {
        let mut sim = create_test_simulation();
        let mut op = LithosphereFeedbackOp::new(
            100.0, // global threshold (unused)
            5.0,   // local threshold
            1e20,  // heat factor
            false, // use local threshold
        );

        // Add lithosphere to first cell only to exceed local threshold
        let first_cell_id = *sim.cells.keys().next().unwrap();
        let column = sim.cells.get_mut(&first_cell_id).unwrap();
        column.lithospheres_next.clear();
        column.lithospheres_next.push(AsthCellLithosphere::new(
            10.0, // 10 km height > 5 km threshold
            MaterialType::Silicate,
            column.default_volume(),
        ));

        // Record initial energy of the first cell
        let initial_energy = sim.cells[&first_cell_id].asth_layers[0].energy_joules();

        // Run the operator
        op.update_sim(&mut sim);

        // Energy of the first cell should increase
        let final_energy = sim.cells[&first_cell_id].asth_layers_next[0].energy_joules();

        assert!(final_energy > initial_energy,
                "Energy should increase in cell with excess lithosphere");
    }
}
