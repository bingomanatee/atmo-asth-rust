/// Pressure adjustment operation
/// Applies pressure compaction to all layers based on overlying mass

use crate::sim::sim_op::SimOp;
use crate::sim::simulation::Simulation;

pub struct PressureAdjustmentOp {
    pub apply_during_simulation: bool,
}

impl PressureAdjustmentOp {
    pub fn new() -> Self {
        Self {
            apply_during_simulation: false, // Usually only needed during init
        }
    }
    
    pub fn continuous() -> Self {
        Self {
            apply_during_simulation: true,
        }
    }
}

impl SimOp for PressureAdjustmentOp {
    fn name(&self) -> &str {
        "PressureAdjustment"
    }

    fn init_sim(&mut self, sim: &mut Simulation) {
        println!("🔧 Applying pressure compaction during initialization...");

        let mut total_cells_processed = 0;
        let mut total_pressure_applied = 0.0;
        let mut max_pressure: f64 = 0.0;

        for cell in sim.cells.values_mut() {
            // Apply pressure compaction to this cell (handles tuple structure internally)
            cell.apply_pressure_compaction();

            // Collect statistics from current state
            let pressures = cell.calculate_layer_pressures();
            let cell_max_pressure = pressures.iter().fold(0.0f64, |a, &b| a.max(b));
            let cell_total_pressure: f64 = pressures.iter().sum();

            total_pressure_applied += cell_total_pressure;
            max_pressure = max_pressure.max(cell_max_pressure);
            total_cells_processed += 1;

            // Debug output removed for cleaner simulation output
        }

        println!("✅ Pressure compaction complete:");
        println!("   - {} cells processed", total_cells_processed);
        println!("   - Max pressure: {:.2e} Pa ({:.1} GPa)", max_pressure, max_pressure / 1e9);
        println!("   - Avg pressure per cell: {:.2e} Pa",
                 total_pressure_applied / total_cells_processed as f64);
    }
    
    fn update_sim(&mut self, sim: &mut Simulation) {
        if !self.apply_during_simulation {
            return;
        }

        if sim.debug {
            println!("🔧 Applying pressure compaction at step {}...", sim.step);
        }

        for cell in sim.cells.values_mut() {
            cell.apply_pressure_compaction();
        }
    }
}
