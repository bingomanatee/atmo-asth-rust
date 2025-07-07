/// Pressure adjustment operation
/// Applies pressure compaction to all layers based on overlying mass

use crate::sim_op::SimOp;
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
        // Applying pressure compaction during initialization

        let mut _total_cells_processed = 0;
        let mut _total_pressure_applied = 0.0;
        let mut max_pressure: f64 = 0.0;

        for cell in sim.cells.values_mut() {
            // Apply pressure compaction to this cell (handles tuple structure internally)
            cell.apply_pressure_compaction();

            // Collect statistics from current state
            let pressures = cell.calculate_layer_pressures();
            let cell_max_pressure = pressures.iter().fold(0.0f64, |a, &b| a.max(b));
            let cell_total_pressure: f64 = pressures.iter().sum();

            _total_pressure_applied += cell_total_pressure;
            max_pressure = max_pressure.max(cell_max_pressure);
            _total_cells_processed += 1;

            // Debug output removed for cleaner simulation output
        }

        // Pressure compaction complete
    }
    
    fn update_sim(&mut self, sim: &mut Simulation) {
        if !self.apply_during_simulation {
            return;
        }

        // Pressure compaction step

        for cell in sim.cells.values_mut() {
            cell.apply_pressure_compaction();
        }
    }
}
