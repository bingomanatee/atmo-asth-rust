/// Heat redistribution operation using Fourier thermal transfer
/// Implements science-backed heat diffusion between layers within each cell
/// Based on the logic from wide_experiment and wide_experiment_with_atmosphere

use crate::sim_op::SimOp;
use crate::sim::simulation::Simulation;
use crate::sim::fourier_thermal_transfer::FourierThermalTransfer;
use crate::constants::SECONDS_PER_YEAR;
use rayon::prelude::*;
use h3o::CellIndex;

pub struct HeatRedistributionOp {
    pub apply_during_simulation: bool,
    pub debug_output: bool,

    current_step: usize,

    // Fourier thermal transfer utility
    fourier_transfer: Option<FourierThermalTransfer>,
}

impl HeatRedistributionOp {
    pub fn new() -> Self {
        Self {
            apply_during_simulation: true,
            debug_output: false,
            current_step: 0,
            fourier_transfer: None,
        }
    }

    pub fn with_debug() -> Self {
        Self {
            apply_during_simulation: true,
            debug_output: true,
            current_step: 0,
            fourier_transfer: None,
        }
    }

    /// Output temperature changes every 20 steps - comparing current vs next states
    fn output_temperature_changes(&mut self, _sim: &Simulation, step: usize) {
        if step % 20 != 0 {
            return;
        }

        // Heat transfer results output removed
    }



    /// Apply heat redistribution to a single cell
    fn redistribute_heat_in_cell(
        &self,
        cell: &mut crate::global_thermal::global_h3_cell::GlobalH3Cell,
    ) -> f64 {

        
        let mut total_energy_transferred = 0.0;

        if cell.layers_t.len() < 2 {
            return 0.0; // Need at least 2 layers for heat transfer
        }

        // Apply heat transfer between adjacent layers using clean indexed iteration
        for i in 0..(cell.layers_t.len() - 1) {
            let energy_transferred = if let Some(ref fourier) = self.fourier_transfer {
                // Split at the boundary to get mutable references to adjacent layers
                let (upper_part, lower_part) = cell.layers_t.split_at_mut(i + 1);
                fourier.transfer_heat_between_layer_tuples(
                    &mut upper_part[i],
                    &mut lower_part[0], // This is layer i+1
                )
            } else {
                0.0 // Fallback if Fourier transfer not initialized
            };

            total_energy_transferred += energy_transferred;
        }
        
        total_energy_transferred
    }
}

impl SimOp for HeatRedistributionOp {
    fn name(&self) -> &str {
        "HeatRedistribution"
    }
    
    fn init_sim(&mut self, sim: &mut Simulation) {
        // Initialize Fourier thermal transfer utility
        self.fourier_transfer = Some(FourierThermalTransfer::new(sim.years_per_step as f64));
    }
    
    fn update_sim(&mut self, sim: &mut Simulation) {
        if !self.apply_during_simulation {
            return;
        }

        // Increment step counter and output temperature changes every 20 steps
        self.current_step += 1;
        self.output_temperature_changes(sim, self.current_step);

        let _years = sim.years_per_step as f64;
        let mut _total_energy_transferred = 0.0;
        let mut _cells_processed = 0;
        
        // Heat redistribution step starting - optimized sequential processing
        
        // Convert HashMap to Vec to enable parallel processing
        let mut cells_vec: Vec<_> = sim.cells.drain().collect();
        
        // Process cells in parallel (most expensive operation)
        cells_vec.par_iter_mut().for_each(|(_, cell)| {
            let _energy_transferred = self.redistribute_heat_in_cell(cell);
            // Note: individual energy tracking removed for parallel efficiency
        });
        
        // Reconstruct HashMap 
        _cells_processed = cells_vec.len();
        for (cell_index, cell) in cells_vec {
            sim.cells.insert(cell_index, cell);
        }
        
        _total_energy_transferred = 0.0; // Energy tracking removed for parallel efficiency
        
        // Heat redistribution complete
    }
}
