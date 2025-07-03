/// Heat redistribution operation using Fourier thermal transfer
/// Implements science-backed heat diffusion between layers within each cell
/// Based on the logic from wide_experiment and wide_experiment_with_atmosphere

use crate::sim::sim_op::SimOp;
use crate::sim::simulation::Simulation;
use crate::sim::fourier_thermal_transfer::FourierThermalTransfer;
use crate::constants::{SECONDS_PER_YEAR, M2_PER_KM2};
use crate::energy_mass_composite::EnergyMassComposite;
use h3o::CellIndex;
use std::collections::HashMap;

/// Physical constants for Fourier heat transfer calculations
mod fourier_constants {
    /// Minimum temperature difference for heat transfer (K)
    pub const MIN_TEMP_DIFF_K: f64 = 0.1;

    /// Maximum energy transfer fraction per timestep for numerical stability
    /// Based on wide experiment: use half the energy difference between neighboring cells
    pub const MAX_ENERGY_TRANSFER_FRACTION: f64 = 0.5;

    /// Conversion from km to m
    pub const KM_TO_M: f64 = 1000.0;
}

pub struct HeatRedistributionOp {
    pub apply_during_simulation: bool,
    pub debug_output: bool,

    // Pre-computed constants for performance
    time_seconds_per_step: f64,
    surface_area_m2: f64, // Identical for all cells
    layer_distances_m: Vec<f64>, // Distance between adjacent layers

    // Step tracking for temperature profile output
    current_step: usize,

    // Previous temperatures for tracking changes
    previous_temperatures: Vec<f64>,

    // Fourier thermal transfer utility
    fourier_transfer: Option<FourierThermalTransfer>,
}

impl HeatRedistributionOp {
    pub fn new() -> Self {
        Self {
            apply_during_simulation: true,
            debug_output: false,
            time_seconds_per_step: 0.0,
            surface_area_m2: 0.0,
            layer_distances_m: Vec::new(),
            current_step: 0,
            previous_temperatures: Vec::new(),
            fourier_transfer: None,
        }
    }

    pub fn with_debug() -> Self {
        Self {
            apply_during_simulation: true,
            debug_output: true,
            time_seconds_per_step: 0.0,
            surface_area_m2: 0.0,
            layer_distances_m: Vec::new(),
            current_step: 0,
            previous_temperatures: Vec::new(),
            fourier_transfer: None,
        }
    }
    


    /// Output temperature changes every 20 steps - comparing current vs next states
    fn output_temperature_changes(&mut self, sim: &Simulation, step: usize) {
        if step % 20 != 0 {
            return;
        }

        // Heat transfer results output removed
    }

    /// Output heat transfer amounts every 20 steps
    fn output_heat_transfer_analysis(&self, sim: &Simulation, step: usize, total_energy_transferred: f64) {
        if step % 20 != 0 {
            return;
        }

        // Heat transfer analysis output removed
    }

    /// Apply heat redistribution to a single cell
    fn redistribute_heat_in_cell(
        &self,
        cell: &mut crate::global_thermal::global_h3_cell::GlobalH3Cell,
    ) -> f64 {
        use fourier_constants::*;
        
        let mut total_energy_transferred = 0.0;
        let layer_count = cell.layers_t.len();
        
        if layer_count < 2 {
            return 0.0; // Need at least 2 layers for heat transfer
        }
        
        // Apply heat transfer between adjacent layers
        for i in 0..(layer_count - 1) {
            let (upper_temp, upper_conductivity, upper_height, lower_temp, lower_conductivity, lower_height) = {
                let upper_layer = &cell.layers_t[i].0; // current state
                let lower_layer = &cell.layers_t[i + 1].0; // current state
                
                (
                    upper_layer.temperature_k(),
                    upper_layer.thermal_conductivity(),
                    upper_layer.height_km,
                    lower_layer.temperature_k(),
                    lower_layer.thermal_conductivity(),
                    lower_layer.height_km,
                )
            };
            
            // Use pre-computed distance between layer centers
            let distance_m = self.layer_distances_m[i];

            // Calculate heat flow from upper to lower layer
            // Calculate and apply heat flow using updated Fourier thermal transfer utility
            // Use split_at_mut to avoid borrowing conflicts
            let energy_transferred = if let Some(ref fourier) = self.fourier_transfer {
                let (upper_layers, lower_layers) = cell.layers_t.split_at_mut(i + 1);
                fourier.apply_heat_transfer_between_layers(
                    &mut upper_layers[i],
                    &mut lower_layers[0], // This is cell.layers_t[i + 1]
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

        // Pre-compute time conversion factor
        self.time_seconds_per_step = sim.years_per_step as f64 * SECONDS_PER_YEAR;

        // Initialize Fourier thermal transfer utility
        self.fourier_transfer = Some(FourierThermalTransfer::new(self.time_seconds_per_step));

        // Get surface area from any cell (all identical)
        if let Some(first_cell) = sim.cells.values().next() {
            self.surface_area_m2 = first_cell.surface_area_km2() * M2_PER_KM2;

            // Pre-compute layer distances (uniform layer structure)
            for i in 0..(first_cell.layers_t.len() - 1) {
                let upper_height = first_cell.layers_t[i].0.height_km;
                let lower_height = first_cell.layers_t[i + 1].0.height_km;
                let distance_m = (upper_height + lower_height) * 0.5 * fourier_constants::KM_TO_M;
                self.layer_distances_m.push(distance_m);
            }
        }

        // Heat redistribution constants pre-computed
    }
    
    fn update_sim(&mut self, sim: &mut Simulation) {
        if !self.apply_during_simulation {
            return;
        }

        // Increment step counter and output temperature changes every 20 steps
        self.current_step += 1;
        self.output_temperature_changes(sim, self.current_step);

        let years = sim.years_per_step as f64;
        let mut total_energy_transferred = 0.0;
        let mut cells_processed = 0;
        
        // Heat redistribution step starting
        
        for cell in sim.cells.values_mut() {
            let cell_energy_transferred = self.redistribute_heat_in_cell(cell);
            total_energy_transferred += cell_energy_transferred;
            cells_processed += 1;
            
            // Cell processing debug output removed
        }
        
        // Heat redistribution complete
    }
}
