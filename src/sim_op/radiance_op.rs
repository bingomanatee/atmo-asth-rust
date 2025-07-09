/// Radiance operation for asthenosphere energy input
/// Integrates the radiance system as a sub-utility to determine upward energy
/// for the root of the asthenosphere using Perlin noise and thermal flows

use crate::sim_op::SimOp;
use crate::sim::simulation::Simulation;
use crate::sim::radiance::RadianceSystem;
use crate::global_thermal::global_h3_cell::GlobalH3Cell;
use crate::h3_utils::H3Utils;
use crate::energy_mass_composite::EnergyMassComposite;
use h3o::CellIndex;
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::Write;
use rayon::prelude::*;

/// Parameters for radiance operation
#[derive(Debug, Clone)]
pub struct RadianceOpParams {
    /// Base core radiance in J per km² per year (Earth default: 2.52e12)
    pub base_core_radiance_j_per_km2_per_year: f64,
    /// Multiplier for radiance system energy contribution (0.0 to 1.0)
    pub radiance_system_multiplier: f64,
    /// Base foundry temperature for deep thermal reservoir layers (Kelvin)
    pub foundry_temperature_k: f64,
    /// Enable detailed reporting
    pub enable_reporting: bool,
    /// Enable energy flow logging for debugging
    pub enable_energy_logging: bool,
}

impl Default for RadianceOpParams {
    fn default() -> Self {
        Self {
            base_core_radiance_j_per_km2_per_year: 2.52e12, // Earth's core radiance
            radiance_system_multiplier: 1.0,
            foundry_temperature_k: 2100.0, // Deep mantle temperature
            enable_reporting: false,
            enable_energy_logging: false,
        }
    }
}

/// Radiance operation for global thermal simulation
/// Injects thermal energy into the deepest asthenosphere layer using radiance system
pub struct RadianceOp {
    params: RadianceOpParams,
    radiance_system: RadianceSystem,
    total_energy_added: f64,
    cells_processed: usize,
    step_count: usize,
}

impl RadianceOp {
    pub fn new(params: RadianceOpParams, radiance_system: RadianceSystem) -> Self {
        // Initialize log file with header if logging is enabled
        if params.enable_energy_logging {
            if let Ok(mut file) = std::fs::OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open("layer_14_energy_flow.log")
            {
                writeln!(file, "# Layer 14 Energy Flow Debug Log").ok();
                writeln!(file, "# Step, Cell, Layer, Depth, CurrentEnergy, NextEnergy, DeltaEnergy, CurrentTemp, NextTemp, DeltaTemp, Phase").ok();
            }
        }

        Self {
            params,
            radiance_system,
            total_energy_added: 0.0,
            cells_processed: 0,
            step_count: 0,
        }
    }

    pub fn new_default() -> Self {
        // Create default radiance system with current year 0
        let radiance_system = RadianceSystem::new(0.0);
        Self::new(RadianceOpParams::default(), radiance_system)
    }

    pub fn new_with_radiance_system(radiance_system: RadianceSystem) -> Self {
        Self::new(RadianceOpParams::default(), radiance_system)
    }
    
    /// Apply radiance energy to all cells in the simulation
    pub fn apply(&mut self, cells: &mut HashMap<CellIndex, GlobalH3Cell>, time_years: f64, current_year: f64) {
        self.total_energy_added = 0.0;
        self.cells_processed = 0;
        self.step_count += 1;

        // Update radiance system for current year
        self.radiance_system.update(current_year);

        // Process cells sequentially (parallel processing has borrowing constraints with HashMap)
        for (cell_index, cell) in cells.iter_mut() {
            let energy_added = self.apply_to_cell(*cell_index, cell, time_years, current_year);
            self.total_energy_added += energy_added;
            self.cells_processed += 1;
        }

        if self.params.enable_reporting && self.step_count % 100 == 0 {
            println!("RadianceOp Step {}: Added {:.2e} J to {} cells", 
                self.step_count, self.total_energy_added, self.cells_processed);
        }
    }

    /// Apply radiance energy to a single cell
    fn apply_to_cell(
        &self,
        cell_index: CellIndex,
        cell: &mut GlobalH3Cell,
        time_years: f64,
        current_year: f64,
    ) -> f64 {
        // Calculate surface area
        let surface_area_km2 = H3Utils::cell_area(cell.planet.resolution, cell.planet.radius_km);

        // Find the foundry layer (single deepest asthenosphere layer as heat source)
        let foundry_layer_indices = self.find_foundry_layers(cell);

        if !foundry_layer_indices.is_empty() {
            // Calculate base core radiance energy
            let base_energy = self.params.base_core_radiance_j_per_km2_per_year * surface_area_km2 * time_years;

            // Calculate radiance system energy contribution
            let radiance_energy = self.calculate_radiance_system_energy(
                cell_index, surface_area_km2, time_years, current_year
            );

            // Combine base and radiance system energies
            let total_energy = base_energy + (radiance_energy * self.params.radiance_system_multiplier);

            // Add radiance energy directly to deepest foundry layer (natural heat source)
            let mut energy_distributed = 0.0;

            if !foundry_layer_indices.is_empty() {
                // Find the deepest (last) foundry layer to add energy to
                let deepest_layer_index = *foundry_layer_indices.last().unwrap();
                
                // Add the total energy directly to the deepest layer
                cell.layers_t[deepest_layer_index].1.add_energy(total_energy);
                energy_distributed = total_energy;

                if self.params.enable_reporting && self.step_count % 1000 == 0 {
                    let layer_temp = cell.layers_t[deepest_layer_index].1.temperature_k();
                    println!("  Cell {:?}: Added {:.2e}J to deepest layer (index {}, now {:.0}K)",
                        cell_index, total_energy, deepest_layer_index, layer_temp);
                }
            }

            // Log energy flow for layer 14 (just above foundry zone) if logging enabled
            if self.params.enable_energy_logging && foundry_layer_indices.len() > 0 {
                self.log_layer_14_energy_flow(cell_index, cell);
            }

            // Additional detailed reporting for debugging (moved above to specific layer reporting)

            energy_distributed
        } else {
            0.0
        }
    }

    /// Find the foundry layer (single deepest asthenosphere layer for energy injection)
    fn find_foundry_layers(&self, cell: &GlobalH3Cell) -> Vec<usize> {
        let mut foundry_layers = Vec::new();

        // Find all non-atmospheric layers (depth >= 0)
        let mut non_atmospheric_indices = Vec::new();
        for (index, (current_layer, _)) in cell.layers_t.iter().enumerate() {
            if current_layer.start_depth_km >= 0.0 {
                non_atmospheric_indices.push(index);
            }
        }

        // Take the bottom 1 layer as foundry layer (single deepest layer)
        let foundry_count = 1.min(non_atmospheric_indices.len());
        if foundry_count > 0 {
            let start_index = non_atmospheric_indices.len() - foundry_count;
            foundry_layers.extend_from_slice(&non_atmospheric_indices[start_index..]);
        }

        foundry_layers
    }

    /// Calculate energy contribution from the radiance system
    fn calculate_radiance_system_energy(
        &self,
        cell_index: CellIndex,
        surface_area_km2: f64,
        time_years: f64,
        current_year: f64,
    ) -> f64 {
        // Get neighbors for thermal flow calculations
        let neighbors = H3Utils::neighbors_for(cell_index);

        // Calculate radiance system energy per km² per year
        let radiance_energy_per_km2_per_year = self.radiance_system.calculate_cell_energy_with_neighbors(
            cell_index, current_year, &neighbors
        );

        // Convert to total energy for this cell and time period
        radiance_energy_per_km2_per_year * surface_area_km2 * time_years
    }

    /// Get total energy added in the last step
    pub fn total_energy_added(&self) -> f64 {
        self.total_energy_added
    }

    /// Get number of cells processed in the last step
    pub fn cells_processed(&self) -> usize {
        self.cells_processed
    }

    /// Get current step count
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Get reference to the radiance system
    pub fn radiance_system(&self) -> &RadianceSystem {
        &self.radiance_system
    }

    /// Get mutable reference to the radiance system
    pub fn radiance_system_mut(&mut self) -> &mut RadianceSystem {
        &mut self.radiance_system
    }

    /// Log energy flow for layer 14 (just above foundry zone) to track energy cliff
    fn log_layer_14_energy_flow(&self, cell_index: CellIndex, cell: &GlobalH3Cell) {
        // Layer 14 is typically the layer just above the foundry zone
        let layer_14_index = 14;

        if layer_14_index < cell.layers_t.len() {
            let (current_layer, next_layer) = &cell.layers_t[layer_14_index];

            // Calculate energy difference between current and next states
            let current_energy = current_layer.energy_mass.energy_joules;
            let next_energy = next_layer.energy_mass.energy_joules;
            let energy_delta = next_energy - current_energy;

            let current_temp = current_layer.temperature_k();
            let next_temp = next_layer.temperature_k();
            let temp_delta = next_temp - current_temp;

            // Log to file (append mode)
            if let Ok(mut file) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open("layer_14_energy_flow.log")
            {
                writeln!(file,
                    "Step:{}, Cell:{:?}, Layer:14, Depth:{:.1}km, CurrentE:{:.2e}J, NextE:{:.2e}J, DeltaE:{:.2e}J, CurrentT:{:.0}K, NextT:{:.0}K, DeltaT:{:.0}K, Phase:{:?}",
                    self.step_count,
                    cell_index,
                    current_layer.start_depth_km,
                    current_energy,
                    next_energy,
                    energy_delta,
                    current_temp,
                    next_temp,
                    temp_delta,
                    current_layer.phase()
                ).ok();
            }

            // Also log foundry layers for comparison
            for foundry_idx in [15, 16, 17] {
                if foundry_idx < cell.layers_t.len() {
                    let (foundry_current, foundry_next) = &cell.layers_t[foundry_idx];
                    let foundry_current_energy = foundry_current.energy_mass.energy_joules;
                    let foundry_next_energy = foundry_next.energy_mass.energy_joules;
                    let foundry_energy_delta = foundry_next_energy - foundry_current_energy;
                    let foundry_current_temp = foundry_current.temperature_k();
                    let foundry_next_temp = foundry_next.temperature_k();

                    if let Ok(mut file) = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open("layer_14_energy_flow.log")
                    {
                        writeln!(file,
                            "Step:{}, Cell:{:?}, Layer:{}, Depth:{:.1}km, CurrentE:{:.2e}J, NextE:{:.2e}J, DeltaE:{:.2e}J, CurrentT:{:.0}K, NextT:{:.0}K, DeltaT:{:.0}K, Phase:{:?} [FOUNDRY]",
                            self.step_count,
                            cell_index,
                            foundry_idx,
                            foundry_current.start_depth_km,
                            foundry_current_energy,
                            foundry_next_energy,
                            foundry_energy_delta,
                            foundry_current_temp,
                            foundry_next_temp,
                            foundry_next_temp - foundry_current_temp,
                            foundry_current.phase()
                        ).ok();
                    }
                }
            }
        }
    }
}

impl SimOp for RadianceOp {
    fn name(&self) -> &str {
        "RadianceOp"
    }

    fn update_sim(&mut self, sim: &mut Simulation) {
        let time_years = sim.years_per_step as f64;
        let current_year = sim.current_step() as f64 * time_years;
        
        self.apply(&mut sim.cells, time_years, current_year);
    }
}

/// Convenience function to create and apply radiance operation
pub fn apply_radiance(
    cells: &mut HashMap<CellIndex, GlobalH3Cell>,
    time_years: f64,
    current_year: f64,
    params: Option<RadianceOpParams>,
    radiance_system: Option<RadianceSystem>,
) -> f64 {
    let radiance_sys = radiance_system.unwrap_or_else(|| RadianceSystem::new(0.0));
    let mut op = RadianceOp::new(params.unwrap_or_default(), radiance_sys);
    op.apply(cells, time_years, current_year);
    op.total_energy_added()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planet::Planet;
    use crate::global_thermal::global_h3_cell::GlobalH3CellConfig;
    use h3o::{Resolution, CellIndex};
    use std::sync::Arc;

    #[test]
    fn test_radiance_op_adds_energy_to_bottom_layer() {
        // Create test planet
        let planet = Arc::new(Planet::earth(Resolution::Two));

        // Create test cell
        let h3_index = CellIndex::try_from(0x821c07fffffffff).unwrap();
        let config = GlobalH3CellConfig::new_earth_like(h3_index, planet.clone());
        let mut cell = GlobalH3Cell::new_with_schedule(h3_index, planet, &config.layer_schedule);

        // Find bottom asthenosphere layer
        let bottom_layer_index = cell.layers_t.iter().enumerate().rev()
            .find(|(_, (layer, _))| layer.start_depth_km >= 0.0)
            .map(|(index, _)| index)
            .expect("Should have at least one non-atmospheric layer");

        // Record initial energy
        let initial_energy = cell.layers_t[bottom_layer_index].0.energy_mass.energy();

        // Create radiance operation
        let mut radiance_op = RadianceOp::new_default();
        
        // Apply to single cell
        let mut cells = HashMap::new();
        cells.insert(h3_index, cell);
        
        radiance_op.apply(&mut cells, 1000.0, 0.0); // 1000 years

        // Verify energy was added
        let cell = cells.get(&h3_index).unwrap();
        let final_energy = cell.layers_t[bottom_layer_index].1.energy_mass.energy();
        let energy_added = final_energy - initial_energy;

        assert!(energy_added > 0.0, "RadianceOp should add energy! Added: {:.2e} J", energy_added);
        assert!(radiance_op.total_energy_added() > 0.0, "Total energy added should be positive");
        assert_eq!(radiance_op.cells_processed(), 1, "Should process exactly one cell");
    }
}
