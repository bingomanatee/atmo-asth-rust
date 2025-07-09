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

/// Cached energy values for hotspot lifecycle phases
#[derive(Debug, Clone)]
struct HotspotEnergyCache {
    pub half_peak: f64,      // Energy at 1/2 peak
    pub peak: f64,           // Energy at peak
    pub peak_plus_half: f64, // Energy at peak + 1/2 peak  
    pub mid_decline: f64,    // Energy halfway between peak+1/2 and end
}

/// Cached perlin noise values for cell configurations
#[derive(Debug, Clone)]
struct PerlinCache {
    pub cell_perlin_values: HashMap<CellIndex, f64>,
}

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
    perlin_cache: PerlinCache,
    hotspot_energy_cache: HashMap<CellIndex, HotspotEnergyCache>,
    energy_accumulator: f64,  // Accumulate energy for 100-year batches
    years_since_last_injection: f64,
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

        let mut op = Self {
            params,
            radiance_system,
            total_energy_added: 0.0,
            cells_processed: 0,
            step_count: 0,
            perlin_cache: PerlinCache {
                cell_perlin_values: HashMap::new(),
            },
            hotspot_energy_cache: HashMap::new(),
            energy_accumulator: 0.0,
            years_since_last_injection: 0.0,
        };
        
        // Pre-compute perlin cache for all cells
        op.initialize_perlin_cache();
        
        op
    }

    pub fn new_default() -> Self {
        // Create default radiance system with current year 0
        let radiance_system = RadianceSystem::new(0.0);
        Self::new(RadianceOpParams::default(), radiance_system)
    }

    pub fn new_with_radiance_system(radiance_system: RadianceSystem) -> Self {
        Self::new(RadianceOpParams::default(), radiance_system)
    }
    
    /// Initialize perlin cache for all cells at resolution
    fn initialize_perlin_cache(&mut self) {
        // Cache perlin values for all H3 cells at resolution 2 (typical simulation resolution)
        let resolution = h3o::Resolution::Two;
        let mut cells: Vec<_> = h3o::CellIndex::base_cells()
            .flat_map(|base| base.children(resolution))
            .collect();
            
        for cell_index in cells {
            let perlin_energy = self.radiance_system.calculate_perlin_energy_at(cell_index);
            self.perlin_cache.cell_perlin_values.insert(cell_index, perlin_energy);
        }
        
        println!("🔥 RadianceOp: Cached perlin values for {} cells", self.perlin_cache.cell_perlin_values.len());
    }
    
    /// Get cached perlin energy for a cell
    fn get_cached_perlin_energy(&self, cell_index: CellIndex) -> f64 {
        *self.perlin_cache.cell_perlin_values.get(&cell_index).unwrap_or(&0.0)
    }
    
    /// Pre-compute hotspot energy cache for a cell
    fn cache_hotspot_energy(&mut self, cell_index: CellIndex, surface_area_km2: f64, time_years: f64) {
        if !self.hotspot_energy_cache.contains_key(&cell_index) {
            // Calculate energy at key lifecycle phases
            let neighbors = H3Utils::neighbors_for(cell_index);
            
            // Sample at different lifecycle points
            let half_peak_year = 0.0;  // Assume current year as reference
            let peak_year = 50_000.0;  // Typical hotspot peak 
            let peak_plus_half_year = 100_000.0;
            let mid_decline_year = 150_000.0;
            
            let cache = HotspotEnergyCache {
                half_peak: self.calculate_radiance_system_energy_cached(cell_index, surface_area_km2, time_years, half_peak_year, &neighbors),
                peak: self.calculate_radiance_system_energy_cached(cell_index, surface_area_km2, time_years, peak_year, &neighbors),
                peak_plus_half: self.calculate_radiance_system_energy_cached(cell_index, surface_area_km2, time_years, peak_plus_half_year, &neighbors),
                mid_decline: self.calculate_radiance_system_energy_cached(cell_index, surface_area_km2, time_years, mid_decline_year, &neighbors),
            };
            
            self.hotspot_energy_cache.insert(cell_index, cache);
        }
    }
    
    /// Apply radiance energy to all cells in the simulation
    pub fn apply(&mut self, cells: &mut HashMap<CellIndex, GlobalH3Cell>, time_years: f64, current_year: f64) {
        // Accumulate energy for 100-year batch injection
        self.years_since_last_injection += time_years;
        
        // Only apply energy every 100 years, but multiply by accumulated time
        if self.years_since_last_injection >= 100.0 {
            let energy_multiplier = self.years_since_last_injection;
            self.years_since_last_injection = 0.0;
            
            self.total_energy_added = 0.0;
            self.cells_processed = 0;
            self.step_count += 1;

            // Update radiance system for current year
            self.radiance_system.update(current_year);

            // New hotspot-focused approach: loop over hotspots and distribute energy
            self.apply_hotspot_focused(cells, time_years * energy_multiplier, current_year);

            if self.params.enable_reporting && self.step_count % 10 == 0 {
                println!("RadianceOp Step {}: Added {:.2e} J to {} cells ({}x multiplier)", 
                    self.step_count, self.total_energy_added, self.cells_processed, energy_multiplier);
            }
        }
    }

    /// Apply radiance energy using hotspot-focused approach for better performance
    fn apply_hotspot_focused(&mut self, cells: &mut HashMap<CellIndex, GlobalH3Cell>, time_years: f64, current_year: f64) {
        // First, apply base Perlin energy to all cells (this still needs to be done for all cells)
        for (cell_index, cell) in cells.iter_mut() {
            let energy_added = self.apply_base_energy_to_cell(*cell_index, cell, time_years);
            self.total_energy_added += energy_added;
            self.cells_processed += 1;
        }

        // Then, apply hotspot energy by looping over hotspots (inflows/outflows)
        self.apply_hotspot_energy(cells, time_years, current_year);
    }

    /// Apply base Perlin energy to a single cell
    fn apply_base_energy_to_cell(&self, cell_index: CellIndex, cell: &mut GlobalH3Cell, time_years: f64) -> f64 {
        // Get surface area from any layer (they all have the same surface area)
        let surface_area_km2 = if let Some((layer, _)) = cell.layers_t.first() {
            layer.surface_area_km2
        } else {
            return 0.0; // No layers
        };

        // Find the foundry layer (single deepest asthenosphere layer as heat source)
        let foundry_layer_indices = self.find_foundry_layers(cell);

        if !foundry_layer_indices.is_empty() {
            // Calculate base core radiance energy
            let base_energy = self.params.base_core_radiance_j_per_km2_per_year * surface_area_km2 * time_years;

            // Calculate Perlin noise energy contribution
            let perlin_energy = self.get_cached_perlin_energy(cell_index) * surface_area_km2 * time_years;

            // Combine base and perlin energies
            let total_energy = base_energy + (perlin_energy * self.params.radiance_system_multiplier);

            // Add energy to deepest foundry layer
            if let Some(&deepest_layer_index) = foundry_layer_indices.last() {
                cell.layers_t[deepest_layer_index].1.add_energy(total_energy);
                return total_energy;
            }
        }
        0.0
    }

    /// Apply energy from hotspots (inflows/outflows) to cells and their neighbors
    fn apply_hotspot_energy(&mut self, cells: &mut HashMap<CellIndex, GlobalH3Cell>, time_years: f64, current_year: f64) {
        // Process inflows (hotspots)
        for inflow in &self.radiance_system.inflows {
            let inflow_energy = self.calculate_inflow_energy(inflow.cell_index, current_year, time_years);
            if inflow_energy > 0.0 {
                // Apply full energy to hotspot cell
                if let Some(cell) = cells.get_mut(&inflow.cell_index) {
                    self.apply_energy_to_foundry_layer(cell, inflow_energy);
                    self.total_energy_added += inflow_energy;
                }

                // Apply 50% energy to neighbor cells
                let neighbors = H3Utils::neighbors_for(inflow.cell_index);
                let neighbor_energy = inflow_energy * 0.5;
                for neighbor_cell_index in neighbors {
                    if let Some(cell) = cells.get_mut(&neighbor_cell_index) {
                        self.apply_energy_to_foundry_layer(cell, neighbor_energy);
                        self.total_energy_added += neighbor_energy;
                    }
                }
            }
        }

        // Process outflows (cooling zones)
        for outflow in &self.radiance_system.outflows {
            let outflow_energy = self.calculate_outflow_energy(outflow.cell_index, current_year, time_years);
            if outflow_energy != 0.0 {
                // Apply full energy to outflow cell (negative energy for cooling)
                if let Some(cell) = cells.get_mut(&outflow.cell_index) {
                    self.apply_energy_to_foundry_layer(cell, outflow_energy);
                    self.total_energy_added += outflow_energy;
                }

                // Apply 50% energy to neighbor cells
                let neighbors = H3Utils::neighbors_for(outflow.cell_index);
                let neighbor_energy = outflow_energy * 0.5;
                for neighbor_cell_index in neighbors {
                    if let Some(cell) = cells.get_mut(&neighbor_cell_index) {
                        self.apply_energy_to_foundry_layer(cell, neighbor_energy);
                        self.total_energy_added += neighbor_energy;
                    }
                }
            }
        }
    }

    /// Apply energy to the foundry layer of a cell
    fn apply_energy_to_foundry_layer(&self, cell: &mut GlobalH3Cell, energy: f64) {
        let foundry_layer_indices = self.find_foundry_layers(cell);
        if let Some(&deepest_layer_index) = foundry_layer_indices.last() {
            cell.layers_t[deepest_layer_index].1.add_energy(energy);
        }
    }

    /// Calculate inflow energy for a specific cell
    fn calculate_inflow_energy(&self, cell_index: CellIndex, current_year: f64, time_years: f64) -> f64 {
        for inflow in &self.radiance_system.inflows {
            if inflow.cell_index == cell_index {
                let age = current_year - inflow.creation_year;
                let lifecycle_factor = self.calculate_lifecycle_factor(age, inflow.lifetime_years);
                return inflow.rate_mw * lifecycle_factor * time_years;
            }
        }
        0.0
    }

    /// Calculate outflow energy for a specific cell
    fn calculate_outflow_energy(&self, cell_index: CellIndex, current_year: f64, time_years: f64) -> f64 {
        for outflow in &self.radiance_system.outflows {
            if outflow.cell_index == cell_index {
                let age = current_year - outflow.creation_year;
                let lifecycle_factor = self.calculate_lifecycle_factor(age, outflow.lifetime_years);
                return -outflow.rate_mw * lifecycle_factor * time_years; // Negative for cooling
            }
        }
        0.0
    }

    /// Calculate lifecycle factor for hotspot energy
    fn calculate_lifecycle_factor(&self, age: f64, lifetime_years: f64) -> f64 {
        if age < 0.0 || age > lifetime_years {
            return 0.0;
        }
        
        // Exponential decay similar to real hotspots
        let normalized_age = age / lifetime_years;
        (1.0 - normalized_age).max(0.0)
    }

    /// Apply radiance energy to a single cell
    fn apply_to_cell(
        &self,
        cell_index: CellIndex,
        cell: &mut GlobalH3Cell,
        time_years: f64,
        current_year: f64,
    ) -> f64 {
        // Get surface area from any layer (they all have the same surface area)
        let surface_area_km2 = if let Some((layer, _)) = cell.layers_t.first() {
            layer.surface_area_km2
        } else {
            return 0.0; // No layers
        };

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

    /// Apply radiance energy to a single cell (cached version)
    fn apply_to_cell_cached(
        &mut self,
        cell_index: CellIndex,
        cell: &mut GlobalH3Cell,
        time_years: f64,
        current_year: f64,
    ) -> f64 {
        // Get surface area from any layer (they all have the same surface area)
        let surface_area_km2 = if let Some((layer, _)) = cell.layers_t.first() {
            layer.surface_area_km2
        } else {
            return 0.0; // No layers
        };

        // Find the foundry layer (single deepest asthenosphere layer as heat source)
        let foundry_layer_indices = self.find_foundry_layers(cell);

        if !foundry_layer_indices.is_empty() {
            // Calculate base core radiance energy
            let base_energy = self.params.base_core_radiance_j_per_km2_per_year * surface_area_km2 * time_years;

            // Pre-cache hotspot energy for this cell
            self.cache_hotspot_energy(cell_index, surface_area_km2, time_years);

            // Calculate radiance system energy contribution using cache
            let radiance_energy = self.calculate_radiance_system_energy_cached(
                cell_index, surface_area_km2, time_years, current_year, &H3Utils::neighbors_for(cell_index)
            );

            // Combine base and radiance system energies
            let total_energy = base_energy + (radiance_energy * self.params.radiance_system_multiplier);

            // Add radiance energy directly to deepest foundry layer (natural heat source)
            if !foundry_layer_indices.is_empty() {
                // Find the deepest (last) foundry layer to add energy to
                let deepest_layer_index = *foundry_layer_indices.last().unwrap();
                
                // Add the total energy directly to the deepest layer
                cell.layers_t[deepest_layer_index].1.add_energy(total_energy);

                if self.params.enable_reporting && self.step_count % 1000 == 0 {
                    let layer_temp = cell.layers_t[deepest_layer_index].1.temperature_k();
                    println!("  Cell {:?}: Added {:.2e}J to deepest layer (index {}, now {:.0}K)",
                        cell_index, total_energy, deepest_layer_index, layer_temp);
                }
                
                return total_energy;
            }
        }
        0.0
    }

    /// Calculate energy contribution from the radiance system (cached version)
    fn calculate_radiance_system_energy_cached(
        &self,
        cell_index: CellIndex,
        surface_area_km2: f64,
        time_years: f64,
        current_year: f64,
        neighbors: &[CellIndex],
    ) -> f64 {
        // Use cached perlin energy
        let perlin_energy = self.get_cached_perlin_energy(cell_index);
        
        // Use simplified hotspot calculation - interpolate from cache if available
        let hotspot_energy = if let Some(cache) = self.hotspot_energy_cache.get(&cell_index) {
            // Simple interpolation based on current year lifecycle position
            // This is a simplified version - could be more sophisticated
            (cache.half_peak + cache.peak + cache.peak_plus_half + cache.mid_decline) / 4.0
        } else {
            0.0
        };
        
        let total_energy_per_km2_per_year = perlin_energy + hotspot_energy;
        total_energy_per_km2_per_year * surface_area_km2 * time_years
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
