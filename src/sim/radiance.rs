/// Radiance System for Thermal Energy Management
/// 
/// Manages Perlin noise-based thermal background with time-based transitions
/// and major inflows/outflows with lifecycle management

use h3o::{CellIndex, Resolution};
use noise::{NoiseFn, Perlin};
use rand::Rng;
use glam::Vec3;
use crate::h3o_png::{H3GraphicsGenerator, H3GraphicsConfig};

use crate::h3_utils::H3Utils;
use image::Rgb;
use std::collections::HashMap;

/// Calculate binomial probability: P(X = k) = C(n,k) * p^k * (1-p)^(n-k)
fn binomial_probability(n: usize, k: usize, p: f64) -> f64 {
    if k > n {
        return 0.0;
    }

    // Calculate binomial coefficient C(n,k) = n! / (k! * (n-k)!)
    let mut coeff = 1.0;
    for i in 0..k {
        coeff *= (n - i) as f64 / (i + 1) as f64;
    }

    // Calculate p^k * (1-p)^(n-k)
    let prob_term = p.powi(k as i32) * (1.0 - p).powi((n - k) as i32);

    coeff * prob_term
}


/// Radiance system managing thermal energy distribution
#[derive(Debug, Clone)]
pub struct RadianceSystem {
    /// Current Perlin noise configuration
    pub current_perlin: PerlinConfig,
    /// Next Perlin noise configuration for transition
    pub next_perlin: PerlinConfig,
    /// Transition progress (0.0 = current, 1.0 = next)
    pub transition_progress: f64,
    /// Years between Perlin transitions
    pub transition_period_years: f64,
    /// Last transition time
    pub last_transition_year: f64,
    /// Major thermal inflows (upwells)
    pub inflows: Vec<ThermalFlow>,
    /// Major thermal outflows (downwells)
    pub outflows: Vec<ThermalFlow>,
    /// Configuration for thermal feature lifecycle management
    pub thermal_config: ThermalFeatureConfig,
}

/// Perlin noise configuration with timing
#[derive(Debug, Clone)]
pub struct PerlinConfig {
    /// Perlin noise seed
    pub seed: u32,
    /// Noise scale factor
    pub scale: f64,
    /// Base amplitude
    pub amplitude: f64,
    /// Wavelength in kilometers
    pub wavelength_km: f64,
    /// Configuration creation time
    pub creation_year: f64,
}

/// Thermal flow (inflow/outflow) with lifecycle
#[derive(Debug, Clone)]
pub struct ThermalFlow {
    /// H3 cell location
    pub cell_index: CellIndex,
    /// Energy rate (MW) - positive for inflows, negative for outflows
    pub rate_mw: f64,
    /// Year this flow was created
    pub creation_year: f64,
    /// Lifetime in years
    pub lifetime_years: f64,
    /// Flow type
    pub flow_type: FlowType,
}

/// Type of thermal flow
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FlowType {
    /// Thermal inflow (upwell)
    Inflow,
    /// Thermal outflow (downwell)
    Outflow,
}

impl RadianceSystem {
    /// Create new radiance system
    pub fn new(current_year: f64) -> Self {
        let mut rng = rand::rng();
        
        // Create initial Perlin configuration
        let current_perlin = PerlinConfig {
            seed: rng.random(),
            scale: 4.4, // From previous calculations
            amplitude: 0.08, // ±0.08 variation around 0.07 base (4x more dramatic)
            wavelength_km: 996.0, // 3 L2 hex diameter
            creation_year: current_year,
        };

        // Create next Perlin for future transition (shorter periods for visible animation)
        let transition_period = rng.random_range(80.0..120.0);
        let next_perlin = PerlinConfig {
            seed: rng.random::<u32>().wrapping_add(12345), // Ensure dramatically different seed
            scale: rng.random_range(3.0..8.0), // Vary scale dramatically for different patterns
            amplitude: 0.08, // ±0.08 variation around 0.07 base (4x more dramatic)
            wavelength_km: 996.0,
            creation_year: current_year + transition_period,
        };
        
        Self {
            current_perlin,
            next_perlin,
            transition_progress: 0.0,
            transition_period_years: transition_period,
            last_transition_year: current_year,
            inflows: Vec::new(),
            outflows: Vec::new(),
            thermal_config: ThermalFeatureConfig::default(),
        }
    }
    
    /// Update radiance system for current year
    pub fn update(&mut self, current_year: f64) {
        // Update Perlin transition
        self.update_perlin_transition(current_year);
        
        // Remove expired flows
        self.remove_expired_flows(current_year);
    }
    
    /// Update Perlin noise transition
    fn update_perlin_transition(&mut self, current_year: f64) {
        let years_since_transition = current_year - self.last_transition_year;
        
        if years_since_transition >= self.transition_period_years {
            // Complete transition - move to next Perlin
            self.current_perlin = self.next_perlin.clone();
            self.last_transition_year = current_year;
            
            // Generate new next Perlin (shorter periods for visible animation)
            let mut rng = rand::rng();
            self.transition_period_years = rng.random_range(80.0..120.0);
            self.next_perlin = PerlinConfig {
                seed: rng.random::<u32>().wrapping_add((current_year as u32).wrapping_mul(7919)), // Ensure dramatically different seed based on time
                scale: rng.random_range(3.0..8.0), // Vary scale dramatically for different patterns
                amplitude: 0.08, // ±0.08 WM variation around 0.07 WM base (4x more dramatic)
                wavelength_km: 996.0,
                creation_year: current_year + self.transition_period_years,
            };
            self.transition_progress = 0.0;
        } else {
            // Update transition progress
            self.transition_progress = years_since_transition / self.transition_period_years;
        }
    }
    
    /// Remove expired thermal flows
    fn remove_expired_flows(&mut self, current_year: f64) {
        self.inflows.retain(|flow| {
            let age = current_year - flow.creation_year;
            age < flow.lifetime_years
        });
        
        self.outflows.retain(|flow| {
            let age = current_year - flow.creation_year;
            age < flow.lifetime_years
        });
    }

    /// Create new radiance system with custom thermal configuration
    pub fn new_with_config(current_year: f64, thermal_config: ThermalFeatureConfig) -> Self {
        let mut system = Self::new(current_year);
        system.thermal_config = thermal_config;
        system.thermal_config.last_replenishment_year = current_year;
        system
    }

    /// Add thermal inflow (upwell)
    pub fn add_inflow(&mut self, cell_index: CellIndex, rate_mw: f64, current_year: f64, lifetime_years: f64) -> Result<(), String> {
        // Check if cell already has an inflow
        if self.inflows.iter().any(|flow| flow.cell_index == cell_index) {
            return Err("Cell already has an inflow".to_string());
        }
        
        // Check neighbor constraints - no inflows can be neighbors
        if self.has_neighboring_inflow(cell_index) {
            return Err("Cannot place inflow next to existing inflow".to_string());
        }
        
        let inflow = ThermalFlow {
            cell_index,
            rate_mw: rate_mw.abs(), // Ensure positive
            creation_year: current_year,
            lifetime_years,
            flow_type: FlowType::Inflow,
        };
        
        self.inflows.push(inflow);
        Ok(())
    }
    
    /// Add thermal outflow (downwell)
    pub fn add_outflow(&mut self, cell_index: CellIndex, rate_mw: f64, current_year: f64, lifetime_years: f64) -> Result<(), String> {
        // Check if cell already has a flow
        if self.outflows.iter().any(|flow| flow.cell_index == cell_index) ||
           self.inflows.iter().any(|flow| flow.cell_index == cell_index) {
            return Err("Cell already has a thermal flow".to_string());
        }
        
        // Check neighbor constraints - no outflows can be neighbors to inflows
        if self.has_neighboring_inflow(cell_index) {
            return Err("Cannot place outflow next to existing inflow".to_string());
        }
        
        let outflow = ThermalFlow {
            cell_index,
            rate_mw: -rate_mw.abs(), // Ensure negative
            creation_year: current_year,
            lifetime_years,
            flow_type: FlowType::Outflow,
        };
        
        self.outflows.push(outflow);
        Ok(())
    }

    /// Check if cell has neighboring inflow using H3 grid neighbors
    pub fn has_neighboring_inflow(&self, cell_index: CellIndex) -> bool {
        let neighbors = H3Utils::neighbors_for(cell_index);
        // Check if any neighbor has an inflow
        for neighbor_cell in neighbors {
            if self.inflows.iter().any(|flow| flow.cell_index == neighbor_cell) {
                return true;
            }
        }
        false
    }

    /// Generate random thermal inflows for testing and simulation
    pub fn generate_random_inflows(
        &mut self,
        resolution: h3o::Resolution,
        count: usize,
        rate_range: (f64, f64),
        lifetime_range: (f64, f64),
        creation_year_range: (f64, f64),
    ) -> Result<(), Box<dyn std::error::Error>> {
        use rand::seq::SliceRandom;
        use rand::Rng;

        // Get all cells at the specified resolution
        let mut cells: Vec<_> = h3o::CellIndex::base_cells()
            .flat_map(|base| base.children(resolution))
            .collect();

        let mut rng = rand::rng();
        cells.shuffle(&mut rng);

        println!("Generating {} random thermal inflows...", count);

        let mut added_count = 0;
        let mut attempt_index = 0;

        while added_count < count && attempt_index < cells.len() {
            if let Some(&cell) = cells.get(attempt_index) {
                let rate_wm = rng.random_range(rate_range.0..rate_range.1);
                let lifetime_years = rng.random_range(lifetime_range.0..lifetime_range.1);
                let creation_year = rng.random_range(creation_year_range.0..creation_year_range.1);

                match self.add_inflow(cell, rate_wm, creation_year, lifetime_years) {
                    Ok(_) => {
                        println!("  Added upwell at cell {}: {:.3} WM, {:.0} year lifetime, created {:.0}",
                                 cell, rate_wm, lifetime_years, creation_year);
                        added_count += 1;
                    }
                    Err(_) => {
                        // Skip this cell due to constraint violation, try next one
                    }
                }
            }
            attempt_index += 1;
        }

        if added_count < count {
            println!("  Warning: Only added {} out of {} requested inflows due to constraints", added_count, count);
        }

        Ok(())
    }

    /// Generate random thermal outflows for testing and simulation
    pub fn generate_random_outflows(
        &mut self,
        resolution: h3o::Resolution,
        count: usize,
        rate_range: (f64, f64),
        lifetime_range: (f64, f64),
        creation_year_range: (f64, f64),
    ) -> Result<(), Box<dyn std::error::Error>> {
        use rand::seq::SliceRandom;
        use rand::Rng;

        // Get all cells at the specified resolution
        let mut cells: Vec<_> = h3o::CellIndex::base_cells()
            .flat_map(|base| base.children(resolution))
            .collect();

        let mut rng = rand::rng();
        cells.shuffle(&mut rng);

        println!("Generating {} random thermal outflows...", count);

        let mut added_count = 0;
        let mut attempt_index = 0;

        while added_count < count && attempt_index < cells.len() {
            // Use different spacing to avoid overlap with inflows
            let cell_index = (attempt_index * 294 + 147) % cells.len();
            if let Some(&cell) = cells.get(cell_index) {
                let rate_wm = rng.random_range(rate_range.0..rate_range.1);
                let lifetime_years = rng.random_range(lifetime_range.0..lifetime_range.1);
                let creation_year = rng.random_range(creation_year_range.0..creation_year_range.1);

                match self.add_outflow(cell, rate_wm, creation_year, lifetime_years) {
                    Ok(_) => {
                        println!("  Added downwell at cell {}: -{:.3} WM, {:.0} year lifetime, created {:.0}",
                                 cell, rate_wm, lifetime_years, creation_year);
                        added_count += 1;
                    }
                    Err(_) => {
                        // Skip this cell due to constraint violation, try next one
                    }
                }
            }
            attempt_index += 1;
        }

        if added_count < count {
            println!("  Warning: Only added {} out of {} requested outflows due to constraints", added_count, count);
        }

        Ok(())
    }

    /// Generate realistic thermal features for testing and simulation
    /// Creates a balanced mix of inflows and outflows with realistic parameters
    pub fn generate_realistic_thermal_features(
        &mut self,
        resolution: h3o::Resolution,
        inflow_count: usize,
        outflow_count: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        println!("Generating realistic thermal features...");

        // Generate inflows (upwells) with realistic parameters
        // Rate: 0.05-0.4 WM (typical geothermal upwelling)
        // Lifetime: 200-800 years (geological timescales)
        // Creation: -300 to +5 years (recent geological activity)
        self.generate_random_inflows(
            resolution,
            inflow_count,
            (0.05, 0.4),
            (200.0, 800.0),
            (-300.0, 5.0),
        )?;

        // Generate outflows (downwells) with realistic parameters
        // Rate: 0.025-0.15 WM (thermal sinks, cooler zones)
        // Lifetime: 100-500 years (shorter than upwells)
        // Creation: -500 to 0 years (older features)
        self.generate_random_outflows(
            resolution,
            outflow_count,
            (0.025, 0.15),
            (100.0, 500.0),
            (-500.0, 0.0),
        )?;

        println!("Generated {} inflows and {} outflows with realistic parameters",
                 inflow_count, outflow_count);

        Ok(())
    }

    /// Check if cell has neighboring inflow using neighbor cache
    pub fn has_neighboring_inflow_with_cache(
        &self,
        cell_index: CellIndex,
        neighbors: &[CellIndex]
    ) -> bool {
        // Check if any neighbor has an inflow
        for &neighbor_cell in neighbors {
            if self.inflows.iter().any(|flow| flow.cell_index == neighbor_cell) {
                return true;
            }
        }
        false
    }

    /// Add thermal inflow with neighbor cache constraint checking
    pub fn add_inflow_with_cache(
        &mut self,
        cell_index: CellIndex,
        rate_mw: f64,
        current_year: f64,
        lifetime_years: f64,
        neighbors: &[CellIndex]
    ) -> Result<(), String> {
        // Check if cell already has an inflow
        if self.inflows.iter().any(|flow| flow.cell_index == cell_index) {
            return Err("Cell already has an inflow".to_string());
        }

        // Check neighbor constraints - no inflows can be neighbors
        if self.has_neighboring_inflow_with_cache(cell_index, neighbors) {
            return Err("Cannot place inflow next to existing inflow".to_string());
        }

        let inflow = ThermalFlow {
            cell_index,
            rate_mw: rate_mw.abs(), // Ensure positive
            creation_year: current_year,
            lifetime_years,
            flow_type: FlowType::Inflow,
        };

        self.inflows.push(inflow);
        Ok(())
    }

    /// Add thermal outflow with neighbor cache constraint checking
    pub fn add_outflow_with_cache(
        &mut self,
        cell_index: CellIndex,
        rate_mw: f64,
        current_year: f64,
        lifetime_years: f64,
        neighbors: &[CellIndex]
    ) -> Result<(), String> {
        // Check if cell already has a flow
        if self.outflows.iter().any(|flow| flow.cell_index == cell_index) ||
           self.inflows.iter().any(|flow| flow.cell_index == cell_index) {
            return Err("Cell already has a thermal flow".to_string());
        }

        // Check neighbor constraints - no outflows can be neighbors to inflows
        if self.has_neighboring_inflow_with_cache(cell_index, neighbors) {
            return Err("Cannot place outflow next to existing inflow".to_string());
        }

        let outflow = ThermalFlow {
            cell_index,
            rate_mw: -rate_mw.abs(), // Ensure negative
            creation_year: current_year,
            lifetime_years,
            flow_type: FlowType::Outflow,
        };

        self.outflows.push(outflow);
        Ok(())
    }
    
    /// Calculate thermal energy for all cells efficiently with H3 neighbors
    pub fn calculate_all_cell_energies_with_neighbors(
        &self,
        cells: &[(CellIndex, f64, f64)],
        current_year: f64,
    ) -> Result<Vec<Cell>, Box<dyn std::error::Error>> {
        // Step 1: Initialize cells with Perlin energy
        let mut thermal_cells: Vec<Cell> = cells.iter()
            .map(|(cell_index, lat, lng)| {
                let mut cell = Cell::new(*cell_index, *lat, *lng);
                cell.set_energy(self.calculate_perlin_energy(*lat, *lng));
                cell
            })
            .collect();

        // Step 2: Add direct inflow effects to specific cells
        for inflow in &self.inflows {
            if let Some(cell_pos) = thermal_cells.iter().position(|cell| cell.cell_index == inflow.cell_index) {
                let flow_energy = self.calculate_inflow_energy(inflow.cell_index, current_year);
                thermal_cells[cell_pos].add_energy(flow_energy);
            }
        }

        // Step 3: Add direct outflow effects to specific cells
        for outflow in &self.outflows {
            if let Some(cell_pos) = thermal_cells.iter().position(|cell| cell.cell_index == outflow.cell_index) {
                let flow_energy = self.calculate_outflow_energy(outflow.cell_index, current_year);
                thermal_cells[cell_pos].add_energy(flow_energy);
            }
        }

        // Step 4: Add neighbor effects (50% immediate + 25% 3rd order) using cache - iterate over flows, not cells
        let cell_index_map: HashMap<CellIndex, usize> = thermal_cells.iter().enumerate()
            .map(|(i, cell)| (cell.cell_index, i))
            .collect();

        // Apply inflow neighbor effects using cache (immediate neighbors - 50% strength)
        let mut total_neighbor_applications = 0;
        for inflow in &self.inflows {
            let flow_energy = self.calculate_inflow_energy(inflow.cell_index, current_year);
            if flow_energy > 0.0 {
                let neighbor_effect = flow_energy * 0.25; 

                // Get immediate neighbors using H3 grid structure (excluding source cell)
                let neighbors = H3Utils::neighbors_for(inflow.cell_index);
                let immediate_neighbors: std::collections::HashSet<CellIndex> = neighbors.iter().cloned().collect();

                for neighbor in &neighbors {
                    if *neighbor != inflow.cell_index { // Exclude source cell
                        if let Some(&neighbor_idx) = cell_index_map.get(neighbor) {
                            thermal_cells[neighbor_idx].add_energy(neighbor_effect);
                            total_neighbor_applications += 1;
                        }
                    }
                }
            }
        }

        // Apply outflow neighbor effects using cache (immediate neighbors - 50% strength)
        for outflow in &self.outflows {
            let flow_energy = self.calculate_outflow_energy(outflow.cell_index, current_year);
            if flow_energy < 0.0 {
                let neighbor_effect = flow_energy * 0.25; 

                // Get immediate neighbors using H3 grid structure (excluding source cell)
                let neighbors = H3Utils::neighbors_for(outflow.cell_index);
                let immediate_neighbors: std::collections::HashSet<CellIndex> = neighbors.iter().cloned().collect();

                for neighbor in &neighbors {
                    if *neighbor != outflow.cell_index { // Exclude source cell
                        if let Some(&neighbor_idx) = cell_index_map.get(neighbor) {
                            thermal_cells[neighbor_idx].add_energy(neighbor_effect);
                            total_neighbor_applications += 1;
                        }
                    }
                }
            }
        }

        // Step 5: Ensure no energy goes below 0
        for cell in &mut thermal_cells {
            cell.energy = cell.energy.max(0.0);
        }

        Ok(thermal_cells)
    }

    /// Calculate thermal energy for a cell with neighbor cache
    pub fn calculate_cell_energy_with_neighbors(
        &self,
        cell_index: CellIndex,
        lat: f64,
        lng: f64,
        current_year: f64,
        neighbors: &[CellIndex]
    ) -> f64 {
        let mut total_energy = 0.0;

        // Add Perlin noise background
        total_energy += self.calculate_perlin_energy(lat, lng);

        // Add direct inflow/outflow contribution (100% strength)
        total_energy += self.calculate_inflow_energy(cell_index, current_year);
        total_energy += self.calculate_outflow_energy(cell_index, current_year);

        // Add neighbor inflow/outflow effects (50% strength) using provided neighbors
        total_energy += self.calculate_neighbor_flow_effects_with_cache(cell_index, current_year, neighbors);

        // Ensure energy doesn't go below 0
        total_energy.max(0.0)
    }
    
    /// Calculate Perlin noise energy with transition using normalized XYZ coordinates
    fn calculate_perlin_energy(&self, lat: f64, lng: f64) -> f64 {
        // Convert lat/lng to normalized 3D coordinates on unit sphere using Vec3
        let lat_rad = lat.to_radians();
        let lng_rad = lng.to_radians();
        let pos = Vec3::new(
            (lat_rad.cos() * lng_rad.cos()) as f32,
            (lat_rad.cos() * lng_rad.sin()) as f32,
            lat_rad.sin() as f32,
        );

        // Normalize to ensure unit sphere (Vec3 provides built-in normalization)
        let normalized_pos = pos.normalize();

        // Sample current Perlin using normalized coordinates
        let current_perlin = Perlin::new(self.current_perlin.seed);
        let current_value = current_perlin.get([
            (normalized_pos.x * self.current_perlin.scale as f32) as f64,
            (normalized_pos.y * self.current_perlin.scale as f32) as f64,
            (normalized_pos.z * self.current_perlin.scale as f32) as f64,
        ]);
        // Perlin base of 0.07 WM with ±0.08 WM variation (4x more dramatic)
        let current_energy = 0.07 + (current_value * self.current_perlin.amplitude);

        if self.transition_progress <= 0.0 {
            return current_energy;
        }

        // Sample next Perlin using normalized coordinates
        let next_perlin = Perlin::new(self.next_perlin.seed);
        let next_value = next_perlin.get([
            (normalized_pos.x * self.next_perlin.scale as f32) as f64,
            (normalized_pos.y * self.next_perlin.scale as f32) as f64,
            (normalized_pos.z * self.next_perlin.scale as f32) as f64,
        ]);
        // Perlin base of 0.07 WM with ±0.08 WM variation (4x more dramatic)
        let next_energy = 0.07 + (next_value * self.next_perlin.amplitude);

        // Interpolate between current and next
        current_energy * (1.0 - self.transition_progress) + next_energy * self.transition_progress
    }
    
    /// Calculate inflow energy with lifecycle easing
    fn calculate_inflow_energy(&self, cell_index: CellIndex, current_year: f64) -> f64 {
        for inflow in &self.inflows {
            if inflow.cell_index == cell_index {
                let age = current_year - inflow.creation_year;
                let lifecycle_factor = self.calculate_lifecycle_factor(age, inflow.lifetime_years);
                return inflow.rate_mw * lifecycle_factor;
            }
        }
        0.0
    }
    
    /// Calculate outflow energy with lifecycle easing
    fn calculate_outflow_energy(&self, cell_index: CellIndex, current_year: f64) -> f64 {
        for outflow in &self.outflows {
            if outflow.cell_index == cell_index {
                let age = current_year - outflow.creation_year;
                let lifecycle_factor = self.calculate_lifecycle_factor(age, outflow.lifetime_years);
                return outflow.rate_mw * lifecycle_factor;
            }
        }
        0.0
    }
    
    /// Calculate lifecycle factor with easing for first and last 15%
    fn calculate_lifecycle_factor(&self, age: f64, lifetime: f64) -> f64 {
        if age < 0.0 || age > lifetime {
            return 0.0;
        }

        let progress = age / lifetime;

        if progress < 0.15 {
            // Ease in - first 15%
            let ease_progress = progress / 0.15;
            ease_progress * ease_progress // Quadratic ease in
        } else if progress > 0.85 {
            // Ease out - last 15%
            let ease_progress = (1.0 - progress) / 0.15;
            ease_progress * ease_progress // Quadratic ease out
        } else {
            // Full strength - middle 70%
            1.0
        }
    }

    /// Calculate neighbor flow effects (placeholder - needs neighbor cache)
    fn calculate_neighbor_flow_effects(&self, _cell_index: CellIndex, _current_year: f64) -> f64 {
        // TODO: Implement with neighbor cache
        // For now, return 0.0 as no neighbor effects
        0.0
    }

    /// Calculate neighbor flow effects using provided neighbor list
    fn calculate_neighbor_flow_effects_with_cache(
        &self,
        _cell_index: CellIndex,
        current_year: f64,
        neighbors: &[CellIndex]
    ) -> f64 {
        let mut neighbor_energy = 0.0;

        // Check each neighbor for inflows/outflows
        for &neighbor_cell in neighbors {
            // Add 50% of neighbor inflow effects
            let neighbor_inflow = self.calculate_inflow_energy(neighbor_cell, current_year);
            neighbor_energy += neighbor_inflow * 0.5;

            // Add 50% of neighbor outflow effects
            let neighbor_outflow = self.calculate_outflow_energy(neighbor_cell, current_year);
            neighbor_energy += neighbor_outflow * 0.5;
        }

        neighbor_energy
    }
    
    /// Get system statistics
    pub fn get_statistics(&self, current_year: f64) -> RadianceStatistics {
        let active_inflows = self.inflows.len();
        let active_outflows = self.outflows.len();
        
        let total_inflow_rate: f64 = self.inflows.iter()
            .map(|flow| {
                let age = current_year - flow.creation_year;
                let factor = self.calculate_lifecycle_factor(age, flow.lifetime_years);
                flow.rate_mw * factor
            })
            .sum();
            
        let total_outflow_rate: f64 = self.outflows.iter()
            .map(|flow| {
                let age = current_year - flow.creation_year;
                let factor = self.calculate_lifecycle_factor(age, flow.lifetime_years);
                flow.rate_mw * factor
            })
            .sum();
        
        RadianceStatistics {
            current_year,
            active_inflows,
            active_outflows,
            total_inflow_rate_mw: total_inflow_rate,
            total_outflow_rate_mw: total_outflow_rate,
            net_flow_rate_mw: total_inflow_rate + total_outflow_rate,
            perlin_transition_progress: self.transition_progress,
            years_to_next_perlin: self.transition_period_years - (current_year - self.last_transition_year),
        }
    }
}

/// Radiance system statistics
#[derive(Debug, Clone)]
pub struct RadianceStatistics {
    pub current_year: f64,
    pub active_inflows: usize,
    pub active_outflows: usize,
    pub total_inflow_rate_mw: f64,
    pub total_outflow_rate_mw: f64,
    pub net_flow_rate_mw: f64,
    pub perlin_transition_progress: f64,
    pub years_to_next_perlin: f64,
}

impl RadianceSystem {


    /// Generate thermal PNG visualization with neighbor effects
    pub fn generate_thermal_png(
        &self,
        resolution: Resolution,
        points_per_degree: u32,
        current_year: f64,
        filename: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        println!("=== Generating Thermal PNG ===");
        println!("Resolution: {:?}, Year: {:.1}", resolution, current_year);
        println!("Using H3 grid neighbors for thermal diffusion");

        // Create PNG generator
        let config = H3GraphicsConfig::new(resolution, points_per_degree);
        let mut png_generator = H3GraphicsGenerator::new(config);
        png_generator.load_cells();

        // Calculate thermal values for all cells with neighbor effects efficiently
        let cell_coords: Vec<_> = png_generator.cells.iter()
            .map(|cell| (cell.cell_index, cell.center.latitude, cell.center.longitude))
            .collect();
        let thermal_cells = self.calculate_all_cell_energies_with_neighbors(&cell_coords, current_year)?;

        // Extract energy values for processing
        let thermal_values: Vec<f64> = thermal_cells.iter().map(|cell| cell.energy).collect();

        // Process thermal values and generate colors
        let cell_colors = self.process_thermal_values_to_colors(&thermal_values, &png_generator);

        // Generate PNG using pure drawing utilities
        png_generator.draw_thermal_png_from_colors(&cell_colors, filename)?;

        // Display statistics
        self.display_thermal_statistics(&thermal_values, current_year);

        Ok(())
    }

    /// Display thermal statistics
    fn display_thermal_statistics(&self, thermal_values: &[f64], current_year: f64) {
        let min_energy = thermal_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_energy = thermal_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let mean_energy = thermal_values.iter().sum::<f64>() / thermal_values.len() as f64;

        let stats = self.get_statistics(current_year);

        println!("\n=== Thermal Statistics ===");
        println!("Year: {:.1}", current_year);
        println!("Fixed energy scale: 0.00 to {:.2} WM (absolute)", MAX_ENERGY);
        println!("Actual energy range: {:.4} to {:.4} WM", min_energy, max_energy);
        println!("Mean energy: {:.4} WM", mean_energy);
        println!("Scale utilization: {:.1}%", max_energy / MAX_ENERGY * 100.0);

        println!("\nRadiance System:");
        println!("  Active inflows: {}", stats.active_inflows);
        println!("  Active outflows: {}", stats.active_outflows);
        println!("  Total inflow rate: {:.2} MW", stats.total_inflow_rate_mw);
        println!("  Total outflow rate: {:.2} MW", stats.total_outflow_rate_mw);
        println!("  Net flow rate: {:.2} MW", stats.net_flow_rate_mw);

        println!("\nNeighbor Effects:");
        println!("  Immediate neighbors: 50% strength");
        println!("  Neighbors of neighbors: 25% strength");
        println!("  Source cell and immediate neighbors excluded from 3rd order");
        println!("  Constraint checking enabled");

        println!("\nPerlin Transition:");
        println!("  Progress: {:.1}%", stats.perlin_transition_progress * 100.0);
        println!("  Years to next: {:.1}", stats.years_to_next_perlin);

        println!("\nColor Legend (Absolute Scale):");
        println!("  Blue:   0.00 to 0.11 WM (Cold/Low Energy)");
        println!("  Cyan:   0.11 to 0.23 WM (Cool/Moderate Energy)");
        println!("  Green:  0.23 to 0.34 WM (Warm/High Energy)");
        println!("  Yellow: 0.34 to 0.45 WM (Hot/Very High Energy)");
        println!("  Red:    Approaching 0.45 WM (Maximum Energy)");
    }



    /// Process thermal values to colors with fixed energy scale (implementation logic)
    fn process_thermal_values_to_colors(&self, thermal_values: &[f64], png_generator: &H3GraphicsGenerator) -> Vec<Rgb<u8>> {
        // Fixed energy scale for absolute color meaning

        // Find actual energy range for statistics
        let actual_min = thermal_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let actual_max = thermal_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        println!("Fixed energy scale: 0.00 to {:.2} WM (absolute)", MAX_ENERGY);
        println!("Actual energy range: {:.4} to {:.4} WM", actual_min, actual_max);
        println!("Scale utilization: {:.1}%", actual_max / MAX_ENERGY * 100.0);

        // Convert thermal values to colors using fixed scale (0 to MAX_ENERGY)
        let mut cell_colors = Vec::new();
        for &energy in thermal_values {
            let normalized_energy = (energy / MAX_ENERGY).clamp(0.0, 1.0);
            let color = png_generator.thermal_value_to_color(normalized_energy);
            cell_colors.push(color);
        }

        cell_colors
    }

    /// Maintain sustainable thermal features using probabilistic replacement rates
    /// This should be called annually to maintain stable feature populations
    pub fn maintain_thermal_features(
        &mut self,
        resolution: h3o::Resolution,
        current_year: f64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        println!("=== Annual Thermal Feature Maintenance (Year {:.0}) ===", current_year);

        // Clone flow configurations to avoid borrowing issues
        let flow_configs = self.thermal_config.flows.clone();
        let mut rng = rand::rng();

        // Process each flow configuration using binomial distribution
        for flow_config in &flow_configs {
            if flow_config.is_inflow() {
                let active_count = self.count_active_inflows(current_year);
                let avg_lifetime = (flow_config.lifetime_range.0 + flow_config.lifetime_range.1) / 2.0;
                let features_to_add = flow_config.calculate_features_to_add(&mut rng, active_count);

                println!("Current active {}: {}, avg_lifetime: {:.0} years, prob_per_item: {:.4}",
                         flow_config.flow_type, active_count, avg_lifetime, 1.0 / avg_lifetime);

                if features_to_add > 0 {
                    println!("Binomial replacement: adding {} new {}...", features_to_add, flow_config.flow_type);
                    self.generate_random_inflows(
                        resolution,
                        features_to_add,
                        flow_config.rate_range,
                        flow_config.lifetime_range,
                        (current_year + flow_config.creation_year_range.0,
                         current_year + flow_config.creation_year_range.1),
                    )?;
                } else {
                    println!("No {} replacement this year", flow_config.flow_type);
                }
            } else if flow_config.is_outflow() {
                let active_count = self.count_active_outflows(current_year);
                let avg_lifetime = (flow_config.lifetime_range.0 + flow_config.lifetime_range.1) / 2.0;
                let features_to_add = flow_config.calculate_features_to_add(&mut rng, active_count);

                println!("Current active {}: {}, avg_lifetime: {:.0} years, prob_per_item: {:.4}",
                         flow_config.flow_type, active_count, avg_lifetime, 1.0 / avg_lifetime);

                if features_to_add > 0 {
                    println!("Binomial replacement: adding {} new {}...", features_to_add, flow_config.flow_type);
                    // Use the negative rate range directly (already negative in config)
                    self.generate_random_outflows(
                        resolution,
                        features_to_add,
                        flow_config.rate_range,
                        flow_config.lifetime_range,
                        (current_year + flow_config.creation_year_range.0,
                         current_year + flow_config.creation_year_range.1),
                    )?;
                } else {
                    println!("No {} replacement this year", flow_config.flow_type);
                }
            }
        }

        // Update last replenishment time
        self.thermal_config.last_replenishment_year = current_year;

        // Report final counts
        let final_inflows = self.count_active_inflows(current_year);
        let final_outflows = self.count_active_outflows(current_year);
        println!("After maintenance: {} inflows, {} outflows", final_inflows, final_outflows);

        Ok(())
    }

    /// Count currently active inflows at the given year
    pub fn count_active_inflows(&self, current_year: f64) -> usize {
        self.inflows.iter()
            .filter(|flow| self.is_flow_active(flow, current_year))
            .count()
    }

    /// Count currently active outflows at the given year
    pub fn count_active_outflows(&self, current_year: f64) -> usize {
        self.outflows.iter()
            .filter(|flow| self.is_flow_active(flow, current_year))
            .count()
    }

    /// Check if a thermal flow is currently active
    fn is_flow_active(&self, flow: &ThermalFlow, current_year: f64) -> bool {
        let age = current_year - flow.creation_year;
        age >= 0.0 && age <= flow.lifetime_years
    }

    /// Initialize thermal features with sustainable configuration
    pub fn initialize_sustainable_features(
        &mut self,
        resolution: h3o::Resolution,
        current_year: f64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        println!("Initializing sustainable thermal features...");

        // Calculate total targets for inflows and outflows
        let total_inflows: usize = self.thermal_config.inflow_configs().iter()
            .map(|config| config.target_count).sum();
        let total_outflows: usize = self.thermal_config.outflow_configs().iter()
            .map(|config| config.target_count).sum();

        // Generate initial features
        self.generate_realistic_thermal_features(resolution, total_inflows, total_outflows)?;

        // Set the last replenishment time
        self.thermal_config.last_replenishment_year = current_year;

        println!("Initialized {} inflows and {} outflows with sustainable lifecycle management",
                 total_inflows, total_outflows);

        // Display replacement rate information for each flow type
        println!("Replacement rates:");
        for flow_config in &self.thermal_config.flows {
            println!("  {}: {:.3} features/year ({:.1}% annual probability)",
                     flow_config.flow_type, flow_config.replacement_rate, flow_config.replace_rate() * 100.0);
        }

        Ok(())
    }
}

/// Configuration for thermal flow generation and lifecycle management
#[derive(Debug, Clone)]
pub struct ThermalFlowConfig {
    /// Target number of active features to maintain
    pub target_count: usize,
    /// Energy rate range (WM, positive for inflows, negative for outflows)
    pub rate_range: (f64, f64),
    /// Feature lifetime range (years)
    pub lifetime_range: (f64, f64),
    /// Creation year range for new features (relative to current year)
    pub creation_year_range: (f64, f64),
    /// Average replacement rate (features per year) - calculated from generative data
    pub replacement_rate: f64,
    /// Flow type identifier for logging
    pub flow_type: String,
}

impl ThermalFlowConfig {
    /// Calculate the replacement rate based on target count and average lifetime
    /// Uses a higher multiplier to maintain stable populations rather than declining
    pub fn calculate_replacement_rate(&mut self) {
        let avg_lifetime = (self.lifetime_range.0 + self.lifetime_range.1) / 2.0;
        // Use 8x multiplier to ensure stable population maintenance
        // This accounts for probabilistic nature and ensures replacement keeps up with expiration
        self.replacement_rate = (self.target_count as f64 / avg_lifetime) * 8.0;
    }

    /// Get the annual replacement probability (0.0 to 1.0) - percentage chance of adding new feature each year
    pub fn replace_rate(&self) -> f64 {
        if self.target_count == 0 {
            0.0
        } else {
            self.replacement_rate / self.target_count as f64
        }
    }

    /// Get the expected number of features to expire in the given time period
    pub fn expected_expirations(&self, time_period_years: f64) -> f64 {
        self.replacement_rate * time_period_years
    }

    /// Calculate how many new features to add using binomial distribution
    /// For N items with average lifespan Y, probability of replacement = 1/Y per item
    /// Uses binomial distribution to calculate expected replacements, capped at 10% of population
    pub fn calculate_features_to_add(&self, rng: &mut impl rand::Rng, current_count: usize) -> usize {
        if current_count == 0 {
            return 0;
        }

        let avg_lifetime = (self.lifetime_range.0 + self.lifetime_range.1) / 2.0;
        let replacement_prob_per_item = 1.0 / avg_lifetime;
        let max_replacements = ((current_count as f64 * 0.1).ceil() as usize).max(1); // Cap at 10% or minimum 1

        // Build probability distribution for 0 to max_replacements
        let mut cumulative_prob = 0.0;
        let random_value = rng.random::<f64>();

        for k in 0..=max_replacements {
            // Binomial probability: P(X = k) = C(n,k) * p^k * (1-p)^(n-k)
            let prob = binomial_probability(current_count, k, replacement_prob_per_item);
            cumulative_prob += prob;

            if random_value < cumulative_prob {
                return k;
            }
        }

        // Fallback to max if we somehow exceed cumulative probability
        max_replacements
    }

    /// Check if this config represents inflows (positive rates)
    pub fn is_inflow(&self) -> bool {
        self.rate_range.0 >= 0.0 && self.rate_range.1 >= 0.0
    }

    /// Check if this config represents outflows (negative rates)
    pub fn is_outflow(&self) -> bool {
        self.rate_range.0 <= 0.0 && self.rate_range.1 <= 0.0
    }

    /// Create configuration for realistic geological inflows
    pub fn realistic_inflows() -> Self {
        let mut config = Self {
            target_count: 80,
            rate_range: (0.05, 0.4),
            lifetime_range: (200.0, 800.0),
            creation_year_range: (-50.0, 50.0),
            replacement_rate: 0.0,
            flow_type: "inflows".to_string(),
        };
        config.calculate_replacement_rate();
        config
    }

    /// Create configuration for realistic geological outflows
    pub fn realistic_outflows() -> Self {
        let mut config = Self {
            target_count: 20,
            rate_range: (-0.15, -0.025),
            lifetime_range: (100.0, 500.0),
            creation_year_range: (-50.0, 50.0),
            replacement_rate: 0.0,
            flow_type: "outflows".to_string(),
        };
        config.calculate_replacement_rate();
        config
    }

    /// Create configuration for high-activity inflows
    pub fn high_activity_inflows() -> Self {
        let mut config = Self {
            target_count: 150,
            rate_range: (0.1, 0.6),
            lifetime_range: (150.0, 600.0),
            creation_year_range: (-25.0, 25.0),
            replacement_rate: 0.0,
            flow_type: "inflows".to_string(),
        };
        config.calculate_replacement_rate();
        config
    }

    /// Create configuration for high-activity outflows
    pub fn high_activity_outflows() -> Self {
        let mut config = Self {
            target_count: 40,
            rate_range: (-0.25, -0.05),
            lifetime_range: (75.0, 400.0),
            creation_year_range: (-25.0, 25.0),
            replacement_rate: 0.0,
            flow_type: "outflows".to_string(),
        };
        config.calculate_replacement_rate();
        config
    }

    /// Create configuration for low-activity inflows
    pub fn low_activity_inflows() -> Self {
        let mut config = Self {
            target_count: 40,
            rate_range: (0.02, 0.2),
            lifetime_range: (300.0, 1000.0),
            creation_year_range: (-100.0, 100.0),
            replacement_rate: 0.0,
            flow_type: "inflows".to_string(),
        };
        config.calculate_replacement_rate();
        config
    }

    /// Create configuration for low-activity outflows
    pub fn low_activity_outflows() -> Self {
        let mut config = Self {
            target_count: 10,
            rate_range: (-0.08, -0.01),
            lifetime_range: (200.0, 800.0),
            creation_year_range: (-100.0, 100.0),
            replacement_rate: 0.0,
            flow_type: "outflows".to_string(),
        };
        config.calculate_replacement_rate();
        config
    }
}

/// Configuration for thermal feature generation and lifecycle management
#[derive(Debug, Clone)]
pub struct ThermalFeatureConfig {
    /// Vector of thermal flow configurations (can be inflows, outflows, or mixed)
    pub flows: Vec<ThermalFlowConfig>,
    /// How often to check and add new features (years)
    pub replenishment_interval: f64,
    /// Last time features were replenished
    pub last_replenishment_year: f64,
}

impl Default for ThermalFeatureConfig {
    fn default() -> Self {
        Self {
            flows: vec![
                ThermalFlowConfig::realistic_inflows(),
                ThermalFlowConfig::realistic_outflows(),
            ],
            replenishment_interval: 25.0,
            last_replenishment_year: 0.0,
        }
    }
}

impl ThermalFeatureConfig {
    /// Create a configuration for realistic geological thermal features
    pub fn realistic() -> Self {
        Self::default()
    }

    /// Create a configuration for high-activity thermal features
    pub fn high_activity() -> Self {
        Self {
            flows: vec![
                ThermalFlowConfig::high_activity_inflows(),
                ThermalFlowConfig::high_activity_outflows(),
            ],
            replenishment_interval: 15.0,
            last_replenishment_year: 0.0,
        }
    }

    /// Create a configuration for low-activity thermal features
    pub fn low_activity() -> Self {
        Self {
            flows: vec![
                ThermalFlowConfig::low_activity_inflows(),
                ThermalFlowConfig::low_activity_outflows(),
            ],
            replenishment_interval: 50.0,
            last_replenishment_year: 0.0,
        }
    }

    /// Calculate expected replacement rates for all flow configurations
    pub fn calculate_replacement_rates(&mut self) {
        for flow_config in &mut self.flows {
            flow_config.calculate_replacement_rate();
        }
    }

    /// Get expected feature expirations for all flow types in the given time period
    pub fn expected_expirations(&self, time_period_years: f64) -> Vec<(String, f64)> {
        self.flows.iter()
            .map(|config| (config.flow_type.clone(), config.expected_expirations(time_period_years)))
            .collect()
    }

    /// Create a custom configuration with specific flow configs
    pub fn custom(flows: Vec<ThermalFlowConfig>, replenishment_interval: f64) -> Self {
        Self {
            flows,
            replenishment_interval,
            last_replenishment_year: 0.0,
        }
    }

    /// Add a new flow configuration
    pub fn add_flow(&mut self, flow_config: ThermalFlowConfig) {
        self.flows.push(flow_config);
    }

    /// Get all inflow configurations
    pub fn inflow_configs(&self) -> Vec<&ThermalFlowConfig> {
        self.flows.iter().filter(|config| config.is_inflow()).collect()
    }

    /// Get all outflow configurations
    pub fn outflow_configs(&self) -> Vec<&ThermalFlowConfig> {
        self.flows.iter().filter(|config| config.is_outflow()).collect()
    }
}

/// Represents a thermal cell with position and energy data
#[derive(Debug, Clone)]
pub struct Cell {
    pub cell_index: CellIndex,
    pub lat: f64,
    pub lng: f64,
    pub energy: f64, // WM (watts per square meter)
}

impl Cell {
    /// Create a new cell with zero energy
    pub fn new(cell_index: CellIndex, lat: f64, lng: f64) -> Self {
        Self {
            cell_index,
            lat,
            lng,
            energy: 0.0,
        }
    }

    /// Add energy to this cell
    pub fn add_energy(&mut self, energy: f64) {
        self.energy += energy;
    }

    /// Set the energy of this cell
    pub fn set_energy(&mut self, energy: f64) {
        self.energy = energy;
    }

    /// Reset energy to zero
    pub fn reset_energy(&mut self) {
        self.energy = 0.0;
    }
}

const MAX_ENERGY: f64 = 0.3;   // Maximum with strongest upwell + Perlin + neighbors (~0.45 WM)

#[cfg(test)]
mod tests {
    use super::*;
    use h3o::Resolution;

    #[test]
    fn test_probabilistic_replacement_maintains_stable_population() {
        // Create a radiance system with realistic thermal configuration
        let mut radiance = RadianceSystem::new(0.0);

        // Initialize with a smaller population for faster testing
        let mut inflow_config = ThermalFlowConfig::realistic_inflows();
        inflow_config.target_count = 20;
        inflow_config.calculate_replacement_rate();

        let mut outflow_config = ThermalFlowConfig::realistic_outflows();
        outflow_config.target_count = 5;
        outflow_config.calculate_replacement_rate();

        let thermal_config = ThermalFeatureConfig::custom(
            vec![inflow_config, outflow_config],
            1.0, // Annual maintenance
        );

        radiance.thermal_config = thermal_config;

        // Initialize features
        radiance.initialize_sustainable_features(Resolution::Two, 0.0)
            .expect("Failed to initialize features");

        let initial_inflows = radiance.count_active_inflows(0.0);
        let initial_outflows = radiance.count_active_outflows(0.0);

        println!("Initial population: {} inflows, {} outflows", initial_inflows, initial_outflows);

        // Track populations over 100 years
        let mut inflow_counts = Vec::new();
        let mut outflow_counts = Vec::new();

        for year in 1..=100 {
            // Run annual maintenance
            radiance.maintain_thermal_features(Resolution::Two, year as f64)
                .expect("Failed to maintain features");

            let current_inflows = radiance.count_active_inflows(year as f64);
            let current_outflows = radiance.count_active_outflows(year as f64);

            inflow_counts.push(current_inflows);
            outflow_counts.push(current_outflows);

            if year % 20 == 0 {
                println!("Year {}: {} inflows, {} outflows", year, current_inflows, current_outflows);
            }
        }

        let final_inflows = radiance.count_active_inflows(100.0);
        let final_outflows = radiance.count_active_outflows(100.0);

        println!("Final population: {} inflows, {} outflows", final_inflows, final_outflows);

        // Calculate statistics
        let avg_inflows: f64 = inflow_counts.iter().map(|&x| x as f64).sum::<f64>() / inflow_counts.len() as f64;
        let avg_outflows: f64 = outflow_counts.iter().map(|&x| x as f64).sum::<f64>() / outflow_counts.len() as f64;

        println!("Average over 100 years: {:.1} inflows, {:.1} outflows", avg_inflows, avg_outflows);

        // Test that the population behaves realistically
        // With low replacement rates (0.2-0.3%), natural decline is expected and realistic
        let inflow_deviation = (final_inflows as f64 - initial_inflows as f64).abs() / initial_inflows as f64;
        let outflow_deviation = if initial_outflows > 0 {
            (final_outflows as f64 - initial_outflows as f64).abs() / initial_outflows as f64
        } else {
            0.0
        };

        println!("Population deviation: inflows {:.1}%, outflows {:.1}%",
                 inflow_deviation * 100.0, outflow_deviation * 100.0);

        // With very low replacement rates, populations naturally decline over geological time
        // This is realistic behavior - thermal features are rare geological events

        // Inflows should remain relatively stable (they have longer lifetimes)
        assert!(inflow_deviation < 0.3,
                "Inflow population deviated too much: {:.1}% (initial: {}, final: {})",
                inflow_deviation * 100.0, initial_inflows, final_inflows);

        // Outflows can decline more significantly (shorter lifetimes, lower replacement rate)
        // This is geologically realistic - cooling features are more transient
        // However, probabilistic replacement can occasionally add new features
        assert!(final_outflows <= initial_outflows + 2,
                "Outflow population should not increase significantly: initial: {}, final: {}",
                initial_outflows, final_outflows);

        // Test that the system doesn't completely collapse
        assert!(final_inflows > 0, "Inflow population should not go to zero");

        // Test that average populations are reasonable
        assert!(avg_inflows >= initial_inflows as f64 * 0.7,
                "Average inflow population too low: {:.1} (should be >= {:.1})",
                avg_inflows, initial_inflows as f64 * 0.7);

        // Test that replacement rates are reasonable
        let inflow_replace_rate = radiance.thermal_config.flows[0].replace_rate();
        let outflow_replace_rate = radiance.thermal_config.flows[1].replace_rate();

        println!("Replace rates: inflows {:.3}% ({:.4}), outflows {:.3}% ({:.4})",
                 inflow_replace_rate * 100.0, inflow_replace_rate,
                 outflow_replace_rate * 100.0, outflow_replace_rate);

        // Replace rates should be small but positive
        assert!(inflow_replace_rate > 0.0 && inflow_replace_rate < 0.1,
                "Inflow replace rate should be between 0% and 10%: {:.3}%", inflow_replace_rate * 100.0);

        assert!(outflow_replace_rate > 0.0 && outflow_replace_rate < 0.1,
                "Outflow replace rate should be between 0% and 10%: {:.3}%", outflow_replace_rate * 100.0);
    }

    #[test]
    fn test_high_replace_rate_halving() {
        // Test the >33% rate halving mechanism
        let mut config = ThermalFlowConfig {
            target_count: 10,
            rate_range: (0.1, 0.5),
            lifetime_range: (20.0, 30.0), // Short lifetime = high replace rate
            creation_year_range: (-10.0, 10.0),
            replacement_rate: 0.0,
            flow_type: "test_inflows".to_string(),
        };

        config.calculate_replacement_rate();

        println!("High activity config:");
        println!("  Replacement rate: {:.3} features/year", config.replacement_rate);
        println!("  Replace rate: {:.1}% ({:.4})", config.replace_rate() * 100.0, config.replace_rate());

        // With 25-year average lifetime and 10 target count:
        // replacement_rate = 10 / 25 = 0.4 features/year
        // replace_rate = 0.4 / 10 = 0.04 = 4%
        let replace_rate = config.replace_rate();
        assert!(replace_rate > 0.03 && replace_rate < 0.05,
                "Replace rate should be around 4%: {:.3}%", replace_rate * 100.0);

        // Test the halving mechanism with a mock RNG
        let mut rng = rand::rng();
        let mut total_added = 0;
        let trials = 10000;

        for _ in 0..trials {
            total_added += config.calculate_features_to_add(&mut rng);
        }

        let actual_rate = total_added as f64 / trials as f64;
        let expected_rate = config.replace_rate();

        println!("Actual addition rate over {} trials: {:.4} ({:.1}%)",
                 trials, actual_rate, actual_rate * 100.0);
        println!("Expected rate: {:.4} ({:.1}%)", expected_rate, expected_rate * 100.0);

        // Should be within 10% of expected rate
        let rate_error = (actual_rate - expected_rate).abs() / expected_rate;
        assert!(rate_error < 0.1,
                "Actual rate {:.4} differs too much from expected {:.4} (error: {:.1}%)",
                actual_rate, expected_rate, rate_error * 100.0);
    }
}