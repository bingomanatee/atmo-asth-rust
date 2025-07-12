/// Heat Plume Entity System
/// 
/// Models thermal plumes as discrete entities that span across layers,
/// transporting energy and material from foundry layers to the surface.
/// Each plume has a single coherent behavior and persists across simulation steps.

use crate::global_thermal::thermal_layer::ThermalLayer;
use crate::energy_mass_composite::{MaterialCompositeType, EnergyMassComposite};
use h3o::CellIndex;
use std::collections::HashMap;

/// A single heat plume entity spanning multiple layers
#[derive(Debug, Clone)]
pub struct HeatPlume {
    /// Unique identifier for this plume
    pub id: String,
    
    /// Cell where this plume is located
    pub cell_id: CellIndex,
    
    /// Current position (layer index from bottom)
    pub current_layer: usize,
    
    /// Target layer (usually surface or atmosphere)
    pub target_layer: usize,
    
    /// Temperature of the plume (K)
    pub temperature_k: f64,
    
    /// Energy content of the plume (J)
    pub energy_joules: f64,
    
    /// Mass of material in the plume (kg)
    pub mass_kg: f64,
    
    /// Material type being transported
    pub material_type: MaterialCompositeType,
    
    /// Velocity of plume rise (m/s)
    pub velocity_m_s: f64,
    
    /// Radius of plume influence (m)
    pub radius_m: f64,
    
    /// Age of plume (simulation steps)
    pub age: u32,
    
    /// Maximum lifespan before dissipation
    pub max_age: u32,
    
    /// Current strength factor (0.0 to 1.0)
    pub strength: f64,
    
    /// Layers this plume currently affects
    pub affected_layers: Vec<usize>,
    
    /// Energy deposited in each layer this step
    pub layer_energy_deposits: HashMap<usize, f64>,
    
    /// Material transported to each layer this step
    pub layer_material_deposits: HashMap<usize, f64>,
}

impl HeatPlume {
    /// Create a new heat plume from a foundry layer
    pub fn new_from_foundry(
        cell_id: CellIndex,
        foundry_layer_idx: usize,
        foundry_layer: &ThermalLayer,
        plume_id: u32,
    ) -> Self {
        let id = format!("{:?}_plume_{}", cell_id, plume_id);
        
        // Initial plume properties based on foundry conditions
        let temperature_k = foundry_layer.temperature_k();
        let energy_fraction = 0.05; // 5% of foundry energy per plume (increased for more impact)
        let mass_fraction = 0.01;   // 1% of foundry mass per plume (increased for more material transport)
        
        let energy_joules = foundry_layer.energy_mass.energy_joules * energy_fraction;
        let mass_kg = foundry_layer.mass_kg() * mass_fraction;
        
        // Calculate initial velocity based on buoyancy
        let velocity_m_s = Self::calculate_buoyancy_velocity(temperature_k, mass_kg);
        
        Self {
            id,
            cell_id,
            current_layer: foundry_layer_idx,
            target_layer: 0, // Surface layer
            temperature_k,
            energy_joules,
            mass_kg,
            material_type: foundry_layer.energy_mass.material_composite_type(),
            velocity_m_s,
            radius_m: 1000.0, // 1km radius influence
            age: 0,
            max_age: 300, // 300 simulation steps max lifespan (reduced to deposit energy faster)
            strength: 1.0,
            affected_layers: vec![foundry_layer_idx],
            layer_energy_deposits: HashMap::new(),
            layer_material_deposits: HashMap::new(),
        }
    }
    
    /// Calculate buoyancy-driven velocity for plume rise
    fn calculate_buoyancy_velocity(temperature_k: f64, mass_kg: f64) -> f64 {
        // Simplified buoyancy calculation
        // v = sqrt(2 * g * h * Δρ/ρ)
        let g = 9.81; // m/s²
        let height_scale = 1000.0; // 1km characteristic height
        let reference_density = 3200.0; // kg/m³
        
        // Temperature-dependent density difference
        let density_ratio = (temperature_k - 1500.0) / 1500.0; // Normalized excess temperature
        let buoyancy_factor = density_ratio.max(0.0) * 0.1; // Max 10% density difference
        
        (2.0 * g * height_scale * buoyancy_factor).sqrt()
    }
    
    /// Update plume position and properties for one simulation step
    pub fn update_step(&mut self, layers: &[(ThermalLayer, ThermalLayer)], years_per_step: u32) -> bool {
        self.age += 1;
        
        // Check if plume should dissipate
        if self.age >= self.max_age || self.strength < 0.1 {
            return false; // Plume should be removed
        }
        
        // Clear previous deposits
        self.layer_energy_deposits.clear();
        self.layer_material_deposits.clear();
        
        // Calculate movement distance this step
        let seconds_per_step = years_per_step as f64 * 365.25 * 24.0 * 3600.0;
        let distance_m = self.velocity_m_s * seconds_per_step;
        
        // Update plume position based on layer thickness
        self.move_through_layers(layers, distance_m);
        
        // Update affected layers based on current position
        self.update_affected_layers(layers);
        
        // Deposit energy and material in affected layers
        self.deposit_energy_and_material(layers);
        
        // Update plume properties (cooling, weakening)
        self.update_physical_properties();
        
        true // Plume continues to exist
    }
    
    /// Move plume through layers based on distance traveled
    fn move_through_layers(&mut self, layers: &[(ThermalLayer, ThermalLayer)], distance_m: f64) {
        let mut remaining_distance = distance_m;
        
        while remaining_distance > 0.0 && self.current_layer > self.target_layer {
            if let Some((current_layer, _)) = layers.get(self.current_layer) {
                let layer_thickness_m = current_layer.height_km * 1000.0;
                
                if remaining_distance >= layer_thickness_m {
                    // Move to next layer up
                    remaining_distance -= layer_thickness_m;
                    self.current_layer = self.current_layer.saturating_sub(1);
                } else {
                    // Partial movement within current layer
                    break;
                }
            } else {
                break;
            }
        }
    }
    
    /// Update which layers are affected by this plume
    fn update_affected_layers(&mut self, layers: &[(ThermalLayer, ThermalLayer)]) {
        self.affected_layers.clear();
        
        // Primary affected layer (where plume currently is)
        self.affected_layers.push(self.current_layer);
        
        // Secondary affected layers (radius of influence)
        let influence_layers = (self.radius_m / 10000.0).ceil() as usize; // ~10km per layer
        
        for offset in 1..=influence_layers {
            // Layer above
            if self.current_layer >= offset {
                let layer_above = self.current_layer - offset;
                if layer_above >= self.target_layer {
                    self.affected_layers.push(layer_above);
                }
            }
            
            // Layer below
            let layer_below = self.current_layer + offset;
            if layer_below < layers.len() {
                self.affected_layers.push(layer_below);
            }
        }
    }
    
    /// Deposit energy and material in affected layers
    fn deposit_energy_and_material(&mut self, layers: &[(ThermalLayer, ThermalLayer)]) {
        let total_affected = self.affected_layers.len() as f64;
        if total_affected == 0.0 {
            return;
        }
        
        // Calculate temperature-differential-based aggressive heat dumping
        let base_energy_deposition_rate = 0.08; // 8% base rate per step
        let temperature_differential_multiplier = self.calculate_temperature_differential_multiplier(layers);
        let energy_deposition_rate = (base_energy_deposition_rate * temperature_differential_multiplier).min(0.25); // Cap at 25% per step
        let total_energy_to_deposit = self.energy_joules * energy_deposition_rate;
        
        // Material deposition scales with temperature differential too
        let base_material_deposition_rate = 0.04; // 4% base rate per step
        let material_deposition_rate = (base_material_deposition_rate * temperature_differential_multiplier).min(0.15); // Cap at 15% per step
        let total_material_to_deposit = self.mass_kg * material_deposition_rate;
        
        // Calculate total temperature-weighted influence for proportional distribution
        let mut total_temp_weighted_influence = 0.0;
        let mut layer_influences = Vec::new();
        
        for &layer_idx in &self.affected_layers {
            if let Some((layer, _)) = layers.get(layer_idx) {
                // Base influence factor based on distance from plume center
                let distance_factor = if layer_idx == self.current_layer {
                    1.0 // Full influence at plume center
                } else {
                    let distance = (layer_idx as i32 - self.current_layer as i32).abs() as f64;
                    (1.0 / (1.0 + distance)).max(0.1) // Inverse distance with minimum
                };
                
                // Temperature differential factor - prefer dumping heat into cooler layers
                let layer_temp = layer.temperature_k();
                let temp_differential = (self.temperature_k - layer_temp).max(0.0);
                let temp_factor = 1.0 + (temp_differential / 200.0); // More energy to cooler layers
                
                // Combined influence factor
                let combined_influence = distance_factor * temp_factor;
                total_temp_weighted_influence += combined_influence;
                layer_influences.push((layer_idx, combined_influence));
            }
        }
        
        // Distribute energy and material based on temperature-weighted influence
        for (layer_idx, influence_factor) in layer_influences {
            if total_temp_weighted_influence > 0.0 {
                // Proportional energy deposit based on temperature-weighted influence
                let energy_deposit = total_energy_to_deposit * influence_factor / total_temp_weighted_influence;
                self.layer_energy_deposits.insert(layer_idx, energy_deposit);
                
                // Proportional material deposit
                let material_deposit = total_material_to_deposit * influence_factor / total_temp_weighted_influence;
                self.layer_material_deposits.insert(layer_idx, material_deposit);
            }
        }
        
        // Remove deposited energy and material from plume
        self.energy_joules -= total_energy_to_deposit;
        self.mass_kg -= total_material_to_deposit;
    }
    
    /// Calculate temperature differential multiplier for aggressive heat dumping
    /// Returns a multiplier (1.0 to 5.0) based on how much hotter the plume is than surrounding layers
    fn calculate_temperature_differential_multiplier(&self, layers: &[(ThermalLayer, ThermalLayer)]) -> f64 {
        let mut max_temp_differential = 0.0;
        
        // Check temperature difference between plume and all affected layers
        for &layer_idx in &self.affected_layers {
            if let Some((layer, _)) = layers.get(layer_idx) {
                let layer_temp = layer.temperature_k();
                let temp_differential = self.temperature_k - layer_temp;
                
                if temp_differential > max_temp_differential {
                    max_temp_differential = temp_differential;
                }
            }
        }
        
        // Calculate aggressive heat dumping multiplier based on temperature differential
        // Larger temperature differences = much more aggressive heat transfer
        let base_multiplier = 1.0;
        let differential_factor = if max_temp_differential > 0.0 {
            // Scale aggressively: 100K difference = 2x, 500K = 3x, 1000K = 4x, 1500K+ = 5x
            let normalized_diff = max_temp_differential / 300.0; // 300K reference
            1.0 + (normalized_diff * 1.5).min(4.0) // Cap additional multiplier at 4x
        } else {
            0.0 // No heat dumping if plume is cooler than layers
        };
        
        base_multiplier + differential_factor
    }
    
    /// Update plume physical properties (cooling, weakening)
    fn update_physical_properties(&mut self) {
        // Cooling due to energy loss
        if self.mass_kg > 0.0 {
            let specific_heat = 1200.0; // J/kg·K (approximate for molten rock)
            let energy_per_kg = self.energy_joules / self.mass_kg;
            self.temperature_k = (energy_per_kg / specific_heat).max(300.0); // Min 300K
        }
        
        // Strength decay over time
        self.strength *= 0.998; // 0.2% decay per step
        
        // Velocity adjustment based on temperature and strength
        self.velocity_m_s = Self::calculate_buoyancy_velocity(self.temperature_k, self.mass_kg) * self.strength;
        
        // Radius adjustment (plumes tend to widen as they rise)
        self.radius_m = self.radius_m * 1.001; // Slight widening
    }
    
    /// Check if plume should initiate from a foundry layer
    pub fn should_initiate_from_foundry(foundry_layer: &ThermalLayer, threshold_temp: f64) -> bool {
        foundry_layer.temperature_k() > threshold_temp && foundry_layer.is_foundry
    }
    
    /// Get energy to deposit in a specific layer
    pub fn get_energy_deposit(&self, layer_idx: usize) -> f64 {
        self.layer_energy_deposits.get(&layer_idx).copied().unwrap_or(0.0)
    }
    
    /// Get material to deposit in a specific layer
    pub fn get_material_deposit(&self, layer_idx: usize) -> f64 {
        self.layer_material_deposits.get(&layer_idx).copied().unwrap_or(0.0)
    }
    
    /// Get current plume statistics for reporting
    pub fn get_stats(&self) -> PlumeStats {
        PlumeStats {
            id: self.id.clone(),
            cell_id: self.cell_id,
            current_layer: self.current_layer,
            temperature_k: self.temperature_k,
            energy_joules: self.energy_joules,
            mass_kg: self.mass_kg,
            velocity_m_s: self.velocity_m_s,
            strength: self.strength,
            age: self.age,
            affected_layers: self.affected_layers.len(),
        }
    }
    
    /// Get total energy being deposited this step (for debugging)
    pub fn get_total_energy_deposition(&self) -> f64 {
        self.layer_energy_deposits.values().sum()
    }
    
    /// Get total material being deposited this step (for debugging)
    pub fn get_total_material_deposition(&self) -> f64 {
        self.layer_material_deposits.values().sum()
    }
}

/// Statistics for a heat plume
#[derive(Debug, Clone)]
pub struct PlumeStats {
    pub id: String,
    pub cell_id: CellIndex,
    pub current_layer: usize,
    pub temperature_k: f64,
    pub energy_joules: f64,
    pub mass_kg: f64,
    pub velocity_m_s: f64,
    pub strength: f64,
    pub age: u32,
    pub affected_layers: usize,
}

/// Collection of heat plumes for a simulation cell
#[derive(Debug, Clone)]
pub struct CellPlumeCollection {
    pub cell_id: CellIndex,
    pub plumes: Vec<HeatPlume>,
    pub next_plume_id: u32,
}

impl CellPlumeCollection {
    pub fn new(cell_id: CellIndex) -> Self {
        Self {
            cell_id,
            plumes: Vec::new(),
            next_plume_id: 0,
        }
    }
    
    /// Add a new plume to the collection
    pub fn add_plume(&mut self, plume: HeatPlume) {
        self.plumes.push(plume);
        self.next_plume_id += 1;
    }
    
    /// Update all plumes in the collection
    pub fn update_all_plumes(&mut self, layers: &[(ThermalLayer, ThermalLayer)], years_per_step: u32) {
        // Update existing plumes and remove dead ones
        self.plumes.retain_mut(|plume| plume.update_step(layers, years_per_step));
    }
    
    /// Get total energy deposits for all layers
    pub fn get_total_energy_deposits(&self) -> HashMap<usize, f64> {
        let mut total_deposits = HashMap::new();
        
        for plume in &self.plumes {
            for (&layer_idx, &energy) in &plume.layer_energy_deposits {
                *total_deposits.entry(layer_idx).or_insert(0.0) += energy;
            }
        }
        
        total_deposits
    }
    
    /// Get total material deposits for all layers
    pub fn get_total_material_deposits(&self) -> HashMap<usize, f64> {
        let mut total_deposits = HashMap::new();
        
        for plume in &self.plumes {
            for (&layer_idx, &material) in &plume.layer_material_deposits {
                *total_deposits.entry(layer_idx).or_insert(0.0) += material;
            }
        }
        
        total_deposits
    }
    
    /// Get statistics for all plumes
    pub fn get_all_stats(&self) -> Vec<PlumeStats> {
        self.plumes.iter().map(|plume| plume.get_stats()).collect()
    }
    
    /// Check if new plumes should be created from foundry layers
    pub fn check_and_create_plumes(&mut self, layers: &[(ThermalLayer, ThermalLayer)], threshold_temp: f64) {
        for (layer_idx, (layer, _)) in layers.iter().enumerate() {
            if HeatPlume::should_initiate_from_foundry(layer, threshold_temp) {
                // Limit plume creation rate (max 1 per foundry layer per 5 steps)
                let existing_plumes_from_layer = self.plumes.iter()
                    .filter(|p| p.current_layer == layer_idx && p.age < 5)
                    .count();
                
                if existing_plumes_from_layer == 0 {
                    let new_plume = HeatPlume::new_from_foundry(
                        self.cell_id,
                        layer_idx,
                        layer,
                        self.next_plume_id,
                    );
                    self.add_plume(new_plume);
                }
            }
        }
    }
}