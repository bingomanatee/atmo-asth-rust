/// Thermal Expansion System
/// 
/// Handles temperature-dependent density changes that affect thermal conductivity.
/// Implements realistic thermal expansion effects where hotter material becomes
/// less dense and conducts heat more efficiently.

use crate::energy_mass_composite::{EnergyMassComposite, MaterialCompositeType};
use crate::global_thermal::thermal_layer::ThermalLayer;

/// Thermal expansion calculator for realistic density-temperature relationships
#[derive(Debug, Clone)]
pub struct ThermalExpansionCalculator {
    /// Reference temperature for expansion calculations (K)
    reference_temp_k: f64,
}

impl ThermalExpansionCalculator {
    pub fn new() -> Self {
        Self {
            reference_temp_k: 293.15, // 20°C reference temperature
        }
    }
    
    /// Calculate temperature-adjusted density based on thermal expansion
    pub fn calculate_thermal_density(
        &self,
        layer: &ThermalLayer,
        current_temp_k: f64,
    ) -> f64 {
        let base_density = layer.energy_mass.density_kgm3();
        let thermal_expansivity = self.get_thermal_expansivity(layer);
        
        // Calculate temperature difference from reference
        let temp_diff = current_temp_k - self.reference_temp_k;
        
        // Linear thermal expansion: ΔL/L = α × ΔT
        // For volumetric expansion: ΔV/V = 3α × ΔT (assuming isotropic expansion)
        let volumetric_expansivity = 3.0 * thermal_expansivity;
        
        // Calculate volume change ratio
        let volume_expansion_ratio = 1.0 + (volumetric_expansivity * temp_diff);
        
        // Density is inversely proportional to volume (mass conserved)
        let thermal_density = base_density / volume_expansion_ratio.max(0.1); // Prevent division by zero
        
        thermal_density
    }
    
    /// Calculate thermal expansion enhanced thermal conductivity
    /// Less dense (hotter) material generally conducts heat faster due to:
    /// 1. Increased molecular motion
    /// 2. Reduced scattering from lattice defects
    /// 3. Enhanced convective-like effects in micro-structure
    pub fn calculate_thermal_conductivity_enhancement(
        &self,
        layer: &ThermalLayer,
        current_temp_k: f64,
    ) -> f64 {
        let base_conductivity = layer.energy_mass.thermal_conductivity();
        let thermal_density = self.calculate_thermal_density(layer, current_temp_k);
        let reference_density = layer.energy_mass.density_kgm3();
        
        // Calculate density ratio (lower ratio = less dense = hotter)
        let density_ratio = thermal_density / reference_density;
        
        // Temperature-dependent enhancement factor
        let temp_enhancement = self.calculate_temperature_enhancement(current_temp_k, layer);
        
        // Density-dependent enhancement (lower density = higher conductivity)
        let density_enhancement = self.calculate_density_enhancement(density_ratio);
        
        // Combined enhancement
        let total_enhancement = temp_enhancement * density_enhancement;
        
        base_conductivity * total_enhancement
    }
    
    /// Get thermal expansivity from material properties
    fn get_thermal_expansivity(&self, layer: &ThermalLayer) -> f64 {
        // Get from material properties - these values are now in materials.json
        // For now, use typical values based on material type
        match layer.energy_mass.material_composite_type() {
            MaterialCompositeType::Basaltic => {
                // Use values from materials.json: solid 1.0e-5, liquid 2.0e-5
                if layer.temperature_k() > 1473.0 { // Above melting point
                    2.0e-5
                } else {
                    1.0e-5
                }
            },
            MaterialCompositeType::Silicate => {
                // Use values from materials.json: solid 1.2e-5, liquid 2.2e-5
                if layer.temperature_k() > 1700.0 { // Above melting point
                    2.2e-5
                } else {
                    1.2e-5
                }
            },
            MaterialCompositeType::Metallic => {
                // Use values from materials.json: solid 1.2e-5, liquid 2.0e-5
                if layer.temperature_k() > 1800.0 { // Above melting point
                    2.0e-5
                } else {
                    1.2e-5
                }
            },
            MaterialCompositeType::Icy => {
                // Use values from materials.json: solid 5.0e-5, liquid 2.1e-4
                if layer.temperature_k() > 273.15 { // Above melting point
                    2.1e-4
                } else {
                    5.0e-5
                }
            },
            MaterialCompositeType::Air => {
                // Gases have high thermal expansion
                3.4e-3 // 1/T for ideal gas
            },
            _ => 1.0e-5, // Default value for unknown materials
        }
    }
    
    /// Calculate temperature-dependent conductivity enhancement
    fn calculate_temperature_enhancement(&self, current_temp_k: f64, layer: &ThermalLayer) -> f64 {
        let material_type = layer.energy_mass.material_composite_type();
        
        match material_type {
            MaterialCompositeType::Air => {
                // Gas conductivity increases with temperature
                let temp_ratio = current_temp_k / 273.15; // Relative to 0°C
                temp_ratio.sqrt() // Kinetic theory scaling
            },
            _ => {
                // Solid/liquid materials: moderate temperature dependence
                let temp_excess = (current_temp_k - 293.15).max(0.0); // Above room temperature
                let enhancement_factor = 1.0 + (temp_excess * 1.0e-4); // Small positive enhancement
                enhancement_factor.min(2.0) // Cap at 2x enhancement
            }
        }
    }
    
    /// Calculate density-dependent conductivity enhancement
    fn calculate_density_enhancement(&self, density_ratio: f64) -> f64 {
        // Lower density (higher temperature) enhances conductivity
        // This represents micro-convective effects and increased molecular mobility
        
        if density_ratio < 1.0 {
            // Expanded (less dense) material conducts better
            let expansion_factor = 1.0 / density_ratio;
            let enhancement = 1.0 + (expansion_factor - 1.0) * 0.5; // 50% of expansion translates to conductivity
            enhancement.min(3.0) // Cap at 3x enhancement
        } else {
            // Compressed material conducts slightly worse
            let compression_factor = density_ratio;
            let reduction = 1.0 / (1.0 + (compression_factor - 1.0) * 0.2); // 20% reduction factor
            reduction.max(0.5) // Minimum 50% of original conductivity
        }
    }
    
    /// Update layer density based on current temperature
    pub fn update_layer_thermal_density(&self, layer: &mut ThermalLayer) {
        let current_temp = layer.temperature_k();
        let new_density = self.calculate_thermal_density(layer, current_temp);
        
        // Update the layer's volume to reflect new density while preserving mass
        let current_mass = layer.mass_kg();
        let surface_area = layer.surface_area_km2 * 1e6; // Convert to m²
        let height_m = layer.height_km * 1000.0; // Convert to m
        
        // Calculate new volume based on new density
        let new_volume_m3 = current_mass / new_density;
        let new_height_m = new_volume_m3 / surface_area;
        let new_height_km = new_height_m / 1000.0;
        
        // Update layer height (this affects volume and thus density)
        layer.height_km = new_height_km;
    }
    
    /// Calculate enhanced thermal diffusivity including thermal expansion effects
    pub fn calculate_enhanced_thermal_diffusivity(
        &self,
        layer: &ThermalLayer,
        current_temp_k: f64,
    ) -> f64 {
        let enhanced_conductivity = self.calculate_thermal_conductivity_enhancement(layer, current_temp_k);
        let thermal_density = self.calculate_thermal_density(layer, current_temp_k);
        let specific_heat = layer.energy_mass.specific_heat_j_kg_k();
        
        // α = k/(ρ × c) with thermal expansion effects
        enhanced_conductivity / (thermal_density * specific_heat)
    }
    
    /// Get thermal expansion statistics for reporting
    pub fn get_expansion_stats(&self, layer: &ThermalLayer) -> ThermalExpansionStats {
        let current_temp = layer.temperature_k();
        let reference_density = layer.energy_mass.density_kgm3();
        let thermal_density = self.calculate_thermal_density(layer, current_temp);
        let conductivity_enhancement = self.calculate_thermal_conductivity_enhancement(layer, current_temp);
        let base_conductivity = layer.energy_mass.thermal_conductivity();
        
        ThermalExpansionStats {
            temperature_k: current_temp,
            reference_density_kg_m3: reference_density,
            thermal_density_kg_m3: thermal_density,
            density_change_percent: ((thermal_density - reference_density) / reference_density) * 100.0,
            base_conductivity_w_m_k: base_conductivity,
            enhanced_conductivity_w_m_k: conductivity_enhancement,
            conductivity_enhancement_factor: conductivity_enhancement / base_conductivity,
            thermal_expansivity: self.get_thermal_expansivity(layer),
        }
    }
}

/// Statistics for thermal expansion effects
#[derive(Debug, Clone)]
pub struct ThermalExpansionStats {
    pub temperature_k: f64,
    pub reference_density_kg_m3: f64,
    pub thermal_density_kg_m3: f64,
    pub density_change_percent: f64,
    pub base_conductivity_w_m_k: f64,
    pub enhanced_conductivity_w_m_k: f64,
    pub conductivity_enhancement_factor: f64,
    pub thermal_expansivity: f64,
}

impl Default for ThermalExpansionCalculator {
    fn default() -> Self {
        Self::new()
    }
}