use crate::energy_mass_composite::{EnergyMassComposite, MaterialCompositeType, MaterialPhase};
/// Science-backed Fourier heat transfer system for thermal diffusion
/// Implements proper Fourier's law with material properties and geometric constraints
/// Updated to work with new ThermalLayer arrays and (current, next) tuple structure
/// Enhanced with thermal expansion effects for density-dependent thermal conductivity
use crate::global_thermal::thermal_layer::ThermalLayer;
use crate::global_thermal::thermal_expansion::ThermalExpansionCalculator;
use crate::thermal_pressure_cache::ThermalPressureCache;
use std::collections::HashMap;
use std::sync::Mutex;
use once_cell::sync::Lazy;
use crate::constants::{KM_TO_M, SECONDS_PER_YEAR};

/// Physical constants for Fourier heat transfer calculations
    /// Minimum temperature difference for heat transfer (K)
    pub const MIN_TEMP_DIFF_K: f64 = 0.1;

    /// Maximum energy transfer fraction per timestep for numerical stability
    /// Increased for geological equilibration over long timescales
    pub const MAX_ENERGY_TRANSFER_FRACTION: f64 = 0.25;

    /// Enhanced transfer rate per year for geological equilibration
    /// Increased to support exponential thermal pressure convection
    pub const BASE_TRANSFER_RATE_PER_YEAR: f64 = 0.05; // 5% per year for enhanced foundry heat transfer
    
    /// Temperature threshold for exponential convection effects (K)
    pub const CONVECTION_THRESHOLD_K: f64 = 800.0;
    
    /// Exponential scaling factor for high-temperature convection
    pub const CONVECTION_EXPONENTIAL_FACTOR: f64 = 1.5;



#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct DirectionalHeatFlowCacheKey {
    from_material_type: MaterialCompositeType,
    from_phase: MaterialPhase,
    from_density_rounded: i32,  // Current density rounded to nearest 0.1 kg/m³
    to_material_type: MaterialCompositeType,
    to_phase: MaterialPhase,
    to_density_rounded: i32,  // Current density rounded to nearest 0.1 kg/m³
    temp_diff_rounded: i32,  // Temperature difference rounded to nearest 5K
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct DensityAdjustedConductivityCacheKey {
    material_type: MaterialCompositeType,
    phase: MaterialPhase,
    current_density_rounded: i32,  // Current density rounded to nearest 0.1 kg/m³
}

#[derive(Debug, Clone)]
pub struct FourierThermalTransfer {
    pub years: f64,
    thermal_expansion_calculator: ThermalExpansionCalculator,
    thermal_pressure_cache: ThermalPressureCache,
}

impl FourierThermalTransfer {
    pub fn new(years: f64) -> Self {
        let thermal_pressure_cache = ThermalPressureCache::new(years);
        thermal_pressure_cache.initialize_cache();
        
        Self { 
            years,
            thermal_expansion_calculator: ThermalExpansionCalculator::new(),
            thermal_pressure_cache,
        }
    }



    
    /// Calculate thermal pressure enhancement factor based on temperature difference
    /// Uses the ratio of thermal pressures for a simpler, more intuitive approach
    pub fn calculate_exponential_thermal_pressure(
        &self,
        lower_temp: f64,
        higher_temp: f64,
    ) -> f64 {

        if (higher_temp - lower_temp).abs() < 2.0 {
            return 1.0;
        }
        // Calculate thermal pressure for both temperatures
        let higher_thermal_pressure = self.calculate_thermal_pressure_coefficient_uncached(higher_temp);
        let lower_thermal_pressure = self.calculate_thermal_pressure_coefficient_uncached(lower_temp);
        
        // Use the ratio of thermal pressures as the enhancement factor
        // Higher temperature differences create stronger pressure gradients
        let pressure_ratio = if lower_thermal_pressure > 0.0 {
            higher_thermal_pressure / lower_thermal_pressure
        } else {
            higher_thermal_pressure.max(1.0) // Fallback to absolute pressure if lower is zero
        };
        
        // Apply reasonable bounds to prevent numerical instability
        pressure_ratio.clamp(0.1, 4.0)
    }
    
    /// Calculate thermal pressure coefficient based on temperature (uncached)
    /// Exponential scaling makes higher temperatures much more "pressure-active"
    fn calculate_thermal_pressure_coefficient_uncached(&self, temp_k: f64) -> f64 {
        if temp_k <= CONVECTION_THRESHOLD_K {
            // Below threshold temperature - minimal thermal pressure
             (temp_k / CONVECTION_THRESHOLD_K).powf(0.5) // Gentle scaling below threshold
        } else {
            // Above threshold temperature - exponential thermal pressure
            let temp_excess = temp_k - CONVECTION_THRESHOLD_K;
             1.0 + (temp_excess / 400.0).powf(CONVECTION_EXPONENTIAL_FACTOR) // Very strong exponential
        }.clamp(0.1, 50.0)
    }


    /// Calculate density-adjusted thermal conductivity based on pressure compaction (uncached)
    /// Uses 200km asthenosphere depth as baseline "norm" for heat transfer
    /// Calculate density-adjusted conductivity including thermal expansion effects
    /// Lower pressures (vacuum effect) enhance heat transfer, higher pressures reduce it
    /// Thermal expansion from higher temperatures reduces density and enhances conductivity
    fn calculate_density_adjusted_conductivity_uncached(
        &self,
        base_conductivity: f64,
        current_density_kg_m3: f64,
        default_density_kg_m3: f64,
        thermal_layer: Option<&ThermalLayer>,
    ) -> f64 {
        if default_density_kg_m3 <= 0.0 {
            return base_conductivity;
        }

        // Apply thermal expansion effects if layer is provided
        let (effective_conductivity, effective_density) = if let Some(layer) = thermal_layer {
            let temp = layer.temperature_k();
            let thermal_enhanced_conductivity = self.thermal_expansion_calculator
                .calculate_thermal_conductivity_enhancement(layer, temp);
            let thermal_density = self.thermal_expansion_calculator
                .calculate_thermal_density(layer, temp);
            (thermal_enhanced_conductivity, thermal_density)
        } else {
            (base_conductivity, current_density_kg_m3)
        };

        // Calculate density ratio relative to surface density using thermal-adjusted density
        let surface_density_ratio = effective_density / default_density_kg_m3;

        // Assume 200km asthenosphere depth has ~2.5x surface density as "normal" baseline
        let asthenosphere_baseline_ratio = 2.5;

        // Calculate pressure-normalized ratio (200km depth = 1.0 baseline)
        let pressure_normalized_ratio = surface_density_ratio / asthenosphere_baseline_ratio;

        // Apply exponentially flattened pressure effects for geological equilibration
        // Use logarithmic scaling to dramatically reduce pressure influence
        let pressure_deviation = (pressure_normalized_ratio - 1.0).abs();
        let flattened_deviation = pressure_deviation.ln_1p() * 1.0; // Logarithmic flattening with tiny coefficient

        let conductivity_multiplier = if pressure_normalized_ratio < 1.0 {
            // Lower pressure than 200km = minimal vacuum effect
            // Exponentially flattened: 1.0 + ln(1 + deviation) * 0.01
            // At surface (ratio ~0.4, deviation 0.6): multiplier = 1.0 + ln(1.6) * 0.01 ≈ 1.005x
            1.0 + flattened_deviation
        } else {
            // Higher pressure than 200km = minimal compression effect
            // Exponentially flattened: 1.0 / (1.0 + ln(1 + deviation) * 0.01)
            // At 400km (ratio ~2.0, deviation 1.0): multiplier = 1.0 / (1.0 + ln(2) * 0.01) ≈ 0.993x
            1.0 / (1.0 + flattened_deviation)
        };

        // Apply the same multiplier for both incoming and outgoing (symmetric heat transfer)
        // Use the thermal-enhanced conductivity as the base, then apply pressure effects
        effective_conductivity * conductivity_multiplier
    }


    /// Create cache key for directional heat flow
    fn create_directional_heat_flow_cache_key(
        from_energy_mass: &dyn EnergyMassComposite,
        to_energy_mass: &dyn EnergyMassComposite,
        temp_diff: f64,
    ) -> DirectionalHeatFlowCacheKey {
        DirectionalHeatFlowCacheKey {
            from_material_type: from_energy_mass.material_composite_type(),
            from_phase: from_energy_mass.phase(),
            from_density_rounded: (from_energy_mass.density_kgm3() * 10.0).round() as i32,  // Round to nearest 0.1
            to_material_type: to_energy_mass.material_composite_type(),
            to_phase: to_energy_mass.phase(),
            to_density_rounded: (to_energy_mass.density_kgm3() * 10.0).round() as i32,  // Round to nearest 0.1
            temp_diff_rounded: (temp_diff / 5.0).round() as i32 * 5,  // Round to nearest 5K
        }
    }


    /// Calculate thermal pressure factor without caching
    fn calculate_thermal_pressure_factor_uncached(&self, temp_diff_abs: f64) -> f64 {
        // Apply thermal pressure based on temperature difference
        if temp_diff_abs > 1000.0 {
            // High temperature difference: strong thermal pressure
            1.0 + (temp_diff_abs - 1000.0) / 2000.0 // Up to 2.5x for 4000K difference
        } else {
            // Low temperature difference: minimal thermal pressure
            1.0 + temp_diff_abs / 10000.0 // Up to 1.1x for 1000K difference
        }
    }


    /// Calculate potential heat flow between layer tuples without applying the transfer
    /// Used for energy distribution calculations in multi-neighbor systems
    pub fn calculate_potential_heat_flow(
        &self,
        upper_layer: &ThermalLayer,
        lower_layer: &ThermalLayer,
    ) -> f64 {
        // Use default area and distance for simple calculations
        let default_area_m2 = upper_layer.surface_area_km2 * 1_000_000.0; // Convert km² to m²
        let default_distance_km = (upper_layer.height_km + lower_layer.height_km) / 2.0;
        
        // Use lateral heat flow calculation with default time step
        self.calculate_potential_heat_flow_simple(
            upper_layer,
            lower_layer,
            default_area_m2,
            default_distance_km,
            self.years, // Use the configured time step
        )
    }
    
    /// Calculate potential heat flow with custom geometry and conductivity
    /// note- because used for LATERAL and VERTICAL transfer the area andd distance must be passed in
    pub fn calculate_potential_heat_flow_simple(
        &self,
        source_layer: &ThermalLayer,
        target_layer: &ThermalLayer,
        transfer_area_m2: f64,
        transfer_distance_km: f64,
        time_step_years: f64,
    ) -> f64 {
        let source_temp = source_layer.temperature_k();
        let target_temp = target_layer.temperature_k();
        let temp_diff = (source_temp - target_temp).abs();
        
        // Only calculate if there's a meaningful temperature difference
        if temp_diff < MIN_TEMP_DIFF_K {
            return 0.0;
        }
        
        // Apply Fourier's law: Q = -k * A * (dT/dx)
        let temp_gradient = temp_diff / (transfer_distance_km * KM_TO_M);
        let conductivity = (source_layer.energy_mass.thermal_conductivity() + target_layer.energy_mass.thermal_conductivity()) / 2.0;
        let heat_flux_w = conductivity * transfer_area_m2 * temp_gradient;
        
        // Convert to energy transfer in joules
        let energy_transfer_j = heat_flux_w * time_step_years * SECONDS_PER_YEAR;
        
        // Apply transfer rate limiting for stability
        let source_energy = source_layer.energy_mass.energy();
        let target_energy = target_layer.energy_mass.energy();
        let max_energy_transfer = (source_energy - target_energy).abs() * MAX_ENERGY_TRANSFER_FRACTION;
        
        // Return limited energy transfer
        energy_transfer_j.abs().min(max_energy_transfer) * temp_diff.signum()
    }

}

#[cfg(test)]
mod tests {


}
