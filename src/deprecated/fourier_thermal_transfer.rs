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
    pub const CONVECTION_EXPONENTIAL_FACTOR: f64 = 2.5;



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
struct CachedDirectionalHeatFlow {
    energy_transfer_factor: f64,  // Pre-calculated factor for energy transfer
    conductivity_factor: f64,     // Pre-calculated conductivity factor
    thermal_pressure_factor: f64, // Pre-calculated thermal pressure factor
}


/// Global directional heat flow cache for performance optimization
static DIRECTIONAL_HEAT_FLOW_CACHE: Lazy<Mutex<HashMap<DirectionalHeatFlowCacheKey, CachedDirectionalHeatFlow>>> = 
    Lazy::new(|| Mutex::new(HashMap::new()));

/// Global density-adjusted conductivity cache for performance optimization
static DENSITY_CONDUCTIVITY_CACHE: Lazy<Mutex<HashMap<DensityAdjustedConductivityCacheKey, f64>>> = 
    Lazy::new(|| Mutex::new(HashMap::new()));

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
    fn calculate_exponential_thermal_pressure(
        &self,
        higher_temp: f64,
        lower_temp: f64,
    ) -> f64 {
        // Calculate thermal pressure for both temperatures
        let higher_thermal_pressure = self.calculate_thermal_pressure_coefficient_uncached(higher_temp, CONVECTION_THRESHOLD_K);
        let lower_thermal_pressure = self.calculate_thermal_pressure_coefficient_uncached(lower_temp, CONVECTION_THRESHOLD_K);
        
        // Use the ratio of thermal pressures as the enhancement factor
        // Higher temperature differences create stronger pressure gradients
        let pressure_ratio = if lower_thermal_pressure > 0.0 {
            higher_thermal_pressure / lower_thermal_pressure
        } else {
            higher_thermal_pressure.max(1.0) // Fallback to absolute pressure if lower is zero
        };
        
        // Apply reasonable bounds to prevent numerical instability
        pressure_ratio.clamp(0.1, 10.0)
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
        from_energy_mass: &dyn crate::energy_mass_composite::EnergyMassComposite,
        to_energy_mass: &dyn crate::energy_mass_composite::EnergyMassComposite,
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

    /// Get cached directional heat flow factors
    fn get_cached_directional_heat_flow(
        &self,
        from_energy_mass: &dyn crate::energy_mass_composite::EnergyMassComposite,
        to_energy_mass: &dyn crate::energy_mass_composite::EnergyMassComposite,
        temp_diff: f64,
    ) -> CachedDirectionalHeatFlow {
        let cache_key = Self::create_directional_heat_flow_cache_key(from_energy_mass, to_energy_mass, temp_diff);
        
        // Try to get from cache first
        if let Ok(cache) = DIRECTIONAL_HEAT_FLOW_CACHE.try_lock() {
            if let Some(cached) = cache.get(&cache_key) {
                return cached.clone();
            }
        }
        
        // Calculate if not in cache
        let cached_flow = self.calculate_directional_heat_flow_uncached(from_energy_mass, to_energy_mass, temp_diff);
        
        // Cache the result
        if let Ok(mut cache) = DIRECTIONAL_HEAT_FLOW_CACHE.try_lock() {
            cache.insert(cache_key, cached_flow.clone());
        }
        
        cached_flow
    }

    /// Calculate directional heat flow factors without caching
    fn calculate_directional_heat_flow_uncached(
        &self,
        from_energy_mass: &dyn crate::energy_mass_composite::EnergyMassComposite,
        to_energy_mass: &dyn crate::energy_mass_composite::EnergyMassComposite,
        temp_diff: f64,
    ) -> CachedDirectionalHeatFlow {
        // Use cached density-adjusted conductivity for performance
        let adjusted_from_conductivity = self.get_cached_density_adjusted_conductivity(from_energy_mass);
        let adjusted_to_conductivity = self.get_cached_density_adjusted_conductivity(to_energy_mass);

        // Calculate factors
        let base_transfer_rate = 0.08; // 8% per timestep (increased to improve thermal conduction from foundry)
        let avg_conductivity = (adjusted_from_conductivity + adjusted_to_conductivity) / 2.0;
        let conductivity_factor = avg_conductivity / 2.0;
        let thermal_pressure_factor = self.calculate_thermal_pressure_factor_uncached(temp_diff.abs());

        CachedDirectionalHeatFlow {
            energy_transfer_factor: base_transfer_rate,
            conductivity_factor,
            thermal_pressure_factor,
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

    /// Get cached density-adjusted conductivity for a material type, phase, and current density
    fn get_cached_density_adjusted_conductivity(
        &self,
        energy_mass: &dyn crate::energy_mass_composite::EnergyMassComposite,
    ) -> f64 {
        let cache_key = DensityAdjustedConductivityCacheKey {
            material_type: energy_mass.material_composite_type(),
            phase: energy_mass.phase(),
            current_density_rounded: (energy_mass.density_kgm3() * 10.0).round() as i32,  // Round to nearest 0.1
        };
        
        // Try to get from cache first
        if let Ok(cache) = DENSITY_CONDUCTIVITY_CACHE.try_lock() {
            if let Some(cached) = cache.get(&cache_key) {
                return *cached;
            }
        }
        
        // Calculate if not in cache
        let base_conductivity = energy_mass.thermal_conductivity();
        let current_density = energy_mass.density_kgm3();
        let default_density = energy_mass.material_composite_profile().density_kg_m3;
        
        let adjusted_conductivity = self.calculate_density_adjusted_conductivity_uncached(
            base_conductivity,
            current_density,
            default_density,
            None, // No thermal layer available in this context
        );
        
        // Cache the result
        if let Ok(mut cache) = DENSITY_CONDUCTIVITY_CACHE.try_lock() {
            cache.insert(cache_key, adjusted_conductivity);
        }
        
        adjusted_conductivity
    }


    /// Calculate directional heat flow between two EnergyMass objects (helper function)
    /// Returns energy transfer magnitude in Joules (always positive)
    fn calculate_directional_heat_flow(
        &self,
        from_energy_mass: &dyn crate::energy_mass_composite::EnergyMassComposite,
        to_energy_mass: &dyn crate::energy_mass_composite::EnergyMassComposite,
        temp_diff: f64,
    ) -> f64 {
        // Use cached directional heat flow factors for performance
        let cached_factors = self.get_cached_directional_heat_flow(from_energy_mass, to_energy_mass, temp_diff);

        // Calculate thermal energy difference
        let thermal_energy_diff = from_energy_mass.thermal_capacity() * temp_diff;
        let base_transfer = thermal_energy_diff * cached_factors.energy_transfer_factor;
        
        // Apply sanity cap: limit transfer to prevent instability while allowing foundry heat flow
        let max_transfer = from_energy_mass.energy() * 0.80; // 80% of total energy in source layer (increased for foundry)
        let energy_diff_cap = (from_energy_mass.energy() - to_energy_mass.energy()).abs() * 0.80; // 80% of energy difference
        let capped_transfer = base_transfer.min(max_transfer).min(energy_diff_cap);

        // Apply cached conductivity and thermal pressure factors
        let scaled_transfer = capped_transfer * cached_factors.conductivity_factor * cached_factors.thermal_pressure_factor;

        scaled_transfer
    }

    /// Calculate bidirectional heat flow between two EnergyMass objects with thickness-based diffusivity scaling
    /// Returns energy transfer in Joules (positive = energy flows from upper to lower)
    pub fn heat_flow_between_energy_masses_with_thickness(
        &self,
        upper_energy_mass: &dyn crate::energy_mass_composite::EnergyMassComposite,
        lower_energy_mass: &dyn crate::energy_mass_composite::EnergyMassComposite,
        avg_thickness_km: Option<f64>,
    ) -> f64 {
        let upper_temp_k = upper_energy_mass.kelvin();
        let lower_temp_k = lower_energy_mass.kelvin();

        // Calculate thickness scaling factor if thickness is provided
        let thickness_scaling = if let Some(thickness_km) = avg_thickness_km {
            // For layer-to-layer heat transfer, thermal resistance scales linearly with thickness
            // Use a simple linear relationship: thicker layers = proportionally slower transfer
            let baseline_thickness_km = 10.0; // Reference thickness (10km layers transfer at 1.0x rate)
            let thickness_factor = baseline_thickness_km / thickness_km;
            
            // Special handling for foundry layers (very thick, high-temperature layers)
            // If this is a very thick layer (>100km) with high temperature (>1600K), 
            // it's likely a foundry layer that needs enhanced heat transfer
            let avg_temp = (upper_temp_k + lower_temp_k) / 2.0;
            if thickness_km > 100.0 && avg_temp > 1600.0 {
                // Foundry layers get enhanced heat transfer despite thickness
                thickness_factor.clamp(0.8, 3.0) // Allow 80% transfer rate minimum for foundry
            } else {
                // Normal layers get standard clamping
                thickness_factor.clamp(0.1, 2.0)
            }
        } else {
            1.0 // No thickness scaling
        };

        if upper_temp_k > lower_temp_k {
            // Heat flows from upper to lower (positive)
            let temp_diff = upper_temp_k - lower_temp_k;
            let heat_flow = self.calculate_directional_heat_flow(upper_energy_mass, lower_energy_mass, temp_diff);
            heat_flow * thickness_scaling
        } else if lower_temp_k > upper_temp_k {
            // Heat flows from lower to upper (negative)
            let temp_diff = lower_temp_k - upper_temp_k;
            let heat_flow = self.calculate_directional_heat_flow(lower_energy_mass, upper_energy_mass, temp_diff);
            -(heat_flow * thickness_scaling)
        } else {
            // Temperatures equal, no heat flow
            0.0
        }
    }



    /// Apply heat transfer between adjacent thermal layer tuples with directional attenuation and thickness scaling
    /// Modifies the 'next' state of both layers based on calculated heat flow
    /// Returns the amount of energy transferred (positive = upper to lower)
    pub fn apply_heat_transfer_between_layers(
        &self,
        upper_layer_tuple: &mut (ThermalLayer, ThermalLayer),
        lower_layer_tuple: &mut (ThermalLayer, ThermalLayer),
    ) -> f64 {
        // Calculate average layer thickness for diffusivity scaling
        let avg_thickness_km = (upper_layer_tuple.0.height_km + lower_layer_tuple.0.height_km) / 2.0;

        // Calculate heat flow using current state with thickness scaling
        let heat_flow = self.heat_flow_between_energy_masses_with_thickness(
            &upper_layer_tuple.0.energy_mass,
            &lower_layer_tuple.0.energy_mass,
            Some(avg_thickness_km)
        );

        if heat_flow.abs() > 0.0 {
            // Apply energy transfer to next state with directional attenuation rules
            let upper_capacity = upper_layer_tuple.0.energy_mass.thermal_capacity();
            let lower_capacity = lower_layer_tuple.0.energy_mass.thermal_capacity();

            // Apply thermal pressure directly to the heat flow for layer-to-layer transfer
            let upper_temp = upper_layer_tuple.0.temperature_k();
            let lower_temp = lower_layer_tuple.0.temperature_k();

            // Exponential temperature-pressure dynamics for faster convection at higher temperatures
            let thermal_pressure_factor = self.calculate_exponential_thermal_pressure(
                upper_temp.max(lower_temp), upper_temp.min(lower_temp)
            );

            // Apply thermal pressure to the base heat flow
            let limited_transfer = heat_flow.abs() * thermal_pressure_factor;

            if heat_flow > 0.0 {
                // Heat flows from upper to lower (downward)
                upper_layer_tuple.1.remove_energy(limited_transfer);
                lower_layer_tuple.1.add_energy(limited_transfer);
                limited_transfer
            } else {
                // Heat flows from lower to upper (upward)
                lower_layer_tuple.1.remove_energy(limited_transfer);
                upper_layer_tuple.1.add_energy(limited_transfer);
                limited_transfer
            }
        } else {
            0.0
        }
    }

    /// Transfer heat between layer tuples (wrapper for backward compatibility)
    pub fn transfer_heat_between_layer_tuples(
        &self,
        upper_layer_tuple: &mut (ThermalLayer, ThermalLayer),
        lower_layer_tuple: &mut (ThermalLayer, ThermalLayer),
    ) -> f64 {
        self.apply_heat_transfer_between_layers(upper_layer_tuple, lower_layer_tuple)
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
    /// Used for lateral transfers between cells where distance and area need to be specified
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
    use crate::energy_mass_composite::{
        EnergyMassComposite, EnergyMassParams, MaterialCompositeType, MaterialPhase,
        StandardEnergyMassComposite,
    };
    use crate::global_thermal::ThermalLayer;
    use crate::fourier_thermal_transfer::FourierThermalTransfer;
    use std::f64::consts::PI;

    #[test]
    fn test_thermal_transfer() {
        const SURFACE_RAD: f64 = 120.0;
        const AREA: f64 = PI as f64 * SURFACE_RAD * SURFACE_RAD;
        let mut thermal_layer = ThermalLayer::new(
            0.0,  // start_depth_km
            10.0, // height_km
            AREA, // surface_area_km2
            MaterialCompositeType::Silicate,
        );

        thermal_layer.energy_mass.set_kelvin(1200.0);

        let mut thermal_layer_tuples = (thermal_layer.clone(), thermal_layer.clone());

        let mut thermal_layer2 = ThermalLayer::new(
            10.0, // start_depth_km
            10.0, // height_km
            AREA, // surface_area_km2
            MaterialCompositeType::Silicate,
        );

        thermal_layer2.energy_mass.set_kelvin(1000.0);

        let mut thermal_layer_tuples2 = (thermal_layer2.clone(), thermal_layer2.clone());

        let mut thermal_layer3 = ThermalLayer::new(
            10.0, // start_depth_km
            10.0, // height_km
            AREA, // surface_area_km2
            MaterialCompositeType::Silicate,
        );

        thermal_layer3.energy_mass.set_kelvin(2000.0);

        let mut thermal_layer_tuples3 = (thermal_layer3.clone(), thermal_layer3.clone());

        let fourier = FourierThermalTransfer::new(200.0);
        let transfer = fourier.transfer_heat_between_layer_tuples(
            &mut thermal_layer_tuples,
            &mut thermal_layer_tuples2,
        );
        println!(
            "Transfer: {:.2e} from {:.2e} ({:.1} K) to {:.2e} ({:.1} K): {:.8}% for a diff of {:.2e}",
            transfer,
            thermal_layer.energy_mass.energy(),
            thermal_layer.energy_mass.kelvin(),
            thermal_layer2.energy_mass.energy(),
            thermal_layer2.energy_mass.kelvin(),
            transfer / thermal_layer.energy_mass.energy() * 100.0,
            thermal_layer.energy_mass.energy() - thermal_layer2.energy_mass.energy(),
        );

        let transfer3 = fourier.transfer_heat_between_layer_tuples(
            &mut thermal_layer_tuples,
            &mut thermal_layer_tuples3,
        );
        println!(
            "Transfer: {:.2e} from {:.2e} ({:.1} K) to {:.2e} ({:.1} K): {:.8}% for a diff of {:.2e}",
            transfer3,
            thermal_layer.energy_mass.energy(),
            thermal_layer.energy_mass.kelvin(),
            thermal_layer3.energy_mass.energy(),
            thermal_layer3.energy_mass.kelvin(),
            transfer3 / thermal_layer3.energy_mass.energy() * 100.0,
            thermal_layer.energy_mass.energy() - thermal_layer2.energy_mass.energy(),
        );


        assert!(transfer > 0.0);
    }
}
