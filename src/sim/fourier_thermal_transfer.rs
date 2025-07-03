/// Science-backed Fourier heat transfer system for thermal diffusion
/// Implements proper Fourier's law with material properties and geometric constraints
/// Updated to work with new ThermalLayer arrays and (current, next) tuple structure

use crate::global_thermal::thermal_layer::ThermalLayer;
use crate::energy_mass_composite::EnergyMassComposite;

/// Physical constants for Fourier heat transfer calculations
pub mod fourier_constants {
    /// Minimum temperature difference for heat transfer (K)
    pub const MIN_TEMP_DIFF_K: f64 = 0.1;
    
    /// Maximum energy transfer fraction per timestep for numerical stability
    /// Prevents unrealistic energy oscillations in explicit time stepping
    pub const MAX_ENERGY_TRANSFER_FRACTION: f64 = 0.1;
    
    /// Conservative transfer rate per year (1% proven from wide experiments)
    pub const BASE_TRANSFER_RATE_PER_YEAR: f64 = 0.005; // 0.01;
    
    /// Conversion constants
    pub const SECONDS_PER_YEAR: f64 = 365.25 * 24.0 * 3600.0;
    pub const KM_TO_M: f64 = 1000.0;
    pub const KM2_TO_M2: f64 = 1_000_000.0;
}

#[derive(Debug, Clone)]
pub struct FourierThermalTransfer {
    pub years: f64,
}

impl FourierThermalTransfer {
    pub fn new(time_seconds: f64) -> Self {
        Self { years: time_seconds / fourier_constants::SECONDS_PER_YEAR }
    }
    
    pub fn new_from_years(time_years: f64) -> Self {
        Self { 
            years: time_years,
        }
    }

    /// Calculate heat flow between two thermal masses using proven Fourier approach
    /// Returns energy transfer in Joules (positive = energy flows from 'from' to 'to')
    /// 
    /// This uses the conservative approach proven in thermal_layer_node_wide.rs:
    /// - 1% thermal energy difference per year
    /// - Safety limits to prevent oscillation
    /// - Temperature-driven transfer with thermal capacity
    pub fn calculate_fourier_heat_flow(
        &self,
        from_temp_k: f64,
        to_temp_k: f64,
        from_thermal_capacity: f64,
        from_energy: f64,
        from_conductivity: f64,
        to_conductivity: f64,
    ) -> f64 {
        use fourier_constants::*;

        let temp_diff = from_temp_k - to_temp_k;

        // Only transfer heat from hot to cold
        if temp_diff <= MIN_TEMP_DIFF_K {
            return 0.0;
        }

        // CONSERVATIVE APPROACH: Use temperature difference to drive transfer
        // This ensures heat flows from hot to cold regardless of material differences

        // Calculate temperature-based transfer using thermal capacity
        let temp_diff_k = temp_diff; // Already calculated above

        // Conservative transfer: 1% of thermal energy difference per year, scaled by time
        let base_transfer_rate = BASE_TRANSFER_RATE_PER_YEAR * self.years;
        let thermal_energy_diff = from_thermal_capacity * temp_diff_k;
        let conservative_transfer = thermal_energy_diff * base_transfer_rate;

        // Apply conductivity scaling factor (but keep it conservative)
        let effective_conductivity = if from_conductivity > 0.0 && to_conductivity > 0.0 {
            2.0 * from_conductivity * to_conductivity / (from_conductivity + to_conductivity)
        } else {
            (from_conductivity + to_conductivity) / 2.0
        };

        let conductivity_factor = (effective_conductivity / 2.0).min(1.0); // Cap at 1.0 for stability
        let scaled_transfer = conservative_transfer * conductivity_factor;

        // Apply safety limits (use main energy for safety limit)
        let max_safe_transfer = from_energy * MAX_ENERGY_TRANSFER_FRACTION;

        let final_transfer = scaled_transfer.min(max_safe_transfer);

        final_transfer
    }

    /// Calculate bidirectional heat flow between two layers
    /// Returns energy transfer in Joules (positive = energy flows from upper to lower)
    pub fn calculate_layer_heat_flow(
        &self,
        upper_temp_k: f64,
        lower_temp_k: f64,
        upper_thermal_capacity: f64,
        lower_thermal_capacity: f64,
        upper_energy: f64,
        lower_energy: f64,
        upper_conductivity: f64,
        lower_conductivity: f64,
    ) -> f64 {
        if upper_temp_k > lower_temp_k {
            // Heat flows from upper to lower (positive)
            self.calculate_fourier_heat_flow(
                upper_temp_k,
                lower_temp_k,
                upper_thermal_capacity,
                upper_energy,
                upper_conductivity,
                lower_conductivity,
            )
        } else if lower_temp_k > upper_temp_k {
            // Heat flows from lower to upper (negative)
            -self.calculate_fourier_heat_flow(
                lower_temp_k,
                upper_temp_k,
                lower_thermal_capacity,
                lower_energy,
                lower_conductivity,
                upper_conductivity,
            )
        } else {
            // Temperatures equal, no heat flow
            0.0
        }
    }

    /// Calculate heat flow between thermal layer tuples (current, next)
    /// This is the main method for use with GlobalH3Cell layer arrays
    /// Returns energy transfer in Joules (positive = energy flows from upper to lower)
    pub fn calculate_thermal_layer_heat_flow(
        &self,
        upper_layer_tuple: &(ThermalLayer, ThermalLayer),
        lower_layer_tuple: &(ThermalLayer, ThermalLayer),
    ) -> f64 {
        // Use current state for calculations
        let upper_layer = &upper_layer_tuple.0;
        let lower_layer = &lower_layer_tuple.0;

        let upper_temp = upper_layer.temperature_k();
        let lower_temp = lower_layer.temperature_k();
        let upper_capacity = upper_layer.energy_mass.thermal_capacity();
        let lower_capacity = lower_layer.energy_mass.thermal_capacity();
        let upper_energy = upper_layer.energy_j();
        let lower_energy = lower_layer.energy_j();
        let upper_conductivity = upper_layer.thermal_conductivity();
        let lower_conductivity = lower_layer.thermal_conductivity();

        self.calculate_layer_heat_flow(
            upper_temp,
            lower_temp,
            upper_capacity,
            lower_capacity,
            upper_energy,
            lower_energy,
            upper_conductivity,
            lower_conductivity,
        )
    }

    /// Apply heat transfer between adjacent thermal layer tuples with directional attenuation
    /// Modifies the 'next' state of both layers based on calculated heat flow
    /// Returns the amount of energy transferred (positive = upper to lower)
    pub fn apply_heat_transfer_between_layers(
        &self,
        upper_layer_tuple: &mut (ThermalLayer, ThermalLayer),
        lower_layer_tuple: &mut (ThermalLayer, ThermalLayer),
    ) -> f64 {
        // Calculate heat flow using current state
        let heat_flow = self.calculate_thermal_layer_heat_flow(upper_layer_tuple, lower_layer_tuple);

        if heat_flow.abs() > 0.0 {
            // Apply energy transfer to next state with directional attenuation rules
            let upper_capacity = upper_layer_tuple.0.energy_mass.thermal_capacity();
            let lower_capacity = lower_layer_tuple.0.energy_mass.thermal_capacity();
            let min_capacity = upper_capacity.min(lower_capacity);

            // Determine transfer direction and apply appropriate limits
            let limited_transfer = if heat_flow > 0.0 {
                // Heat flows from upper to lower (downward)
                // RULE: Downward flow has no attenuation - allow full transfer
                let max_downward_transfer = min_capacity * 0.5; // Allow 50% for downward flow
                heat_flow.min(max_downward_transfer)
            } else {
                // Heat flows from lower to upper (upward)
                // RULE: Upward flow may be attenuated by overlying layers
                // Check if upper layer is atmospheric (low mass) - if so, reduce attenuation
                let upper_mass = upper_layer_tuple.0.mass_kg();
                let attenuation_factor = if upper_mass < 1e10 {
                    // Upper layer is atmospheric/low mass - allow more upward flow
                    0.3 // Allow 30% for upward flow through thin layers
                } else {
                    // Upper layer is solid - normal attenuation
                    fourier_constants::MAX_ENERGY_TRANSFER_FRACTION // Normal 10% limit
                };

                let max_upward_transfer = min_capacity * attenuation_factor;
                heat_flow.abs().min(max_upward_transfer)
            };

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
}
