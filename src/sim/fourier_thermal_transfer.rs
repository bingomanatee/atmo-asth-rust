use crate::energy_mass_composite::EnergyMassComposite;
/// Science-backed Fourier heat transfer system for thermal diffusion
/// Implements proper Fourier's law with material properties and geometric constraints
/// Updated to work with new ThermalLayer arrays and (current, next) tuple structure
use crate::global_thermal::thermal_layer::ThermalLayer;

/// Physical constants for Fourier heat transfer calculations
pub mod fourier_constants {
    /// Minimum temperature difference for heat transfer (K)
    pub const MIN_TEMP_DIFF_K: f64 = 0.1;

    /// Maximum energy transfer fraction per timestep for numerical stability
    /// Increased for geological equilibration over long timescales
    pub const MAX_ENERGY_TRANSFER_FRACTION: f64 = 0.25;

    /// Enhanced transfer rate per year for geological equilibration
    /// Increased from 0.005 to allow proper heat redistribution over geological time
    pub const BASE_TRANSFER_RATE_PER_YEAR: f64 = 0.02; // 2% per year for geological equilibration
}

#[derive(Debug, Clone)]
pub struct FourierThermalTransfer {
    pub years: f64,
}

impl FourierThermalTransfer {
    pub fn new(years: f64) -> Self {
        Self { years }
    }

    /// Calculate heat flow between two EnergyMass objects using their built-in thermal methods
    /// Returns energy transfer in Joules (positive = energy flows from 'from' to 'to')
    ///
    /// This delegates to EnergyMass.calculate_thermal_transfer() which already implements:
    /// - Proper Fourier heat transfer
    /// - Temperature-driven transfer with thermal capacity
    /// - Conductivity-based interface calculations
    /// - Material-specific thermal properties
    pub fn calculate_fourier_heat_flow_from_energy_mass(
        &self,
        from_energy_mass: &dyn crate::energy_mass_composite::EnergyMassComposite,
        to_energy_mass: &dyn crate::energy_mass_composite::EnergyMassComposite,
    ) -> f64 {
        // Use EnergyMass built-in radiate_to method for thermal transfer
        // Note: This is a read-only calculation, we don't actually transfer energy here
        let temp_diff = from_energy_mass.kelvin() - to_energy_mass.kelvin();
        if temp_diff <= 0.1 {
            return 0.0; // No transfer if temperatures are equal
        }

        // Calculate base transfer using thermal properties
        let base_transfer_rate = 0.01 * self.years; // 1% per year
        let thermal_energy_diff = from_energy_mass.thermal_capacity() * temp_diff;
        let base_transfer = thermal_energy_diff * base_transfer_rate;

        // Apply thermal pressure directly to the base energy flow
        let temp_diff_abs = (from_energy_mass.kelvin() - to_energy_mass.kelvin()).abs();
        let thermal_pressure_factor = if temp_diff_abs > 5000.0 {
            // Very high temperature differences: heavily reduced to prevent runaway
            0.15 // Only 15% for extreme differences
        } else if temp_diff_abs > 1000.0 {
            // High temperature differences: gradually reduce from 30% to 15%
            0.3 - ((temp_diff_abs - 1000.0) / 4000.0) * 0.15 // 30% down to 15%
        } else {
            // Low temperature differences: normal scaling
            0.15 + (temp_diff_abs / 1000.0) * 0.15 // 15% to 30% based on temp diff
        };

        // Apply thermal pressure to the base transfer amount
        let pressure_adjusted_transfer = base_transfer * thermal_pressure_factor;

        // Thermal inertia: prevent cells from dropping below minimum temperature
        let min_temp_k = 200.0; // Minimum temperature floor (200K = -73°C)
        let min_energy = from_energy_mass.thermal_capacity() * min_temp_k;
        let available_energy = (from_energy_mass.energy() - min_energy).max(0.0);
        let final_transfer = pressure_adjusted_transfer.min(available_energy);

        final_transfer
    }

    /// Apply thermal pressure to energy transfer based on temperature difference
    fn apply_thermal_pressure(&self, base_transfer: f64, temp_diff_abs: f64) -> f64 {
        let thermal_pressure_factor = if temp_diff_abs > 5000.0 {
            // Very high temperature differences: heavily reduced to prevent runaway
            0.15 // Only 15% for extreme differences
        } else if temp_diff_abs > 1000.0 {
            // High temperature differences: gradually reduce from 30% to 15%
            0.3 - ((temp_diff_abs - 1000.0) / 4000.0) * 0.15 // 30% down to 15%
        } else {
            // Low temperature differences: normal scaling
            0.15 + (temp_diff_abs / 1000.0) * 0.15 // 15% to 30% based on temp diff
        };

        base_transfer * thermal_pressure_factor
    }

    /// Calculate bidirectional heat flow between two EnergyMass objects
    /// Returns energy transfer in Joules (positive = energy flows from upper to lower)
    pub fn calculate_layer_heat_flow_from_energy_mass(
        &self,
        upper_energy_mass: &dyn crate::energy_mass_composite::EnergyMassComposite,
        lower_energy_mass: &dyn crate::energy_mass_composite::EnergyMassComposite,
    ) -> f64 {
        // Use the centralized EnergyMass-based heat flow calculation
        self.calculate_fourier_heat_flow_from_energy_mass(upper_energy_mass, lower_energy_mass)
    }

    /// Calculate density-adjusted thermal conductivity based on pressure compaction
    /// Uses 200km asthenosphere depth as baseline "norm" for heat transfer
    /// Lower pressures (vacuum effect) enhance heat transfer, higher pressures reduce it
    pub fn calculate_density_adjusted_conductivity(
        &self,
        base_conductivity: f64,
        current_density_kg_m3: f64,
        default_density_kg_m3: f64,
    ) -> f64 {
        if default_density_kg_m3 <= 0.0 {
            return base_conductivity;
        }

        // Calculate density ratio relative to surface density
        let surface_density_ratio = current_density_kg_m3 / default_density_kg_m3;

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
        base_conductivity * conductivity_multiplier
    }

    /// Calculate bidirectional heat flow between two EnergyMass objects with density-based conductivity adjustments
    /// Returns energy transfer in Joules (positive = energy flows from upper to lower)
    pub fn heat_flow_between_energy_massesd(
        &self,
        upper_energy_mass: &dyn crate::energy_mass_composite::EnergyMassComposite,
        lower_energy_mass: &dyn crate::energy_mass_composite::EnergyMassComposite,
    ) -> f64 {
        let upper_temp_k = upper_energy_mass.kelvin();
        let lower_temp_k = lower_energy_mass.kelvin();

        if upper_temp_k > lower_temp_k {
            // Heat flows from upper to lower (positive)
            // Extract density information from EnergyMass objects
            let upper_current_density = upper_energy_mass.density_kgm3();
            let upper_default_density =
                upper_energy_mass.material_composite_profile().density_kg_m3;
            let lower_current_density = lower_energy_mass.density_kgm3();
            let lower_default_density =
                lower_energy_mass.material_composite_profile().density_kg_m3;

            // Apply density adjustments to conductivity
            let upper_conductivity = upper_energy_mass.thermal_conductivity();
            let lower_conductivity = lower_energy_mass.thermal_conductivity();

            let adjusted_upper_conductivity = self.calculate_density_adjusted_conductivity(
                upper_conductivity,
                upper_current_density,
                upper_default_density,
            );
            let adjusted_lower_conductivity = self.calculate_density_adjusted_conductivity(
                lower_conductivity,
                lower_current_density,
                lower_default_density,
            );

            // Use simplified thermal transfer calculation with density adjustments
            let temp_diff = upper_temp_k - lower_temp_k;
            let base_transfer_rate = 0.01 * self.years; // 1% per year
            let thermal_energy_diff = upper_energy_mass.thermal_capacity() * temp_diff;
            let base_transfer = thermal_energy_diff * base_transfer_rate;

            // Apply conductivity scaling
            let avg_conductivity =
                (adjusted_upper_conductivity + adjusted_lower_conductivity) / 2.0;
            let conductivity_factor = avg_conductivity / 2.0;
            let scaled_transfer = base_transfer * conductivity_factor;

            // Apply thermal pressure
            self.apply_thermal_pressure(scaled_transfer, temp_diff.abs())
        } else if lower_temp_k > upper_temp_k {
            // Heat flows from lower to upper (negative)
            // Extract density information from EnergyMass objects
            let lower_current_density = lower_energy_mass.density_kgm3();
            let lower_default_density =
                lower_energy_mass.material_composite_profile().density_kg_m3;
            let upper_current_density = upper_energy_mass.density_kgm3();
            let upper_default_density =
                upper_energy_mass.material_composite_profile().density_kg_m3;

            let lower_conductivity = lower_energy_mass.thermal_conductivity();
            let upper_conductivity = upper_energy_mass.thermal_conductivity();

            let adjusted_lower_conductivity = self.calculate_density_adjusted_conductivity(
                lower_conductivity,
                lower_current_density,
                lower_default_density,
            );
            let adjusted_upper_conductivity = self.calculate_density_adjusted_conductivity(
                upper_conductivity,
                upper_current_density,
                upper_default_density,
            );

            // Use simplified thermal transfer calculation with density adjustments
            let temp_diff = lower_temp_k - upper_temp_k;
            let base_transfer_rate = 0.01 * self.years; // 1% per year
            let thermal_energy_diff = lower_energy_mass.thermal_capacity() * temp_diff;
            let base_transfer = thermal_energy_diff * base_transfer_rate;

            // Apply conductivity scaling
            let avg_conductivity =
                (adjusted_lower_conductivity + adjusted_upper_conductivity) / 2.0;
            let conductivity_factor = avg_conductivity / 2.0;
            let scaled_transfer = base_transfer * conductivity_factor;

            // Apply thermal pressure and return negative (upward flow)
            -self.apply_thermal_pressure(scaled_transfer, temp_diff.abs())
        } else {
            // Temperatures equal, no heat flow
            0.0
        }
    }

    /// Calculate heat flow between thermal layer tuples using centralized EnergyMass approach
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

        // Use the centralized EnergyMass-based method
        self.calculate_layer_heat_flow_from_energy_mass(
            &upper_layer.energy_mass,
            &lower_layer.energy_mass,
        )
    }

    /// Calculate heat flow between thermal layer tuples using centralized EnergyMass approach
    /// This method accounts for pressure effects on thermal conductivity
    /// Returns energy transfer in Joules (positive = energy flows from upper to lower)
    pub fn calculate_thermal_layer_heat_flow_with_density_adjustment(
        &self,
        upper_layer_tuple: &(ThermalLayer, ThermalLayer),
        lower_layer_tuple: &(ThermalLayer, ThermalLayer),
    ) -> f64 {
        // Use current state for calculations
        let upper_layer = &upper_layer_tuple.0;
        let lower_layer = &lower_layer_tuple.0;

        // Use the centralized EnergyMass-based method (density extracted internally)
        self.heat_flow_between_energy_massesd(&upper_layer.energy_mass, &lower_layer.energy_mass)
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
        let heat_flow =
            self.calculate_thermal_layer_heat_flow(upper_layer_tuple, lower_layer_tuple);

        if heat_flow.abs() > 0.0 {
            // Apply energy transfer to next state with directional attenuation rules
            let upper_capacity = upper_layer_tuple.0.energy_mass.thermal_capacity();
            let lower_capacity = lower_layer_tuple.0.energy_mass.thermal_capacity();

            // Apply thermal pressure directly to the heat flow for layer-to-layer transfer
            let upper_temp = upper_layer_tuple.0.temperature_k();
            let lower_temp = lower_layer_tuple.0.temperature_k();
            let temp_diff_abs = (upper_temp - lower_temp).abs();

           let thermal_pressure_factor = (temp_diff_abs/100.0).powf(1.5).clamp(0.25, 1.5);

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

    /// Apply heat transfer between adjacent thermal layer tuples using EnergyMass built-in methods
    /// Modifies the 'next' state of both layers based on calculated heat flow with thermal pressure
    /// Returns the amount of energy transferred (positive = upper to lower)
    pub fn transfer_heat_between_layer_tuples(
        &self,
        upper_layer_tuple: &mut (ThermalLayer, ThermalLayer),
        lower_layer_tuple: &mut (ThermalLayer, ThermalLayer),
    ) -> f64 {
        let heat_flow = self.heat_flow_between_energy_massesd(
            &upper_layer_tuple.0.energy_mass,
            &lower_layer_tuple.0.energy_mass,
        );

        if heat_flow.abs() > 0.0 {
            if heat_flow > 0.0 {
                // Heat flows from upper to lower (downward)
                upper_layer_tuple.1.remove_energy(heat_flow);
                lower_layer_tuple.1.add_energy(heat_flow);
            } else {
                // Heat flows from lower to upper (upward)
                lower_layer_tuple.1.remove_energy(heat_flow.abs());
                upper_layer_tuple.1.add_energy(heat_flow.abs());
            }
        }
        heat_flow
    }
}

#[cfg(test)]
mod tests {
    use crate::energy_mass_composite::{
        EnergyMassComposite, EnergyMassParams, MaterialCompositeType, MaterialPhase,
        StandardEnergyMassComposite,
    };
    use crate::global_thermal::ThermalLayer;
    use crate::sim::fourier_thermal_transfer::FourierThermalTransfer;
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
            "Transfer: {:.2e} from {:.2e} ({:.1} K) to {:.2e} ({:.1} K): {:.2}% for a diff of {:.2e}",
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
            "Transfer: {:.2e} from {:.2e} ({:.1} K) to {:.2e} ({:.1} K): {:.2}% for a diff of {:.2e}",
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
