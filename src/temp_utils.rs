//! Utilities for converting between temperature and thermal energy
//! using Kelvin and standard geophysical constants.

use crate::constants::{
    EARTH_RADIUS_KM, GLOBAL_ENERGY_LOSS_TABLE, GLOBAL_HEAT_INPUT_ASTHENOSPHERE, KM3_TO_M3,
    MANTLE_DENSITY_KGM3, SPECIFIC_HEAT_CAPACITY_MANTLE_J_PER_KG_K, TO_KELVIN,
};
use crate::h3_utils::H3Utils;
use h3o::Resolution;

/// Converts Celsius to Kelvin.
pub fn celsius_to_kelvin(temp_c: f64) -> f64 {
    temp_c + TO_KELVIN
}

/// Converts Kelvin to Celsius.
pub fn kelvin_to_celsius(temp_k: f64) -> f64 {
    temp_k - TO_KELVIN
}

/// Computes energy (Joules) from temperature in Kelvin and volume in km³.
///
/// # Arguments
/// - `temp_k`: Temperature in Kelvin
/// - `volume_km3`: Volume of the layer in km³
/// - `specific_heat`: Specific heat capacity in J/(kg·K)
///
/// # Returns
/// Total thermal energy in Joules
pub fn energy_from_kelvin(temp_k: f64, volume_km3: f64, specific_heat: f64) -> f64 {
    let volume_m3 = volume_km3 * 1.0e9;
    let mass_kg = volume_m3 * MANTLE_DENSITY_KGM3;

    mass_kg * specific_heat * temp_k
}

/// Computes temperature in Kelvin from energy in Joules and volume in km³.
///
/// # Arguments
/// - `energy_j`: Total energy in Joules
/// - `volume_km3`: Volume of the layer in km³
///
/// # Returns
/// Mean temperature in Kelvin
pub fn joules_volume_to_kelvin(energy_j: f64, volume_km3: f64, specific_heat: f64) -> f64 {
    let volume_m3 = volume_km3 * KM3_TO_M3;
    let mass_kg = volume_m3 * MANTLE_DENSITY_KGM3;

    energy_j / (mass_kg * specific_heat)
}

/// Convenience function that uses the standard mantle specific heat
pub fn joules_volume_to_kelvin_mantle(energy_j: f64, volume_km3: f64) -> f64 {
    joules_volume_to_kelvin(energy_j, volume_km3, SPECIFIC_HEAT_CAPACITY_MANTLE_J_PER_KG_K)
}

/// Convenience function that uses the standard mantle specific heat
pub fn energy_from_kelvin_mantle(temp_k: f64, volume_km3: f64) -> f64 {
    energy_from_kelvin(temp_k, volume_km3, SPECIFIC_HEAT_CAPACITY_MANTLE_J_PER_KG_K)
}

/// Convenience function that uses the standard mantle specific heat
pub fn kelvin_volume_to_joules_mantle(temp_k: f64, volume_km3: f64) -> f64 {
    volume_kelvin_to_joules(volume_km3, temp_k, SPECIFIC_HEAT_CAPACITY_MANTLE_J_PER_KG_K)
}

pub fn volume_kelvin_to_joules(volume_km3: f64, temp_k: f64, specific_heat: f64) -> f64 {
    const KM3_TO_M3: f64 = 1.0e9;

    let volume_m3 = volume_km3 * KM3_TO_M3;
    let mass_kg = volume_m3 * MANTLE_DENSITY_KGM3;

    mass_kg * specific_heat * temp_k
}

fn cooling_j_my_for_thickness(thick_km: f64) -> f64 {
    // clamp outside table
    if thick_km <= GLOBAL_ENERGY_LOSS_TABLE[0].0 {
        return GLOBAL_ENERGY_LOSS_TABLE[0].1;
    }
    if thick_km >= GLOBAL_ENERGY_LOSS_TABLE[GLOBAL_ENERGY_LOSS_TABLE.len() - 1].0 {
        return GLOBAL_ENERGY_LOSS_TABLE[GLOBAL_ENERGY_LOSS_TABLE.len() - 1].1;
    }
    // find bounding segments
    for win in GLOBAL_ENERGY_LOSS_TABLE.windows(2) {
        let (t1, e1) = win[0];
        let (t2, e2) = win[1];
        if thick_km >= t1 && thick_km <= t2 {
            // linear in log-space for exponential decay shape
            let f = (thick_km - t1) / (t2 - t1);
            let log_e1 = e1.log10();
            let log_e2 = e2.log10();
            return 10_f64.powf(log_e1 + f * (log_e2 - log_e1));
        }
    }
    unreachable!()
}

/// J/Myr radiated per H3 cell at `res`, given planet radius km and lithosphere thickness km
pub fn cooling_per_cell_per_year(
    res: Resolution,
    planet_radius_km: f64,
    lithosphere_height_km: f64,
) -> f64 {
    // 1. global loss for this thickness on Earth
    let earth_loss = cooling_j_my_for_thickness(lithosphere_height_km);

    // 2. rescale for different planetary surface area ( ∝ R² )
    let scale = (planet_radius_km / EARTH_RADIUS_KM as f64).powf(2.0);
    let planetary_loss = earth_loss * scale;

    // 3. divide by number of cells at this resolution
    let cells: f64 = H3Utils::cell_count_at_resolution(res) as f64;
    planetary_loss / cells
}

pub fn radiance_per_cell_per_year(res: Resolution, planet_radius_km: f64, lithosphere_km: f64) -> f64 {
    let cells: f64 = H3Utils::cell_count_at_resolution(res) as f64;
    let planet_scale = (planet_radius_km / EARTH_RADIUS_KM as f64).powf(2.0);
    // exponential attenuation through the lithosphere
    const D0: f64 = 50.0; // km  (tunable)
    let attenuated = GLOBAL_HEAT_INPUT_ASTHENOSPHERE * (-lithosphere_km / D0).exp();
    
    attenuated * planet_scale / cells
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_celsius_kelvin_conversion() {
        let test_cases = vec![
            (0.0, 273.15),    // Freezing point of water
            (100.0, 373.15),  // Boiling point of water
            (1400.0, 1673.15), // Silicate peak growth temp
            (1600.0, 1873.15), // Silicate formation temp
        ];

        for (celsius, expected_kelvin) in test_cases {
            let kelvin = celsius_to_kelvin(celsius);
            let back_to_celsius = kelvin_to_celsius(kelvin);

            assert_abs_diff_eq!(kelvin, expected_kelvin, epsilon = 0.01);
            assert_abs_diff_eq!(back_to_celsius, celsius, epsilon = 0.01);
        }
    }

    #[test]
    fn test_energy_temperature_conversion_roundtrip() {
        let specific_heat = SPECIFIC_HEAT_CAPACITY_MANTLE_J_PER_KG_K;
        let test_volumes = vec![10.0, 50.0, 100.0, 200.0]; // km³
        let test_temperatures = vec![1000.0, 1400.0, 1673.15, 1873.15, 2000.0]; // K

        for volume in &test_volumes {
            for &temperature in &test_temperatures {
                // Convert temperature to energy
                let energy = energy_from_kelvin(temperature, *volume, specific_heat);

                // Convert energy back to temperature
                let recovered_temp = joules_volume_to_kelvin(energy, *volume, specific_heat);

                // Should be very close (within 0.01 K)
                assert_abs_diff_eq!(recovered_temp, temperature, epsilon = 0.01);

                println!("Volume: {:.1} km³, Temp: {:.2} K -> Energy: {:.2e} J -> Temp: {:.2} K",
                         volume, temperature, energy, recovered_temp);
            }
        }
    }

    #[test]
    fn test_convenience_functions_consistency() {
        let volume = 100.0; // km³
        let temperature = 1673.15; // K (Silicate peak growth temp)

        // Test that convenience functions give same results as explicit functions
        let energy1 = energy_from_kelvin(temperature, volume, SPECIFIC_HEAT_CAPACITY_MANTLE_J_PER_KG_K);
        let energy2 = energy_from_kelvin_mantle(temperature, volume);
        let energy3 = kelvin_volume_to_joules_mantle(temperature, volume);

        assert_abs_diff_eq!(energy1, energy2, epsilon = 1.0);
        assert_abs_diff_eq!(energy1, energy3, epsilon = 1.0);

        // Test reverse conversion
        let temp1 = joules_volume_to_kelvin(energy1, volume, SPECIFIC_HEAT_CAPACITY_MANTLE_J_PER_KG_K);
        let temp2 = joules_volume_to_kelvin_mantle(energy1, volume);

        assert_abs_diff_eq!(temp1, temperature, epsilon = 0.01);
        assert_abs_diff_eq!(temp2, temperature, epsilon = 0.01);
        assert_abs_diff_eq!(temp1, temp2, epsilon = 0.01);
    }

    #[test]
    fn test_energy_scales_with_volume() {
        let temperature = 1500.0; // K
        let specific_heat = SPECIFIC_HEAT_CAPACITY_MANTLE_J_PER_KG_K;

        let volume1 = 50.0;  // km³
        let volume2 = 100.0; // km³ (double)

        let energy1 = energy_from_kelvin(temperature, volume1, specific_heat);
        let energy2 = energy_from_kelvin(temperature, volume2, specific_heat);

        // Energy should scale linearly with volume
        assert_abs_diff_eq!(energy2, energy1 * 2.0, epsilon = 1.0);

        // Temperature should be consistent regardless of volume
        let temp1 = joules_volume_to_kelvin(energy1, volume1, specific_heat);
        let temp2 = joules_volume_to_kelvin(energy2, volume2, specific_heat);

        assert_abs_diff_eq!(temp1, temperature, epsilon = 0.01);
        assert_abs_diff_eq!(temp2, temperature, epsilon = 0.01);
    }

    #[test]
    fn test_energy_scales_with_temperature() {
        let volume = 100.0; // km³
        let specific_heat = SPECIFIC_HEAT_CAPACITY_MANTLE_J_PER_KG_K;

        let temp1 = 1000.0; // K
        let temp2 = 2000.0; // K (double)

        let energy1 = energy_from_kelvin(temp1, volume, specific_heat);
        let energy2 = energy_from_kelvin(temp2, volume, specific_heat);

        // Energy should scale linearly with temperature
        assert_abs_diff_eq!(energy2, energy1 * 2.0, epsilon = 1.0);
    }

    #[test]
    fn test_lithosphere_relevant_temperatures() {
        let volume = 100.0; // km³
        let specific_heat = SPECIFIC_HEAT_CAPACITY_MANTLE_J_PER_KG_K;

        // Test specific temperatures relevant to lithosphere formation
        let lithosphere_temps = vec![
            (1673.15, "Silicate peak growth temperature"),
            (1773.15, "Silicate halfway temperature"),
            (1873.15, "Silicate formation temperature"),
        ];

        for (temp, description) in lithosphere_temps {
            let energy = energy_from_kelvin_mantle(temp, volume);
            let recovered_temp = joules_volume_to_kelvin_mantle(energy, volume);

            assert_abs_diff_eq!(recovered_temp, temp, epsilon = 0.01);
            println!("{}: {:.2} K <-> {:.2e} J", description, temp, energy);
        }
    }
}
