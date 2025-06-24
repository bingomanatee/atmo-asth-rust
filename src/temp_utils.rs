//! Utilities for converting between temperature and thermal energy
//! using Kelvin and standard geophysical constants.

use crate::constants::{KM3_TO_M3, MANTLE_DENSITY_KGM3, SPECIFIC_HEAT_CAPACITY_MANTLE_J_PER_KG_K, TO_KELVIN};

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
///
/// # Returns
/// Total thermal energy in Joules
pub fn energy_from_kelvin(temp_k: f64, volume_km3: f64) -> f64 {
    let volume_m3 = volume_km3 * 1.0e9;
    let mass_kg = volume_m3 * MANTLE_DENSITY_KGM3;

    mass_kg * SPECIFIC_HEAT_CAPACITY_MANTLE_J_PER_KG_K * temp_k
}

/// Computes temperature in Kelvin from energy in Joules and volume in km³.
///
/// # Arguments
/// - `energy_j`: Total energy in Joules
/// - `volume_km3`: Volume of the layer in km³
///
/// # Returns
/// Mean temperature in Kelvin
pub fn joules_volume_to_kelvin(energy_j: f64, volume_km3: f64) -> f64 {
    let volume_m3 = volume_km3 * KM3_TO_M3;
    let mass_kg = volume_m3 * MANTLE_DENSITY_KGM3;

    energy_j / (mass_kg * SPECIFIC_HEAT_CAPACITY_MANTLE_J_PER_KG_K)
}

pub fn volume_kelvin_to_joules(volume_km3: f64, temp_k: f64) -> f64 {
    const KM3_TO_M3: f64 = 1.0e9;

    let volume_m3 = volume_km3 * KM3_TO_M3;
    let mass_kg = volume_m3 * MANTLE_DENSITY_KGM3;

    mass_kg * SPECIFIC_HEAT_CAPACITY_MANTLE_J_PER_KG_K * temp_k
}