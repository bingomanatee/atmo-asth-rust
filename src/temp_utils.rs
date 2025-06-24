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
