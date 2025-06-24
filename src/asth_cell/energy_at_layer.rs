use crate::constants::{ASTHENOSPHERE_SURFACE_START_TEMP_K, GEOTHERMAL_GRADIENT_K_PER_KM, MANTLE_DENSITY_KGM3, SPECIFIC_HEAT_CAPACITY_MANTLE_J_PER_KG_K};

const KM3_TO_M3: f64 = 1.0e9;
/// Compute energy (Joules) at layer `n`
/// given constant volume in km³ and constants from constants.rs
pub fn energy_at_layer(layer_index: usize, layer_height: f64, volume_km3: f64, surface_temp_k: f64) -> f64 {

    let depth_km = (layer_index as f64 + 0.5) * layer_height;
    let temp_k = surface_temp_k + GEOTHERMAL_GRADIENT_K_PER_KM * depth_km;
    let volume_m3 = volume_km3 * KM3_TO_M3;
    let mass_kg = volume_m3 * MANTLE_DENSITY_KGM3;
    
    mass_kg * SPECIFIC_HEAT_CAPACITY_MANTLE_J_PER_KG_K * temp_k
}
