use crate::constants::GEOTHERMAL_GRADIENT_K_PER_KM;
use crate::energy_mass::{EnergyMass, StandardEnergyMass};
use crate::material::MaterialType;

/// Compute energy (Joules) at layer `n` using material-specific properties
/// given constant volume in km³ and material type
pub fn energy_at_layer(layer_index: usize, layer_height: f64, volume_km3: f64, surface_temp_k: f64) -> f64 {
    energy_at_layer_with_material(MaterialType::Silicate, layer_index, layer_height, volume_km3, surface_temp_k)
}

/// Compute energy (Joules) at layer `n` using material-specific properties
/// given constant volume in km³ and specific material type
pub fn energy_at_layer_with_material(
    material_type: MaterialType,
    layer_index: usize,
    layer_height: f64,
    volume_km3: f64,
    surface_temp_k: f64
) -> f64 {
    let depth_km = (layer_index as f64 + 0.5) * layer_height;
    let temp_k = surface_temp_k + GEOTHERMAL_GRADIENT_K_PER_KM * depth_km;

    // Use EnergyMass trait to get material-specific energy calculation
    let energy_mass = StandardEnergyMass::new_with_material(material_type, temp_k, volume_km3);
    energy_mass.energy()
}
