use once_cell::sync::Lazy;
use serde::Deserialize;
use serde_json;
use crate::json_parser::JsonParser;

/// JSON structure for loading material data
#[derive(Deserialize)]
struct MaterialsJson {
    granite: MaterialPhaseData,
    water: MaterialPhaseData,
    basalt: MaterialPhaseData,
    steel: MaterialPhaseData,
    silicate: MaterialPhaseData,
}

#[derive(Deserialize)]
struct MaterialPhaseData {
    solid: MaterialPhaseProperties,
    liquid: MaterialPhaseProperties,
    gas: MaterialPhaseProperties,
    emission_compounds: Option<std::collections::HashMap<String, f64>>,
}

#[derive(Deserialize)]
struct MaterialPhaseProperties {
    density_kg_m3: f64,
    specific_heat_capacity_j_per_kg_k: f64,
    thermal_conductivity_w_m_k: f64,
    thermal_transmission_r0_min: f64,
    thermal_transmission_r0_max: f64,
    #[serde(default)]
    melt_temp: Option<f64>,
    #[serde(default)]
    melt_temp_min: Option<f64>,
    #[serde(default)]
    melt_temp_max: Option<f64>,
    latent_heat_fusion: f64,
    boil_temp: f64,
    latent_heat_vapor: f64,
}

/// Material phase state
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[repr(u8)]
pub enum MaterialPhase {
    Solid = 0,
    Liquid = 1,
    Gas = 2,
}

impl MaterialPhase {
    pub const COUNT: usize = 3;

    pub fn as_index(self) -> usize {
        self as usize
    }
}

#[derive(Hash, Eq, PartialEq, Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
#[repr(u8)]
pub enum MaterialCompositeType {
    Silicate = 0,   // General silicate/peridotite (mantle material)
    Basaltic = 1,   // Basaltic rock (oceanic crust)
    Granitic = 2,   // Granitic rock (continental crust) - lowest melting point
    Metallic = 3,   // Metallic materials (iron/nickel)
    Icy = 4,        // Ice/water
    Air = 5,        // Atmospheric gases
}

impl MaterialCompositeType {
    pub const COUNT: usize = 6;
    /// index is used in creating the flattened array used for optimizing retrieval from a flat array of a sate profile
    /// via get_profile_fast.
    pub fn as_index(self) -> usize {
        self as usize
    }
}

impl std::fmt::Display for MaterialCompositeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MaterialCompositeType::Silicate => write!(f, "Silicate"),
            MaterialCompositeType::Basaltic => write!(f, "Basaltic"),
            MaterialCompositeType::Granitic => write!(f, "Granitic"),
            MaterialCompositeType::Metallic => write!(f, "Metallic"),
            MaterialCompositeType::Icy => write!(f, "Icy"),
            MaterialCompositeType::Air => write!(f, "Air"),
        }
    }
}

// MaterialComposite struct removed - all functionality now uses direct profile lookups

/// this is the profile of a material in a specific phase - liquid, gas, solid
#[derive(Clone, Copy, Debug)]
pub struct MaterialStateProfile {
    pub density_kg_m3: f64,
    pub specific_heat_capacity_j_per_kg_k: f64,
    pub thermal_conductivity_w_m_k: f64,
    pub thermal_transmission_r0_min: f64,
    pub thermal_transmission_r0_max: f64,

    // Phase transition properties
    pub melt_temp: f64,
    pub melt_temp_min: Option<f64>,  // Minimum temperature for gradual melting
    pub melt_temp_max: Option<f64>,  // Maximum temperature for gradual melting
    pub latent_heat_fusion: f64,  // J/kg for solid->liquid transition
    pub boil_temp: f64,
    pub latent_heat_vapor: f64,   // J/kg for liquid->gas transition
}

/// Load material profiles directly from JSON into the lookup table
fn load_profiles_from_json() -> [MaterialStateProfile; MaterialCompositeType::COUNT * MaterialPhase::COUNT] {
    let json_str = include_str!("materials.json");
    
    // One-step: Load JSON directly
    let json_value = JsonParser::load_json_str("materials.json", json_str)
        .expect("Failed to parse materials.json");
    
    // Convert the cached JSON value back to a string for serde
    let materials_json: MaterialsJson = serde_json::from_str(&json_value.to_string())
        .expect("Failed to parse materials.json structure");

    let mut table = [MaterialStateProfile {
        density_kg_m3: 0.0,
        specific_heat_capacity_j_per_kg_k: 0.0,
        thermal_conductivity_w_m_k: 0.0,
        thermal_transmission_r0_min: 0.0,
        thermal_transmission_r0_max: 0.0,
        melt_temp: 0.0,
        melt_temp_min: None,
        melt_temp_max: None,
        latent_heat_fusion: 0.0,
        boil_temp: 0.0,
        latent_heat_vapor: 0.0,
    }; MaterialCompositeType::COUNT * MaterialPhase::COUNT];

    // Helper function to populate profiles from JSON data directly into table
    let populate_material = |table: &mut [MaterialStateProfile], material_type: MaterialCompositeType, data: &MaterialPhaseData| {
        let base_index = material_type.as_index() * MaterialPhase::COUNT;

        // Calculate melt_temp from min/max if not provided
        let solid_melt_temp = data.solid.melt_temp.unwrap_or_else(|| {
            match (data.solid.melt_temp_min, data.solid.melt_temp_max) {
                (Some(min), Some(max)) => (min + max) / 2.0,
                (Some(min), None) => min,
                (None, Some(max)) => max,
                (None, None) => 1600.0, // Default fallback
            }
        });

        // Solid phase
        table[base_index + MaterialPhase::Solid.as_index()] = MaterialStateProfile {
            density_kg_m3: data.solid.density_kg_m3,
            specific_heat_capacity_j_per_kg_k: data.solid.specific_heat_capacity_j_per_kg_k,
            thermal_conductivity_w_m_k: data.solid.thermal_conductivity_w_m_k,
            thermal_transmission_r0_min: data.solid.thermal_transmission_r0_min,
            thermal_transmission_r0_max: data.solid.thermal_transmission_r0_max,
            melt_temp: solid_melt_temp,
            melt_temp_min: data.solid.melt_temp_min,
            melt_temp_max: data.solid.melt_temp_max,
            latent_heat_fusion: data.solid.latent_heat_fusion,
            boil_temp: data.solid.boil_temp,
            latent_heat_vapor: data.solid.latent_heat_vapor,
        };

        // Calculate melt_temp from min/max if not provided
        let liquid_melt_temp = data.liquid.melt_temp.unwrap_or_else(|| {
            match (data.liquid.melt_temp_min, data.liquid.melt_temp_max) {
                (Some(min), Some(max)) => (min + max) / 2.0,
                (Some(min), None) => min,
                (None, Some(max)) => max,
                (None, None) => 1600.0, // Default fallback
            }
        });

        // Liquid phase
        table[base_index + MaterialPhase::Liquid.as_index()] = MaterialStateProfile {
            density_kg_m3: data.liquid.density_kg_m3,
            specific_heat_capacity_j_per_kg_k: data.liquid.specific_heat_capacity_j_per_kg_k,
            thermal_conductivity_w_m_k: data.liquid.thermal_conductivity_w_m_k,
            thermal_transmission_r0_min: data.liquid.thermal_transmission_r0_min,
            thermal_transmission_r0_max: data.liquid.thermal_transmission_r0_max,
            melt_temp: liquid_melt_temp,
            melt_temp_min: data.liquid.melt_temp_min,
            melt_temp_max: data.liquid.melt_temp_max,
            latent_heat_fusion: data.liquid.latent_heat_fusion,
            boil_temp: data.liquid.boil_temp,
            latent_heat_vapor: data.liquid.latent_heat_vapor,
        };

        // Calculate melt_temp from min/max if not provided
        let gas_melt_temp = data.gas.melt_temp.unwrap_or_else(|| {
            match (data.gas.melt_temp_min, data.gas.melt_temp_max) {
                (Some(min), Some(max)) => (min + max) / 2.0,
                (Some(min), None) => min,
                (None, Some(max)) => max,
                (None, None) => 1600.0, // Default fallback
            }
        });

        // Gas phase
        table[base_index + MaterialPhase::Gas.as_index()] = MaterialStateProfile {
            density_kg_m3: data.gas.density_kg_m3,
            specific_heat_capacity_j_per_kg_k: data.gas.specific_heat_capacity_j_per_kg_k,
            thermal_conductivity_w_m_k: data.gas.thermal_conductivity_w_m_k,
            thermal_transmission_r0_min: data.gas.thermal_transmission_r0_min,
            thermal_transmission_r0_max: data.gas.thermal_transmission_r0_max,
            melt_temp: gas_melt_temp,
            melt_temp_min: data.gas.melt_temp_min,
            melt_temp_max: data.gas.melt_temp_max,
            latent_heat_fusion: data.gas.latent_heat_fusion,
            boil_temp: data.gas.boil_temp,
            latent_heat_vapor: data.gas.latent_heat_vapor,
        };
    };

    // Populate all materials from JSON
    populate_material(&mut table, MaterialCompositeType::Silicate, &materials_json.silicate);
    populate_material(&mut table, MaterialCompositeType::Basaltic, &materials_json.basalt);
    populate_material(&mut table, MaterialCompositeType::Granitic, &materials_json.granite);
    populate_material(&mut table, MaterialCompositeType::Metallic, &materials_json.steel);
    populate_material(&mut table, MaterialCompositeType::Icy, &materials_json.water);

    // Manually populate Air profiles (not in JSON)
    let air_base_index = MaterialCompositeType::Air.as_index() * MaterialPhase::COUNT;

    // Air solid phase (frozen nitrogen at very low temps)
    table[air_base_index + MaterialPhase::Solid.as_index()] = MaterialStateProfile {
        density_kg_m3: 1026.5,  // Solid nitrogen density
        specific_heat_capacity_j_per_kg_k: 1040.0,  // Solid nitrogen specific heat
        thermal_conductivity_w_m_k: 0.234,  // Solid nitrogen thermal conductivity
        thermal_transmission_r0_min: 0.1,
        thermal_transmission_r0_max: 0.3,
        melt_temp: 63.15,  // Nitrogen melting point
        melt_temp_min: None,
        melt_temp_max: None,
        latent_heat_fusion: 25500.0,  // Nitrogen latent heat of fusion
        boil_temp: 77.36,  // Nitrogen boiling point
        latent_heat_vapor: 199000.0,  // Nitrogen latent heat of vaporization
    };

    // Air liquid phase (liquid nitrogen)
    table[air_base_index + MaterialPhase::Liquid.as_index()] = MaterialStateProfile {
        density_kg_m3: 808.5,  // Liquid nitrogen density
        specific_heat_capacity_j_per_kg_k: 2042.0,  // Liquid nitrogen specific heat
        thermal_conductivity_w_m_k: 0.1404,  // Liquid nitrogen thermal conductivity
        thermal_transmission_r0_min: 0.05,
        thermal_transmission_r0_max: 0.15,
        melt_temp: 63.15,  // Same transition temperatures
        melt_temp_min: None,
        melt_temp_max: None,
        latent_heat_fusion: 25500.0,
        boil_temp: 77.36,
        latent_heat_vapor: 199000.0,
    };

    // Air gas phase (normal atmospheric conditions)
    table[air_base_index + MaterialPhase::Gas.as_index()] = MaterialStateProfile {
        density_kg_m3: 1.225,  // Air density at sea level
        specific_heat_capacity_j_per_kg_k: 1005.0,  // Air specific heat at constant pressure
        thermal_conductivity_w_m_k: 0.0262,  // Air thermal conductivity
        thermal_transmission_r0_min: 0.01,
        thermal_transmission_r0_max: 0.05,
        melt_temp: 63.15,  // Same transition temperatures
        melt_temp_min: None,
        melt_temp_max: None,
        latent_heat_fusion: 25500.0,
        boil_temp: 77.36,
        latent_heat_vapor: 199000.0,
    };

    table
}

/// Material layers for each base MaterialType
// MaterialComposite HashMap removed - all functionality now uses direct profile lookups

/// ------------------------- fast lookup index for profile ---------------------
/// 
/// Flat array for fast O(1) profile lookups
/// Layout: [material_type_index * PHASE_COUNT + phase_index]
/// Total size: MaterialCompositeType::COUNT * MaterialPhase::COUNT entries
static PROFILE_LOOKUP_TABLE: Lazy<[MaterialStateProfile; MaterialCompositeType::COUNT * MaterialPhase::COUNT]> = Lazy::new(|| {
    load_profiles_from_json()
});


/// ------------------------ both of these methods take and return the same things 
/// 
/// Fast O(1) lookup for material profile by type and phase
/// Uses flat array indexing: material_index * PHASE_COUNT + phase_index
pub fn get_profile_fast(material_type: &MaterialCompositeType, phase: &MaterialPhase) -> &'static MaterialStateProfile {
    let index = material_type.as_index() * MaterialPhase::COUNT + phase.as_index();
    &PROFILE_LOOKUP_TABLE[index]
}

// Old HashMap-based lookup removed - use get_profile_fast() instead

/// Determine the correct phase for a material at a given temperature and depth
/// This ensures temperature and phase are always consistent, accounting for pressure effects
pub fn resolve_phase_from_temperature(material_type: &MaterialCompositeType, temp_k: f64) -> MaterialPhase {
    // Get the solid phase profile to access transition temperatures
    let solid_profile = get_profile_fast(material_type, &MaterialPhase::Solid);

    if temp_k < solid_profile.melt_temp {
        MaterialPhase::Solid
    } else if temp_k < solid_profile.boil_temp {
        MaterialPhase::Liquid
    } else {
        MaterialPhase::Gas
    }
}

/// Determine the correct phase for a material at a given temperature and depth
/// Accounts for pressure effects on phase transitions (more realistic for deep Earth)
pub fn resolve_phase_from_temperature_and_depth(material_type: &MaterialCompositeType, temp_k: f64, depth_km: f64) -> MaterialPhase {
    // Calculate pressure from depth using average Earth gradient
    let pressure_gpa = depth_km * 0.033; // ~0.033 GPa per km depth
    resolve_phase_from_temperature_and_pressure(material_type, temp_k, pressure_gpa)
}

/// Determine the correct phase for a material at a given temperature and pressure
/// Uses actual calculated pressure from cumulative mass above (most realistic)
pub fn resolve_phase_from_temperature_and_pressure(material_type: &MaterialCompositeType, temp_k: f64, pressure_gpa: f64) -> MaterialPhase {
    // Get the solid phase profile to access base transition temperatures
    let solid_profile = get_profile_fast(material_type, &MaterialPhase::Solid);

    // Pressure effect on melting point: Clausius-Clapeyron relation
    // Enhanced for deep Earth conditions - much stronger pressure effects
    let pressure_melt_increase = match material_type {
        MaterialCompositeType::Silicate => pressure_gpa * 50.0,  // 50 K/GPa (enhanced)
        MaterialCompositeType::Basaltic => pressure_gpa * 45.0,  // 45 K/GPa
        MaterialCompositeType::Granitic => pressure_gpa * 40.0,  // 40 K/GPa
        MaterialCompositeType::Metallic => pressure_gpa * 60.0,  // 60 K/GPa (iron core)
        _ => pressure_gpa * 45.0,  // Default
    };

    // Boiling point increases much more dramatically with pressure
    let pressure_boil_increase = pressure_melt_increase * 2.5; // Much stronger boiling point increase

    // Adjusted transition temperatures
    let effective_melt_temp = solid_profile.melt_temp + pressure_melt_increase;
    let effective_boil_temp = solid_profile.boil_temp + pressure_boil_increase;

    if temp_k < effective_melt_temp {
        MaterialPhase::Solid
    } else if temp_k < effective_boil_temp {
        MaterialPhase::Liquid
    } else {
        MaterialPhase::Gas
    }
}

/// Helper functions to get melting/boiling points directly from profiles
pub fn get_melting_point_k(material_type: &MaterialCompositeType) -> f64 {
    let solid_profile = get_profile_fast(material_type, &MaterialPhase::Solid);
    solid_profile.melt_temp
}

pub fn get_boiling_point_k(material_type: &MaterialCompositeType) -> f64 {
    let solid_profile = get_profile_fast(material_type, &MaterialPhase::Solid);
    solid_profile.boil_temp
}

/// Get emission compound ratios for a material type when it melts/outgasses
pub fn get_emission_compound_ratios(material_type: &MaterialCompositeType) -> std::collections::HashMap<String, f64> {
    // For now, return default emission ratios based on material type
    // TODO: Load from materials.json when emission_compounds are properly integrated
    let mut ratios = std::collections::HashMap::new();

    match material_type {
        MaterialCompositeType::Basaltic => {
            ratios.insert("CO2".to_string(), 0.45);
            ratios.insert("H2O".to_string(), 0.25);
            ratios.insert("SO2".to_string(), 0.15);
            ratios.insert("N2".to_string(), 0.08);
            ratios.insert("H2S".to_string(), 0.04);
            ratios.insert("CO".to_string(), 0.02);
            ratios.insert("H2".to_string(), 0.01);
        },
        MaterialCompositeType::Granitic => {
            ratios.insert("H2O".to_string(), 0.50);
            ratios.insert("CO2".to_string(), 0.30);
            ratios.insert("SO2".to_string(), 0.08);
            ratios.insert("N2".to_string(), 0.06);
            ratios.insert("H2S".to_string(), 0.03);
            ratios.insert("CO".to_string(), 0.02);
            ratios.insert("H2".to_string(), 0.01);
        },
        MaterialCompositeType::Silicate => {
            ratios.insert("CO2".to_string(), 0.40);
            ratios.insert("H2O".to_string(), 0.30);
            ratios.insert("SO2".to_string(), 0.12);
            ratios.insert("N2".to_string(), 0.10);
            ratios.insert("H2S".to_string(), 0.05);
            ratios.insert("CO".to_string(), 0.02);
            ratios.insert("H2".to_string(), 0.01);
        },
        MaterialCompositeType::Metallic => {
            ratios.insert("H2".to_string(), 0.60);
            ratios.insert("CO".to_string(), 0.20);
            ratios.insert("H2O".to_string(), 0.10);
            ratios.insert("CO2".to_string(), 0.05);
            ratios.insert("N2".to_string(), 0.03);
            ratios.insert("SO2".to_string(), 0.01);
            ratios.insert("H2S".to_string(), 0.01);
        },
        MaterialCompositeType::Icy => {
            ratios.insert("H2O".to_string(), 1.00);
        },
        _ => {
            // Default for Air and unknown materials
            ratios.insert("CO2".to_string(), 0.40);
            ratios.insert("H2O".to_string(), 0.30);
            ratios.insert("N2".to_string(), 0.30);
        }
    }

    ratios
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_material_loading() {
        // Test that materials are loaded from JSON correctly using direct profile access
        let basalt_melt = get_melting_point_k(&MaterialCompositeType::Basaltic);
        let basalt_boil = get_boiling_point_k(&MaterialCompositeType::Basaltic);
        println!("Basalt melting range: {}K - {}K (avg: {}K)",
                 basalt_melt, basalt_boil, (basalt_melt + basalt_boil) / 2.0);

        let basalt_solid = get_profile_fast(&MaterialCompositeType::Basaltic, &MaterialPhase::Solid);
        println!("Basalt solid density: {} kg/m³", basalt_solid.density_kg_m3);
        println!("Basalt solid melt temp: {}K", basalt_solid.melt_temp);
        println!("Basalt solid boil temp: {}K", basalt_solid.boil_temp);
        println!("Basalt solid latent heat fusion: {} J/kg", basalt_solid.latent_heat_fusion);
        println!("Basalt solid latent heat vapor: {} J/kg", basalt_solid.latent_heat_vapor);

        // Verify the values match our JSON
        assert_eq!(basalt_solid.melt_temp, 1473.0);
        assert_eq!(basalt_solid.boil_temp, 2900.0);
        assert_eq!(basalt_solid.latent_heat_fusion, 400000.0);
        assert_eq!(basalt_solid.latent_heat_vapor, 2000000.0);
        assert_eq!(basalt_solid.density_kg_m3, 3000.0);

        // Verify the helper functions work correctly
        assert_eq!(basalt_melt, 1473.0);  // Solid->Liquid
        assert_eq!(basalt_boil, 2900.0);  // Liquid->Gas

        // Test water/ice
        let water_melt = get_melting_point_k(&MaterialCompositeType::Icy);
        let water_boil = get_boiling_point_k(&MaterialCompositeType::Icy);
        let water_solid = get_profile_fast(&MaterialCompositeType::Icy, &MaterialPhase::Solid);
        println!("Water melting range: {}K - {}K (avg: {}K)",
                 water_melt, water_boil, (water_melt + water_boil) / 2.0);
        println!("Ice latent heat fusion: {} J/kg", water_solid.latent_heat_fusion);
        println!("Water latent heat vapor: {} J/kg", water_solid.latent_heat_vapor);

        assert_eq!(water_solid.melt_temp, 273.15);
        assert_eq!(water_solid.boil_temp, 373.15);
        assert_eq!(water_solid.latent_heat_fusion, 334000.0);
        assert_eq!(water_solid.latent_heat_vapor, 2260000.0);

        // Verify the helper functions work correctly
        assert_eq!(water_melt, 273.15);  // Ice->Water
        assert_eq!(water_boil, 373.15);  // Water->Steam
    }
}
