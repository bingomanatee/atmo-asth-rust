use once_cell::sync::Lazy;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use serde_json;

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
}

#[derive(Deserialize)]
struct MaterialPhaseProperties {
    density_kg_m3: f64,
    specific_heat_capacity_j_per_kg_k: f64,
    thermal_conductivity_w_m_k: f64,
    thermal_transmission_r0_min: f64,
    thermal_transmission_r0_max: f64,
    melt_temp: f64,
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

/// MaterialComposite is the state of a material in
/// three phases - solid, liquid, and gas.
/// It keeps this information in profiles.
/// There are a few trans-state properties like melting point that are kept in the root Composite container.
///
#[derive(Clone, Debug)]
pub struct MaterialComposite {
    pub kind: MaterialCompositeType,
    // Melting and lithosphere formation stats (root)
    pub melting_point_min_k: f64,
    pub melting_point_max_k: f64,
    pub melting_point_avg_k: f64,
    pub max_lith_formation_temp_kv: f64,
    pub peak_lith_growth_temp_kv: f64,
    pub max_lith_growth_km_per_year: f64,
    pub max_lith_height_km: f64,

    /// Material profiles indexed by phase state
    pub profiles: HashMap<MaterialPhase, MaterialStateProfile>,
}

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
    pub latent_heat_fusion: f64,  // J/kg for solid->liquid transition
    pub boil_temp: f64,
    pub latent_heat_vapor: f64,   // J/kg for liquid->gas transition
}

/// Load materials from JSON file
fn load_materials_from_json() -> HashMap<MaterialCompositeType, MaterialComposite> {
    let json_str = include_str!("materials.json");
    let materials_json: MaterialsJson = serde_json::from_str(json_str)
        .expect("Failed to parse materials.json");

    let mut materials = HashMap::new();

    // Helper function to convert JSON phase data to our MaterialStateProfile
    let convert_phase = |props: &MaterialPhaseProperties| -> MaterialStateProfile {
        MaterialStateProfile {
            density_kg_m3: props.density_kg_m3,
            specific_heat_capacity_j_per_kg_k: props.specific_heat_capacity_j_per_kg_k,
            thermal_conductivity_w_m_k: props.thermal_conductivity_w_m_k,
            thermal_transmission_r0_min: props.thermal_transmission_r0_min,
            thermal_transmission_r0_max: props.thermal_transmission_r0_max,
            melt_temp: props.melt_temp,
            latent_heat_fusion: props.latent_heat_fusion,
            boil_temp: props.boil_temp,
            latent_heat_vapor: props.latent_heat_vapor,
        }
    };

    // Convert granite
    let mut granite_profiles = HashMap::new();
    granite_profiles.insert(MaterialPhase::Solid, convert_phase(&materials_json.granite.solid));
    granite_profiles.insert(MaterialPhase::Liquid, convert_phase(&materials_json.granite.liquid));
    granite_profiles.insert(MaterialPhase::Gas, convert_phase(&materials_json.granite.gas));

    materials.insert(MaterialCompositeType::Granitic, MaterialComposite {
        kind: MaterialCompositeType::Granitic,
        melting_point_min_k: materials_json.granite.solid.melt_temp,  // Solid->Liquid transition
        melting_point_max_k: materials_json.granite.solid.boil_temp,  // Liquid->Gas transition
        melting_point_avg_k: (materials_json.granite.solid.melt_temp + materials_json.granite.solid.boil_temp) / 2.0,
        max_lith_formation_temp_kv: 0.0,
        peak_lith_growth_temp_kv: 0.0,
        max_lith_growth_km_per_year: 0.0,
        max_lith_height_km: 0.0,
        profiles: granite_profiles,
    });

    // Convert water (Icy)
    let mut water_profiles = HashMap::new();
    water_profiles.insert(MaterialPhase::Solid, convert_phase(&materials_json.water.solid));
    water_profiles.insert(MaterialPhase::Liquid, convert_phase(&materials_json.water.liquid));
    water_profiles.insert(MaterialPhase::Gas, convert_phase(&materials_json.water.gas));

    materials.insert(MaterialCompositeType::Icy, MaterialComposite {
        kind: MaterialCompositeType::Icy,
        melting_point_min_k: materials_json.water.solid.melt_temp,  // Ice->Water transition (273.15K)
        melting_point_max_k: materials_json.water.solid.boil_temp,  // Water->Steam transition (373.15K)
        melting_point_avg_k: (materials_json.water.solid.melt_temp + materials_json.water.solid.boil_temp) / 2.0,
        max_lith_formation_temp_kv: 0.0,
        peak_lith_growth_temp_kv: 0.0,
        max_lith_growth_km_per_year: 0.0,
        max_lith_height_km: 0.0,
        profiles: water_profiles,
    });

    // Convert basalt (Basaltic)
    let mut basalt_profiles = HashMap::new();
    basalt_profiles.insert(MaterialPhase::Solid, convert_phase(&materials_json.basalt.solid));
    basalt_profiles.insert(MaterialPhase::Liquid, convert_phase(&materials_json.basalt.liquid));
    basalt_profiles.insert(MaterialPhase::Gas, convert_phase(&materials_json.basalt.gas));

    materials.insert(MaterialCompositeType::Basaltic, MaterialComposite {
        kind: MaterialCompositeType::Basaltic,
        melting_point_min_k: materials_json.basalt.solid.melt_temp,  // Solid->Liquid transition (1473K)
        melting_point_max_k: materials_json.basalt.solid.boil_temp,  // Liquid->Gas transition (2900K)
        melting_point_avg_k: (materials_json.basalt.solid.melt_temp + materials_json.basalt.solid.boil_temp) / 2.0,
        max_lith_formation_temp_kv: 0.0,
        peak_lith_growth_temp_kv: 0.0,
        max_lith_growth_km_per_year: 0.0,
        max_lith_height_km: 0.0,
        profiles: basalt_profiles,
    });

    // Convert steel (Metallic)
    let mut steel_profiles = HashMap::new();
    steel_profiles.insert(MaterialPhase::Solid, convert_phase(&materials_json.steel.solid));
    steel_profiles.insert(MaterialPhase::Liquid, convert_phase(&materials_json.steel.liquid));
    steel_profiles.insert(MaterialPhase::Gas, convert_phase(&materials_json.steel.gas));

    materials.insert(MaterialCompositeType::Metallic, MaterialComposite {
        kind: MaterialCompositeType::Metallic,
        melting_point_min_k: materials_json.steel.solid.melt_temp,  // Solid->Liquid transition (1808K)
        melting_point_max_k: materials_json.steel.solid.boil_temp,  // Liquid->Gas transition (3133K)
        melting_point_avg_k: (materials_json.steel.solid.melt_temp + materials_json.steel.solid.boil_temp) / 2.0,
        max_lith_formation_temp_kv: 0.0,
        peak_lith_growth_temp_kv: 0.0,
        max_lith_growth_km_per_year: 0.0,
        max_lith_height_km: 0.0,
        profiles: steel_profiles,
    });

    // Convert silicate (Silicate)
    let mut silicate_profiles = HashMap::new();
    silicate_profiles.insert(MaterialPhase::Solid, convert_phase(&materials_json.silicate.solid));
    silicate_profiles.insert(MaterialPhase::Liquid, convert_phase(&materials_json.silicate.liquid));
    silicate_profiles.insert(MaterialPhase::Gas, convert_phase(&materials_json.silicate.gas));

    materials.insert(MaterialCompositeType::Silicate, MaterialComposite {
        kind: MaterialCompositeType::Silicate,
        melting_point_min_k: materials_json.silicate.solid.melt_temp,  // Solid->Liquid transition (1600K)
        melting_point_max_k: materials_json.silicate.solid.boil_temp,  // Liquid->Gas transition (3200K)
        melting_point_avg_k: (materials_json.silicate.solid.melt_temp + materials_json.silicate.solid.boil_temp) / 2.0,
        max_lith_formation_temp_kv: 0.0,
        peak_lith_growth_temp_kv: 0.0,
        max_lith_growth_km_per_year: 0.0,
        max_lith_height_km: 0.0,
        profiles: silicate_profiles,
    });

    // Add remaining materials that aren't in JSON (Air) with default values

    // Air profiles (atmospheric gases)
    let mut air_profiles = HashMap::new();
    air_profiles.insert(MaterialPhase::Solid, MaterialStateProfile {
        density_kg_m3: 1.5,
        specific_heat_capacity_j_per_kg_k: 1000.0,
        thermal_conductivity_w_m_k: 0.02,
        thermal_transmission_r0_min: 0.001,
        thermal_transmission_r0_max: 0.003,
        melt_temp: 54.36,
        latent_heat_fusion: 25000.0,
        boil_temp: 90.20,
        latent_heat_vapor: 200000.0,
    });
    air_profiles.insert(MaterialPhase::Liquid, MaterialStateProfile {
        density_kg_m3: 0.9,
        specific_heat_capacity_j_per_kg_k: 1200.0,
        thermal_conductivity_w_m_k: 0.15,
        thermal_transmission_r0_min: 0.0008,
        thermal_transmission_r0_max: 0.002,
        melt_temp: 54.36,
        latent_heat_fusion: 25000.0,
        boil_temp: 90.20,
        latent_heat_vapor: 200000.0,
    });
    air_profiles.insert(MaterialPhase::Gas, MaterialStateProfile {
        density_kg_m3: 1.225,
        specific_heat_capacity_j_per_kg_k: 1005.0,
        thermal_conductivity_w_m_k: 0.026,
        thermal_transmission_r0_min: 0.0005,
        thermal_transmission_r0_max: 0.0015,
        melt_temp: 54.36,
        latent_heat_fusion: 25000.0,
        boil_temp: 90.20,
        latent_heat_vapor: 200000.0,
    });

    materials.insert(MaterialCompositeType::Air, MaterialComposite {
        kind: MaterialCompositeType::Air,
        melting_point_min_k: 54.36,
        melting_point_max_k: 90.20,
        melting_point_avg_k: 72.28,
        max_lith_formation_temp_kv: 0.0,
        peak_lith_growth_temp_kv: 0.0,
        max_lith_growth_km_per_year: 0.0,
        max_lith_height_km: 0.0,
        profiles: air_profiles,
    });

    materials
}

/// Material layers for each base MaterialType
pub static MATERIAL_COMPOSITES: Lazy<HashMap<MaterialCompositeType, MaterialComposite>> = Lazy::new(|| {
    load_materials_from_json()
});

/// ------------------------- fast lookup index for profile ---------------------
/// 
/// Flat array for fast O(1) profile lookups
/// Layout: [material_type_index * PHASE_COUNT + phase_index]
/// Total size: MaterialCompositeType::COUNT * MaterialPhase::COUNT entries
static PROFILE_LOOKUP_TABLE: Lazy<[MaterialStateProfile; MaterialCompositeType::COUNT * MaterialPhase::COUNT]> = Lazy::new(|| {
    let mut table = [MaterialStateProfile {
        density_kg_m3: 0.0,
        specific_heat_capacity_j_per_kg_k: 0.0,
        thermal_conductivity_w_m_k: 0.0,
        thermal_transmission_r0_min: 0.0,
        thermal_transmission_r0_max: 0.0,
        melt_temp: 0.0,
        latent_heat_fusion: 0.0,
        boil_temp: 0.0,
        latent_heat_vapor: 0.0,
    }; MaterialCompositeType::COUNT * MaterialPhase::COUNT];

    // Populate the flat array from the HashMap data
    for (material_type, layers) in MATERIAL_COMPOSITES.iter() {
        for (phase, profile) in &layers.profiles {
            let pi = phase.clone().as_index();
            let mi =  material_type.clone().as_index();
            let index = mi * MaterialPhase::COUNT + pi;
            table[index] = *profile;
        }
    }

    table
});


/// ------------------------ both of these methods take and return the same things 
/// 
/// Fast O(1) lookup for material profile by type and phase
/// Uses flat array indexing: material_index * PHASE_COUNT + phase_index
pub fn get_profile_fast(material_type: &MaterialCompositeType, phase: &MaterialPhase) -> &'static MaterialStateProfile {
    let index = material_type.as_index() * MaterialPhase::COUNT + phase.as_index();
    &PROFILE_LOOKUP_TABLE[index]
}

/// Get material profile for a specific material type and phase state (HashMap version - slower)
pub fn get_profile_for_state(material_type: MaterialCompositeType, phase: MaterialPhase) -> Option<&'static MaterialStateProfile> {
    MATERIAL_COMPOSITES.get(&material_type)
        .and_then(|layers| layers.profiles.get(&phase))
}

/// Determine the correct phase for a material at a given temperature
/// This ensures temperature and phase are always consistent
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

/// ------------------------ get core ----------------------

pub fn get_material_core(material_type: &MaterialCompositeType) -> &MaterialComposite {
    match MATERIAL_COMPOSITES.get(material_type) {
        None => { panic!("cannot get core")}
        Some(composite) => { composite }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_material_loading() {
        // Test that materials are loaded from JSON correctly
        let basalt = get_material_core(&MaterialCompositeType::Basaltic);
        println!("Basalt melting range: {}K - {}K (avg: {}K)",
                 basalt.melting_point_min_k, basalt.melting_point_max_k, basalt.melting_point_avg_k);

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

        // Verify the composite uses the correct transition range
        assert_eq!(basalt.melting_point_min_k, 1473.0);  // Solid->Liquid
        assert_eq!(basalt.melting_point_max_k, 2900.0);  // Liquid->Gas

        // Test water/ice
        let water = get_material_core(&MaterialCompositeType::Icy);
        let water_solid = get_profile_fast(&MaterialCompositeType::Icy, &MaterialPhase::Solid);
        println!("Water melting range: {}K - {}K (avg: {}K)",
                 water.melting_point_min_k, water.melting_point_max_k, water.melting_point_avg_k);
        println!("Ice latent heat fusion: {} J/kg", water_solid.latent_heat_fusion);
        println!("Water latent heat vapor: {} J/kg", water_solid.latent_heat_vapor);

        assert_eq!(water_solid.melt_temp, 273.15);
        assert_eq!(water_solid.boil_temp, 373.15);
        assert_eq!(water_solid.latent_heat_fusion, 334000.0);
        assert_eq!(water_solid.latent_heat_vapor, 2260000.0);

        // Verify the composite uses the correct transition range
        assert_eq!(water.melting_point_min_k, 273.15);  // Ice->Water
        assert_eq!(water.melting_point_max_k, 373.15);  // Water->Steam
    }
}