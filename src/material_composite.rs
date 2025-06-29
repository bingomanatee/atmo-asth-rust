use once_cell::sync::Lazy;
use std::collections::HashMap;

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
#[derive(Clone, Copy, Debug)]
pub struct MaterialStateProfile {
    pub density_kg_m3: f64,
    pub specific_heat_capacity_j_per_kg_k: f64,
    pub thermal_conductivity_w_m_k: f64,

    pub thermal_transmission_r0_min: f64,
    pub thermal_transmission_r0_max: f64,
}

/// Material layers for each base MaterialType
pub static MATERIAL_COMPOSITES: Lazy<HashMap<MaterialCompositeType, MaterialComposite>> = Lazy::new(|| {
    let mut m = HashMap::new();

    // Silicate profiles
    let mut silicate_profiles: HashMap<MaterialPhase, MaterialStateProfile> = HashMap::new();
    silicate_profiles.insert(MaterialPhase::Solid, MaterialStateProfile {
        density_kg_m3: 3300.0,
        specific_heat_capacity_j_per_kg_k: 1200.0,
        thermal_conductivity_w_m_k: 3.2,
        thermal_transmission_r0_min: 0.0005,
        thermal_transmission_r0_max: 0.0012,
    });
    silicate_profiles.insert(MaterialPhase::Liquid, MaterialStateProfile {
        density_kg_m3: 3200.0,
        specific_heat_capacity_j_per_kg_k: 1300.0,
        thermal_conductivity_w_m_k: 2.5,
        thermal_transmission_r0_min: 0.0007,
        thermal_transmission_r0_max: 0.0015,
    });
    silicate_profiles.insert(MaterialPhase::Gas, MaterialStateProfile {
        density_kg_m3: 1.0,  // Very low density for gas
        specific_heat_capacity_j_per_kg_k: 1400.0,
        thermal_conductivity_w_m_k: 0.1,  // Very low conductivity for gas
        thermal_transmission_r0_min: 0.001,
        thermal_transmission_r0_max: 0.003,
    });

    m.insert(
        MaterialCompositeType::Silicate,
        MaterialComposite {
            kind: MaterialCompositeType::Silicate,
            melting_point_min_k: 1200.0,
            melting_point_max_k: 1600.0,
            melting_point_avg_k: 1400.0,
            max_lith_formation_temp_kv: 0.0,
            peak_lith_growth_temp_kv: 0.0,
            max_lith_growth_km_per_year: 0.0,
            max_lith_height_km: 0.0,
            profiles: silicate_profiles,
        },
    );

    // Basaltic profiles
    let mut basaltic_profiles = HashMap::new();
    basaltic_profiles.insert(MaterialPhase::Solid, MaterialStateProfile {
        density_kg_m3: 2900.0,
        specific_heat_capacity_j_per_kg_k: 1000.0,
        thermal_conductivity_w_m_k: 2.0,
        thermal_transmission_r0_min: 0.0004,
        thermal_transmission_r0_max: 0.0010,
    });
    basaltic_profiles.insert(MaterialPhase::Liquid, MaterialStateProfile {
        density_kg_m3: 2800.0,
        specific_heat_capacity_j_per_kg_k: 1050.0,
        thermal_conductivity_w_m_k: 1.6,
        thermal_transmission_r0_min: 0.0006,
        thermal_transmission_r0_max: 0.0013,
    });
    basaltic_profiles.insert(MaterialPhase::Gas, MaterialStateProfile {
        density_kg_m3: 0.8,  // Very low density for gas
        specific_heat_capacity_j_per_kg_k: 1100.0,
        thermal_conductivity_w_m_k: 0.08,  // Very low conductivity for gas
        thermal_transmission_r0_min: 0.0008,
        thermal_transmission_r0_max: 0.002,
    });

    m.insert(
        MaterialCompositeType::Basaltic,
        MaterialComposite {
            kind: MaterialCompositeType::Basaltic,
            melting_point_min_k: 1100.0,
            melting_point_max_k: 1350.0,
            melting_point_avg_k: 1225.0,
            max_lith_formation_temp_kv: 0.0,
            peak_lith_growth_temp_kv: 0.0,
            max_lith_growth_km_per_year: 0.0,
            max_lith_height_km: 0.0,
            profiles: basaltic_profiles,
        },
    );

    // Granitic profiles
    let mut granitic_profiles = HashMap::new();
    granitic_profiles.insert(MaterialPhase::Solid, MaterialStateProfile {
        density_kg_m3: 2700.0,
        specific_heat_capacity_j_per_kg_k: 800.0,
        thermal_conductivity_w_m_k: 2.5,
        thermal_transmission_r0_min: 0.0005,
        thermal_transmission_r0_max: 0.0011,
    });
    granitic_profiles.insert(MaterialPhase::Liquid, MaterialStateProfile {
        density_kg_m3: 2650.0,
        specific_heat_capacity_j_per_kg_k: 850.0,
        thermal_conductivity_w_m_k: 2.0,
        thermal_transmission_r0_min: 0.0007,
        thermal_transmission_r0_max: 0.0014,
    });
    granitic_profiles.insert(MaterialPhase::Gas, MaterialStateProfile {
        density_kg_m3: 0.6,  // Very low density for gas
        specific_heat_capacity_j_per_kg_k: 900.0,
        thermal_conductivity_w_m_k: 0.06,  // Very low conductivity for gas
        thermal_transmission_r0_min: 0.001,
        thermal_transmission_r0_max: 0.0025,
    });

    m.insert(
        MaterialCompositeType::Granitic,
        MaterialComposite {
            kind: MaterialCompositeType::Granitic,
            melting_point_min_k: 1000.0,
            melting_point_max_k: 1220.0,
            melting_point_avg_k: 1110.0,
            max_lith_formation_temp_kv: 0.0,
            peak_lith_growth_temp_kv: 0.0,
            max_lith_growth_km_per_year: 0.0,
            max_lith_height_km: 0.0,
            profiles: granitic_profiles,
        },
    );

    // Metallic profiles
    let mut metallic_profiles = HashMap::new();
    metallic_profiles.insert(MaterialPhase::Solid, MaterialStateProfile {
        density_kg_m3: 7800.0,
        specific_heat_capacity_j_per_kg_k: 500.0,
        thermal_conductivity_w_m_k: 50.0,
        thermal_transmission_r0_min: 0.0003,
        thermal_transmission_r0_max: 0.0008,
    });
    metallic_profiles.insert(MaterialPhase::Liquid, MaterialStateProfile {
        density_kg_m3: 7700.0,
        specific_heat_capacity_j_per_kg_k: 520.0,
        thermal_conductivity_w_m_k: 40.0,
        thermal_transmission_r0_min: 0.0005,
        thermal_transmission_r0_max: 0.0010,
    });
    metallic_profiles.insert(MaterialPhase::Gas, MaterialStateProfile {
        density_kg_m3: 2.0,  // Low density for metallic gas
        specific_heat_capacity_j_per_kg_k: 600.0,
        thermal_conductivity_w_m_k: 0.2,  // Low conductivity for gas
        thermal_transmission_r0_min: 0.0008,
        thermal_transmission_r0_max: 0.002,
    });

    m.insert(
        MaterialCompositeType::Metallic,
        MaterialComposite {
            kind: MaterialCompositeType::Metallic,
            melting_point_min_k: 1800.0,
            melting_point_max_k: 1900.0,
            melting_point_avg_k: 1850.0,
            max_lith_formation_temp_kv: 0.0,
            peak_lith_growth_temp_kv: 0.0,
            max_lith_growth_km_per_year: 0.0,
            max_lith_height_km: 0.0,
            profiles: metallic_profiles,
        },
    );

    // Icy profiles
    let mut icy_profiles = HashMap::new();
    icy_profiles.insert(MaterialPhase::Solid, MaterialStateProfile {
        density_kg_m3: 917.0,  // Ice density
        specific_heat_capacity_j_per_kg_k: 2100.0,
        thermal_conductivity_w_m_k: 2.2,
        thermal_transmission_r0_min: 0.0005,
        thermal_transmission_r0_max: 0.0012,
    });
    icy_profiles.insert(MaterialPhase::Liquid, MaterialStateProfile {
        density_kg_m3: 1000.0,  // Water density
        specific_heat_capacity_j_per_kg_k: 4186.0,  // Water specific heat
        thermal_conductivity_w_m_k: 0.6,  // Water thermal conductivity
        thermal_transmission_r0_min: 0.0008,
        thermal_transmission_r0_max: 0.0015,
    });
    icy_profiles.insert(MaterialPhase::Gas, MaterialStateProfile {
        density_kg_m3: 0.8,  // Water vapor density
        specific_heat_capacity_j_per_kg_k: 2010.0,  // Water vapor specific heat
        thermal_conductivity_w_m_k: 0.025,  // Water vapor thermal conductivity
        thermal_transmission_r0_min: 0.001,
        thermal_transmission_r0_max: 0.003,
    });

    m.insert(
        MaterialCompositeType::Icy,
        MaterialComposite {
            kind: MaterialCompositeType::Icy,
            melting_point_min_k: 273.15,
            melting_point_max_k: 273.15,
            melting_point_avg_k: 273.15,
            max_lith_formation_temp_kv: 0.0,
            peak_lith_growth_temp_kv: 0.0,
            max_lith_growth_km_per_year: 0.0,
            max_lith_height_km: 0.0,
            profiles: icy_profiles,
        },
    );

    // Air profiles (atmospheric gases)
    let mut air_profiles = HashMap::new();
    air_profiles.insert(MaterialPhase::Solid, MaterialStateProfile {
        density_kg_m3: 1.5,  // Solid air (very cold, compressed)
        specific_heat_capacity_j_per_kg_k: 1000.0,
        thermal_conductivity_w_m_k: 0.02,
        thermal_transmission_r0_min: 0.001,
        thermal_transmission_r0_max: 0.003,
    });
    air_profiles.insert(MaterialPhase::Liquid, MaterialStateProfile {
        density_kg_m3: 0.9,  // Liquid air (very cold)
        specific_heat_capacity_j_per_kg_k: 1200.0,
        thermal_conductivity_w_m_k: 0.15,
        thermal_transmission_r0_min: 0.0008,
        thermal_transmission_r0_max: 0.002,
    });
    air_profiles.insert(MaterialPhase::Gas, MaterialStateProfile {
        density_kg_m3: 1.225,  // Standard air density at sea level
        specific_heat_capacity_j_per_kg_k: 1005.0,  // Air specific heat
        thermal_conductivity_w_m_k: 0.026,  // Air thermal conductivity
        thermal_transmission_r0_min: 0.0005,
        thermal_transmission_r0_max: 0.0015,
    });

    m.insert(
        MaterialCompositeType::Air,
        MaterialComposite {
            kind: MaterialCompositeType::Air,
            melting_point_min_k: 54.36,   // Nitrogen freezing point (major component)
            melting_point_max_k: 90.20,   // Oxygen freezing point
            melting_point_avg_k: 72.28,   // Average
            max_lith_formation_temp_kv: 0.0,  // Air doesn't form lithosphere
            peak_lith_growth_temp_kv: 0.0,
            max_lith_growth_km_per_year: 0.0,
            max_lith_height_km: 0.0,
            profiles: air_profiles,
        },
    );

    m
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

/// ------------------------ get core ----------------------

pub fn get_material_core(material_type: &MaterialCompositeType) -> &MaterialComposite {
    match MATERIAL_COMPOSITES.get(material_type) {
        None => { panic!("cannot get core")}
        Some(composite) => { composite }
    }
}