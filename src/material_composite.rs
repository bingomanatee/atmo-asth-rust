use once_cell::sync::Lazy;
use std::collections::HashMap;
use crate::material_composite::MaterialType::Basaltic;

#[derive(Hash, Eq, PartialEq)]
pub enum MaterialType {
    Silicate,   // General silicate/peridotite (mantle material)
    Basaltic,   // Basaltic rock (oceanic crust)
    Granitic,   // Granitic rock (continental crust) - lowest melting point
    Metallic,   // Metallic materials (iron/nickel)
    Icy,        // Ice/water
}
/// Layer container holding common melting and formation stats plus lithosphere and asthenosphere profiles
pub struct Layers {
    pub kind: MaterialType,
    // Melting and lithosphere formation stats (root)
    pub melting_point_min_k: f64,
    pub melting_point_max_k: f64,
    pub melting_point_avg_k: f64,
    pub max_lith_formation_temp_kv: f64,
    pub peak_lith_growth_temp_kv: f64,
    pub max_lith_growth_km_per_year: f64,
    pub max_lith_height_km: f64,

    /// Lithosphere material profile
    pub lith: MaterialProfile,
    /// Asthenosphere material profile
    pub asth: MaterialProfile,
}
pub struct MaterialProfile {
    pub density_kg_m3: f64,
    pub specific_heat_capacity_j_per_kg_k: f64,
    pub thermal_conductivity_w_m_k: f64,

    pub thermal_transmission_r0_min: f64,
    pub thermal_transmission_r0_max: f64,
}

/// Material layers for each base MaterialType
pub static MATERIAL_LAYERS: Lazy<HashMap<MaterialType, Layers>> = Lazy::new(|| {
    let mut m = HashMap::new();

    // Silicate profiles
    m.insert(
        MaterialType::Silicate,
        Layers {
            kind: MaterialType::Silicate,
            melting_point_min_k: 1200.0,
            melting_point_max_k: 1600.0,
            melting_point_avg_k: 1400.0,
            max_lith_formation_temp_kv: 0.0,
            peak_lith_growth_temp_kv: 0.0,
            max_lith_growth_km_per_year: 0.0,
            max_lith_height_km: 0.0,
            lith: MaterialProfile {
                density_kg_m3: 3300.0,
                specific_heat_capacity_j_per_kg_k: 1200.0,
                thermal_conductivity_w_m_k: 3.2,
                thermal_transmission_r0_min: 0.0005,
                thermal_transmission_r0_max: 0.0012,
     
            },
            asth: MaterialProfile {
                density_kg_m3: 3200.0,
                specific_heat_capacity_j_per_kg_k: 1300.0,
                thermal_conductivity_w_m_k: 2.5,
                thermal_transmission_r0_min: 0.0007,
                thermal_transmission_r0_max: 0.0015,
           
            },
        },
    );

    // Basaltic profiles
    m.insert(
        MaterialType::Basaltic,
        Layers {
            kind: Basaltic,
            melting_point_min_k: 1100.0,
            melting_point_max_k: 1350.0,
            melting_point_avg_k: 1225.0,
            max_lith_formation_temp_kv: 0.0,
            peak_lith_growth_temp_kv: 0.0,
            max_lith_growth_km_per_year: 0.0,
            max_lith_height_km: 0.0,
            lith: MaterialProfile {
                density_kg_m3: 2900.0,
                specific_heat_capacity_j_per_kg_k: 1000.0,
                thermal_conductivity_w_m_k: 2.0,
                thermal_transmission_r0_min: 0.0004,
                thermal_transmission_r0_max: 0.0010,
              
            },
            asth: MaterialProfile {
                density_kg_m3: 2800.0,
                specific_heat_capacity_j_per_kg_k: 1050.0,
                thermal_conductivity_w_m_k: 1.6,
                thermal_transmission_r0_min: 0.0006,
                thermal_transmission_r0_max: 0.0013,
            },
        },
    );

    // Granitic profiles
    m.insert(
        MaterialType::Granitic,
        Layers {
            kind: MaterialType::Granitic,
            melting_point_min_k: 1000.0,
            melting_point_max_k: 1220.0,
            melting_point_avg_k: 1110.0,
            max_lith_formation_temp_kv: 0.0,
            peak_lith_growth_temp_kv: 0.0,
            max_lith_growth_km_per_year: 0.0,
            max_lith_height_km: 0.0,
            lith: MaterialProfile {
                density_kg_m3: 2700.0,
                specific_heat_capacity_j_per_kg_k: 800.0,
                thermal_conductivity_w_m_k: 2.5,
                thermal_transmission_r0_min: 0.0005,
                thermal_transmission_r0_max: 0.0011,
       
            },
            asth: MaterialProfile {
                density_kg_m3: 2650.0,
                specific_heat_capacity_j_per_kg_k: 850.0,
                thermal_conductivity_w_m_k: 2.0,
                thermal_transmission_r0_min: 0.0007,
                thermal_transmission_r0_max: 0.0014,
            },
        },
    );

    // Metallic profiles
    m.insert(
        MaterialType::Metallic,
        Layers {
            kind: MaterialType::Metallic,
            melting_point_min_k: 1800.0,
            melting_point_max_k: 1900.0,
            melting_point_avg_k: 1850.0,
            max_lith_formation_temp_kv: 0.0,
            peak_lith_growth_temp_kv: 0.0,
            max_lith_growth_km_per_year: 0.0,
            max_lith_height_km: 0.0,
            lith: MaterialProfile {
                density_kg_m3: 7800.0,
                specific_heat_capacity_j_per_kg_k: 500.0,
                thermal_conductivity_w_m_k: 50.0,
                thermal_transmission_r0_min: 0.0003,
                thermal_transmission_r0_max: 0.0008,
            },
            asth: MaterialProfile {
                density_kg_m3: 7700.0,
                specific_heat_capacity_j_per_kg_k: 520.0,
                thermal_conductivity_w_m_k: 40.0,
                thermal_transmission_r0_min: 0.0005,
                thermal_transmission_r0_max: 0.0010,
            },
        },
    );

    // Icy profiles (identical)
    m.insert(
        MaterialType::Icy,
        Layers {
            kind: MaterialType::Icy,
            melting_point_min_k: 273.15,
            melting_point_max_k: 273.15,
            melting_point_avg_k: 273.15,
            max_lith_formation_temp_kv: 0.0,
            peak_lith_growth_temp_kv: 0.0,
            max_lith_growth_km_per_year: 0.0,
            max_lith_height_km: 0.0,
            lith: MaterialProfile {
                density_kg_m3: 1000.0,
                specific_heat_capacity_j_per_kg_k: 2100.0,
                thermal_conductivity_w_m_k: 2.2,
                thermal_transmission_r0_min: 0.0005,
                thermal_transmission_r0_max: 0.0012,
      
            },
            asth: MaterialProfile {
                density_kg_m3: 1000.0,
                specific_heat_capacity_j_per_kg_k: 2100.0,
                thermal_conductivity_w_m_k: 2.2,
                thermal_transmission_r0_min: 0.0005,
                thermal_transmission_r0_max: 0.0012,
            },
        },
    );

    m
});
