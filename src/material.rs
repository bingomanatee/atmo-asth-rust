// src/material.rs - Material system with thermal and physical properties

use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MaterialType {
    Silicate,
    Basaltic,
    Metallic,
    Icy,
}

impl MaterialType {
    pub fn as_str(&self) -> &'static str {
        match self {
            MaterialType::Silicate => "silicate",
            MaterialType::Basaltic => "basaltic",
            MaterialType::Metallic => "metallic",
            MaterialType::Icy => "icy",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "silicate" => Some(MaterialType::Silicate),
            "basaltic" => Some(MaterialType::Basaltic),
            "metallic" => Some(MaterialType::Metallic),
            "icy" => Some(MaterialType::Icy),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MaterialProfile {
    pub kind: MaterialType,
    pub max_lith_formation_temp_kv: f64,
    pub peak_lith_growth_temp_kv: f64,
    pub max_lith_growth_km_per_year: f64,
    pub max_lith_height_km: f64,
    pub density_kg_m3: f64,
    pub specific_heat_capacity_j_per_kg_k: f64,
    pub thermal_conductivity_w_m_k: f64,
}

impl MaterialProfile {
    pub fn growth_at_kelvin(&self, kelvin: f64) -> f64 {
        if kelvin >= self.max_lith_formation_temp_kv {
            0.0
        } else if kelvin <= self.peak_lith_growth_temp_kv {
            self.max_lith_growth_km_per_year
        } else {
            let temp_range = (self.max_lith_formation_temp_kv - self.peak_lith_growth_temp_kv).abs();
            let temp_diff = (kelvin - self.max_lith_formation_temp_kv).abs();
            self.max_lith_growth_km_per_year * (temp_diff/temp_range)
        }
    }
}

pub static MATERIAL_PROFILES: Lazy<HashMap<MaterialType, MaterialProfile>> = Lazy::new(|| {
    use MaterialType::*;
    let mut m = HashMap::new();

    m.insert(Silicate, MaterialProfile {
        kind: Silicate,
        max_lith_formation_temp_kv: 1873.15,
        peak_lith_growth_temp_kv: 1673.15,
        max_lith_growth_km_per_year: 0.001,
        max_lith_height_km: 100.0,
        density_kg_m3: 3300.0,
        specific_heat_capacity_j_per_kg_k: 1000.0,
        thermal_conductivity_w_m_k: 3.2,
    });

    m.insert(Basaltic, MaterialProfile {
        kind: Basaltic,
        max_lith_formation_temp_kv: 1773.15,
        peak_lith_growth_temp_kv: 1573.15,
        max_lith_growth_km_per_year: 0.0012,
        max_lith_height_km: 20.0,
        density_kg_m3: 2900.0,
        specific_heat_capacity_j_per_kg_k: 840.0,
        thermal_conductivity_w_m_k: 2.1,
    });

    m.insert(Metallic, MaterialProfile {
        kind: Metallic,
        max_lith_formation_temp_kv: 1973.15,
        peak_lith_growth_temp_kv: 1773.15,
        max_lith_growth_km_per_year: 0.0008,
        max_lith_height_km: 70.0,
        density_kg_m3: 7800.0,
        specific_heat_capacity_j_per_kg_k: 450.0,
        thermal_conductivity_w_m_k: 80.0,
    });

    m.insert(Icy, MaterialProfile {
        kind: Icy,
        max_lith_formation_temp_kv: 273.15,
        peak_lith_growth_temp_kv: 250.0,
        max_lith_growth_km_per_year: 0.002,
        max_lith_height_km: 150.0,
        density_kg_m3: 930.0,
        specific_heat_capacity_j_per_kg_k: 2100.0,
        thermal_conductivity_w_m_k: 2.2,
    });

    m
});

pub fn get_profile(kind: MaterialType) -> Option<&'static MaterialProfile> {
    MATERIAL_PROFILES.get(&kind)
}
