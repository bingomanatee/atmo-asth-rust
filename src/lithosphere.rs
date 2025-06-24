// src/lithosphere.rs

use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LithosphereType {
    Silicate,
    Basaltic,
    Metallic,
    Icy,
}

impl LithosphereType {
    pub fn as_str(&self) -> &'static str {
        match self {
            LithosphereType::Silicate => "silicate",
            LithosphereType::Basaltic => "basaltic",
            LithosphereType::Metallic => "metallic",
            LithosphereType::Icy => "icy",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "silicate" => Some(LithosphereType::Silicate),
            "basaltic" => Some(LithosphereType::Basaltic),
            "metallic" => Some(LithosphereType::Metallic),
            "icy" => Some(LithosphereType::Icy),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LithosphereProfile {
    pub kind: LithosphereType,
    pub formation_temp_k: f64,             // No growth above this
    pub peak_growth_temp_k: f64,           // Max growth at or below this
    pub max_growth_km_per_year: f64,       // Maximum vertical growth rate
    pub thermal_conductivity_w_m_k: f64,   // For cooling dynamics
    pub density_kg_m3: f64,                // For mass/energy modeling
}

pub static LITHOSPHERE_PROFILES: Lazy<HashMap<LithosphereType, LithosphereProfile>> = Lazy::new(|| {
    use LithosphereType::*;
    let mut m = HashMap::new();

    m.insert(Silicate, LithosphereProfile {
        kind: Silicate,
        formation_temp_k: 1873.15,            // 1600 °C
        peak_growth_temp_k: 1673.15,          // 1400 °C
        max_growth_km_per_year: 0.001,        // 1 meter/year
        thermal_conductivity_w_m_k: 3.2,
        density_kg_m3: 3300.0,
    });

    m.insert(Basaltic, LithosphereProfile {
        kind: Basaltic,
        formation_temp_k: 1773.15,
        peak_growth_temp_k: 1573.15,
        max_growth_km_per_year: 0.0012,
        thermal_conductivity_w_m_k: 2.1,
        density_kg_m3: 2900.0,
    });

    m.insert(Metallic, LithosphereProfile {
        kind: Metallic,
        formation_temp_k: 1973.15,
        peak_growth_temp_k: 1773.15,
        max_growth_km_per_year: 0.0008,
        thermal_conductivity_w_m_k: 80.0,
        density_kg_m3: 7800.0,
    });

    m.insert(Icy, LithosphereProfile {
        kind: Icy,
        formation_temp_k: 273.15,
        peak_growth_temp_k: 250.0,
        max_growth_km_per_year: 0.002,
        thermal_conductivity_w_m_k: 2.2,
        density_kg_m3: 930.0,
    });

    m
});

pub fn get_profile(kind: LithosphereType) -> Option<&'static LithosphereProfile> {
    LITHOSPHERE_PROFILES.get(&kind)
}
