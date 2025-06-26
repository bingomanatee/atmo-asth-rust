// src/material.rs - Material system with thermal and physical properties

use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MaterialType {
    Silicate,   // General silicate/peridotite (mantle material)
    Basaltic,   // Basaltic rock (oceanic crust)
    Granitic,   // Granitic rock (continental crust) - lowest melting point
    Metallic,   // Metallic materials (iron/nickel)
    Icy,        // Ice/water
}

impl MaterialType {
    pub fn as_str(&self) -> &'static str {
        match self {
            MaterialType::Silicate => "silicate",
            MaterialType::Basaltic => "basaltic",
            MaterialType::Granitic => "granitic",
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
    /// R0 thermal transmission coefficient range for thermal diffusion equations
    /// Controls energy transfer efficiency between layers (tunable for equilibrium)
    /// Each EnergyMass instance gets a random value within this range
    pub thermal_transmission_r0_min: f64,
    pub thermal_transmission_r0_max: f64,
    /// Scientifically-based melting point information
    pub melting_point_min_k: f64,  // Minimum melting point (K)
    pub melting_point_max_k: f64,  // Maximum melting point (K)
    pub melting_point_avg_k: f64,  // Average melting point (K) - used as base
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

    /// Calculate thickness-dependent melting point
    /// Melting point increases with thickness due to pressure effects
    /// thickness_km: Total lithosphere thickness above this layer
    pub fn melting_point_at_thickness(&self, thickness_km: f64) -> f64 {
        // Pressure effect: ~50K increase per 100km thickness (rough geological estimate)
        let pressure_increase_k = (thickness_km / 100.0) * 50.0;

        // Base melting point + pressure effect, capped at maximum
        let adjusted_melting_point = self.melting_point_avg_k + pressure_increase_k;

        // Cap at reasonable maximum (don't exceed max + 500K)
        adjusted_melting_point.min(self.melting_point_max_k + 500.0)
    }

    /// Get melting point range information
    pub fn melting_point_range(&self) -> (f64, f64, f64) {
        (self.melting_point_min_k, self.melting_point_avg_k, self.melting_point_max_k)
    }
}

pub static MATERIAL_PROFILES: Lazy<HashMap<MaterialType, MaterialProfile>> = Lazy::new(|| {
    use MaterialType::*;
    let mut m = HashMap::new();

    m.insert(Silicate, MaterialProfile {
        kind: Silicate,
        max_lith_formation_temp_kv: 1750.0, // Average mantle ~1600-2000K, use mid-high range
        peak_lith_growth_temp_kv: 1550.0,   // Peak growth below melting point
        max_lith_growth_km_per_year: 0.001,
        max_lith_height_km: 300.0, // Allow up to 300km total (15 layers × 20km)
        density_kg_m3: 3300.0,
        specific_heat_capacity_j_per_kg_k: 1000.0,
        thermal_conductivity_w_m_k: 3.2,
        thermal_transmission_r0_min: 0.0000001, // 0.000001% minimum thermal transmission (ultra-realistic geological)
        thermal_transmission_r0_max: 0.0000002, // 0.000002% maximum thermal transmission (ultra-realistic geological)
        melting_point_min_k: 1700.0,       // Peridotite minimum ~1700K
        melting_point_max_k: 1900.0,       // Peridotite maximum ~1900K
        melting_point_avg_k: 1800.0,       // Peridotite average ~1800K
    });

    m.insert(Basaltic, MaterialProfile {
        kind: Basaltic,
        max_lith_formation_temp_kv: 1523.0, // Basalt ~1473-1573K, use mid-range
        peak_lith_growth_temp_kv: 1373.0,   // Peak growth below melting point
        max_lith_growth_km_per_year: 0.0012,
        max_lith_height_km: 20.0,
        density_kg_m3: 2900.0,
        specific_heat_capacity_j_per_kg_k: 840.0,
        thermal_conductivity_w_m_k: 2.1,
        thermal_transmission_r0_min: 0.00000008,
        thermal_transmission_r0_max: 0.00000016,
        melting_point_min_k: 1473.0,       // Basalt minimum ~1473K
        melting_point_max_k: 1573.0,       // Basalt maximum ~1573K
        melting_point_avg_k: 1523.0,       // Basalt average ~1523K
    });

    m.insert(Granitic, MaterialProfile {
        kind: Granitic,
        max_lith_formation_temp_kv: 1238.0, // Granite ~1215-1260K, use mid-range
        peak_lith_growth_temp_kv: 1100.0,   // Peak growth well below melting point
        max_lith_growth_km_per_year: 0.0008,
        max_lith_height_km: 50.0, // Continental crust can be thicker
        density_kg_m3: 2650.0,    // Granite density
        specific_heat_capacity_j_per_kg_k: 790.0, // Granite specific heat
        thermal_conductivity_w_m_k: 2.5,   // Granite thermal conductivity
        thermal_transmission_r0_min: 0.0000006,
        thermal_transmission_r0_max: 0.0000014,
        melting_point_min_k: 1215.0,       // Granite minimum ~1215K
        melting_point_max_k: 1260.0,       // Granite maximum ~1260K
        melting_point_avg_k: 1238.0,       // Granite average ~1238K
    });

    m.insert(Metallic, MaterialProfile {
        kind: Metallic,
        max_lith_formation_temp_kv: 1811.0, // Iron melting point
        peak_lith_growth_temp_kv: 1650.0,   // Peak growth below melting point
        max_lith_growth_km_per_year: 0.0008,
        max_lith_height_km: 70.0,
        density_kg_m3: 7800.0,
        specific_heat_capacity_j_per_kg_k: 450.0,
        thermal_conductivity_w_m_k: 80.0,
        thermal_transmission_r0_min: 0.0000002,
        thermal_transmission_r0_max: 0.0000003,
        melting_point_min_k: 1811.0,       // Iron melting point (pure iron)
        melting_point_max_k: 1811.0,       // Iron melting point (pure iron)
        melting_point_avg_k: 1811.0,       // Iron melting point (pure iron)
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
        thermal_transmission_r0_min: 0.00000005,
        thermal_transmission_r0_max: 0.00000012,
        melting_point_min_k: 273.15,       // Ice melting point (0°C)
        melting_point_max_k: 273.15,       // Ice melting point (0°C)
        melting_point_avg_k: 273.15,       // Ice melting point (0°C)
    });

    m
});

pub fn get_profile(kind: MaterialType) -> Option<&'static MaterialProfile> {
    MATERIAL_PROFILES.get(&kind)
}
