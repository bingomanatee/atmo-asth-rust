pub const EARTH_RADIUS_KM: i32 = 6372;
pub const RHO_EARTH: f64 = 4.5; // g/cm³

// Physical constants for energy calculations (resolution-independent)
pub const MANTLE_DENSITY_KGM3: f64 = 3300.0;
pub const SPECIFIC_HEAT_CAPACITY_MANTLE_J_PER_KG_K: f64 = 1000.0; // Specific heat of mantle rock
pub const HEIGHT_PER_ASTH_LAYER: f64 = 10.0; // km per layer
pub const DENSITY_TO_TEMP_CONSTANT: f64 = 1e9;
pub const KM3_TO_M3: f64 = 1.0e9;
pub const TO_KELVIN: f64 = 273.15;
pub const DENSITY_TO_TEMP_CONVERSION: f64 =
    MANTLE_DENSITY_KGM3 * DENSITY_TO_TEMP_CONSTANT / SPECIFIC_HEAT_CAPACITY_MANTLE_J_PER_KG_K;
pub const GEOTHERMAL_GRADIENT_K_PER_KM: f64 = 0.5; // convection-dominated interior

// default sim start settings:
pub const ASTHENOSPHERE_SURFACE_START_TEMP_C: f64 = 1673.15; // Surface asthenosphere layer starting temp
pub const ASTHENOSPHERE_EQUILIBRIUM_TEMP_C: f64 = 1450.0; // Equilibrium asthenosphere temperature (above lithosphere formation)
pub const ASTHENOSPHERE_SURFACE_START_TEMP_K: f64 = ASTHENOSPHERE_SURFACE_START_TEMP_C + TO_KELVIN;
pub const ASTHENOSPHERE_EQUILIBRIUM_TEMP_K: f64 = ASTHENOSPHERE_EQUILIBRIUM_TEMP_C + TO_KELVIN;
pub const DEFAULT_LAYER_HEIGHT_KM: f64 = 10.0;


// Temperature cutoffs
pub const LITHOSPHERE_FORMATION_TEMP_K: f64 = 1973.15; // 1300 °C
pub const LITHOSPHERE_PEAK_FORMATION_TEMP_K: f64 = 1773.15;
pub const LITHOSPHERE_PEAK_FORMATION_HEIGHT: f64 = 5.0; // km height of generation

pub const LITHOSPHERE_MAX_VOLCANIC_LOSS_HEIGHT: f64 = 100.0;
pub const LITHOSPHERE_MAX_FRICTION_LOSS_HEIGHT: f64 = 5.0;
pub const MAX_DEGRADATION_KM_PER_MYR: f64 = 20.0;
pub const DEGRADATION_PER_K_ABOVE_PEAK: f64 = 0.1;

// === Global Energy Loss Based on Lithosphere Thickness (J per MIO years) ===
// Science-based cooling rates that depend on lithosphere insulation effect
pub const GLOBAL_ENERGY_LOSS_NO_LITHOSPHERE: f64 = 1.07e33; // J per MIO years - maximum heat loss
pub const GLOBAL_ENERGY_LOSS_1KM_LITHOSPHERE: f64 = 6.03e31; // J per MIO years - 1km insulation  
pub const GLOBAL_ENERGY_LOSS_5KM_LITHOSPHERE: f64 = 2.7e31; // J per MIO years - 10km insulation
pub const GLOBAL_ENERGY_LOSS_10KM_LITHOSPHERE: f64 = 6.03e30; // J per MIO years - 10km insulation
pub const GLOBAL_ENERGY_LOSS_100KM_LITHOSPHERE: f64 = 6.03e29; // J per MIO years - 100km insulation
pub const GLOBAL_ENERGY_LOSS_200KM_LITHOSPHERE: f64 = 0.0; // full insulation

/// --- the above points of reference are boiled down to the lookup table below
pub const GLOBAL_ENERGY_LOSS_TABLE: &[(f64, f64)] = &[
    (0.0, 1.07e33),   // No lithosphere
    (1.0, 6.03e31),   // Thin lithosphere
    (5.0, 2.7e31),    // Modest insulation
    (10.0, 6.03e30),  // Thicker crust
    (100.0, 6.03e29), // Strong insulation
    (200.0, 0.0),     // Full insulation
];
