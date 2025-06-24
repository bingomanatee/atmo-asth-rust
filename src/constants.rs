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
pub const ASTHENOSPHERE_SURFACE_START_TEMP_C: f64 = 1873.15; // Surface asthenosphere layer starting temp
pub const ASTHENOSPHERE_EQUILIBRIUM_TEMP_C: f64 = 1450.0; // Equilibrium asthenosphere temperature (above lithosphere formation)
pub const ASTHENOSPHERE_SURFACE_START_TEMP_K: f64 = ASTHENOSPHERE_SURFACE_START_TEMP_C + TO_KELVIN;
pub const ASTHENOSPHERE_EQUILIBRIUM_TEMP_K: f64 = ASTHENOSPHERE_EQUILIBRIUM_TEMP_C + TO_KELVIN;
pub const DEFAULT_LAYER_HEIGHT_KM: f64 = 10.0;


// Temperature cutoffs
pub const LITHOSPHERE_FORMATION_TEMP_K: f64 = 1973.15; // 1300 °C
pub const LITHOSPHERE_PEAK_FORMATION_TEMP_K: f64 = 1673.15;
pub const LITHOSPHERE_PEAK_FORMATION_HEIGHT: f64 = 5.0; // km height of generation
pub const LITHOSPHERE_GROWTH_RATE_KM_PER_YEAR: f64 =0.0001;
pub const LITHOSPHERE_MAX_VOLCANIC_LOSS_HEIGHT: f64 = 100.0;
pub const LITHOSPHERE_MAX_FRICTION_LOSS_HEIGHT: f64 = 5.0;
pub const MAX_DEGRADATION_KM_PER_MYR: f64 = 20.0;
pub const DEGRADATION_PER_K_ABOVE_PEAK: f64 = 0.1;

// === Global Energy Loss Based on Lithosphere Thickness (J per MIO years) ===

// Science-based cooling rates that depend on lithosphere insulation effect
// Units: Joules per year (J/y) across entire planetary surface

pub const MIO: f64 = 1_000_000.0;

pub const GLOBAL_ENERGY_LOSS_0KM_LITHOSPHERE: f64 = 1.5e27/MIO;      // Fully exposed mantle
pub const GLOBAL_ENERGY_LOSS_1KM_LITHOSPHERE: f64 = 6.03e31/MIO;     // Very thin insulation
pub const GLOBAL_ENERGY_LOSS_5KM_LITHOSPHERE: f64 = 2.7e31/MIO;      // Thin crust
pub const GLOBAL_ENERGY_LOSS_10KM_LITHOSPHERE: f64 = 7.5e26/MIO;     // Moderate insulation
pub const GLOBAL_ENERGY_LOSS_50KM_LITHOSPHERE: f64 = 2.0e26/MIO;
pub const GLOBAL_ENERGY_LOSS_100KM_LITHOSPHERE: f64 = 1.0e25/MIO;    // Thick lithosphere
pub const GLOBAL_ENERGY_LOSS_200KM_LITHOSPHERE: f64 = 0.0;       // Full insulation (no heat loss)

pub const GLOBAL_ENERGY_LOSS_TABLE: &[(f64, f64)] = &[
    (0.0, GLOBAL_ENERGY_LOSS_0KM_LITHOSPHERE),       // Exposed mantle
    (1.0, GLOBAL_ENERGY_LOSS_1KM_LITHOSPHERE),       // 1 km
    (5.0, GLOBAL_ENERGY_LOSS_5KM_LITHOSPHERE),       // 5 km
    (10.0, GLOBAL_ENERGY_LOSS_10KM_LITHOSPHERE),     // 10 km
    (50.0, GLOBAL_ENERGY_LOSS_50KM_LITHOSPHERE),     // 50 km
    (100.0, GLOBAL_ENERGY_LOSS_100KM_LITHOSPHERE),   // 100 km
    (200.0, GLOBAL_ENERGY_LOSS_200KM_LITHOSPHERE),   // 200 km
];

// yearly radiant heat input to the upper layer of the asthenosphere
pub const GLOBAL_HEAT_INPUT_ASTHENOSPHERE: f64 = 1.3e21;
