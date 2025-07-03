/// Test utilities specifically for 1km³ thermal experiments
/// 
/// This module provides test-specific helpers, assertions, and baseline constants
/// for validating thermal simulation experiments. It builds on the platonic
/// math_utils module to provide domain-specific testing functionality.

use atmo_asth_rust::math_utils::{lerp, deviation};

use atmo_asth_rust::material_composite::{get_melting_point_k, get_boiling_point_k, MaterialCompositeType};
use atmo_asth_rust::constants::{LITHOSPHERE_FORMATION_TEMP_K, LITHOSPHERE_PEAK_FORMATION_TEMP_K, ASTHENOSPHERE_SURFACE_START_TEMP_K, ASTHENOSPHERE_EQUILIBRIUM_TEMP_K};

/// Temperature baseline constants for geological materials (in Kelvin)
///
/// These functions retrieve temperature values from the material composite library
/// instead of hardcoding them, ensuring consistency with the simulation's material system.
pub mod temperature_baselines {
    use super::*;

    /// Water freezing point (0°C) - from Icy material composite
    pub fn water_freezing_k() -> f64 {
        get_melting_point_k(&MaterialCompositeType::Icy)
    }

    /// Water boiling point (100°C) - standard constant
    pub const WATER_BOILING_K: f64 = 373.15;

    /// Basalt melting range - from Basaltic material composite
    pub fn basalt_melting_min_k() -> f64 {
        get_melting_point_k(&MaterialCompositeType::Basaltic)
    }

    pub fn basalt_melting_max_k() -> f64 {
        get_boiling_point_k(&MaterialCompositeType::Basaltic)
    }

    pub fn basalt_melting_avg_k() -> f64 {
        let melting = get_melting_point_k(&MaterialCompositeType::Basaltic);
        let boiling = get_boiling_point_k(&MaterialCompositeType::Basaltic);
        (melting + boiling) / 2.0
    }

    /// Peridotite (silicate) melting range - from Silicate material composite
    pub fn peridotite_melting_min_k() -> f64 {
        get_melting_point_k(&MaterialCompositeType::Silicate)
    }

    pub fn peridotite_melting_max_k() -> f64 {
        get_boiling_point_k(&MaterialCompositeType::Silicate)
    }

    pub fn peridotite_melting_avg_k() -> f64 {
        let melting = get_melting_point_k(&MaterialCompositeType::Silicate);
        let boiling = get_boiling_point_k(&MaterialCompositeType::Silicate);
        (melting + boiling) / 2.0
    }

    /// Lithosphere formation temperatures - from constants
    pub fn lithosphere_formation_temp_k() -> f64 {
        LITHOSPHERE_FORMATION_TEMP_K
    }

    pub fn lithosphere_peak_formation_temp_k() -> f64 {
        LITHOSPHERE_PEAK_FORMATION_TEMP_K
    }

    /// Asthenosphere temperatures - from constants
    pub fn asthenosphere_surface_start_temp_k() -> f64 {
        ASTHENOSPHERE_SURFACE_START_TEMP_K
    }

    pub fn asthenosphere_equilibrium_temp_k() -> f64 {
        ASTHENOSPHERE_EQUILIBRIUM_TEMP_K
    }

    /// Phase transition threshold - use basalt melting point as reference
    pub fn phase_transition_threshold_k() -> f64 {
        basalt_melting_avg_k()
    }

    /// Iron melting point - from Metallic material composite
    pub fn iron_melting_k() -> f64 {
        let melting = get_melting_point_k(&MaterialCompositeType::Metallic);
        let boiling = get_boiling_point_k(&MaterialCompositeType::Metallic);
        (melting + boiling) / 2.0
    }
}

/// Pressure baseline constants and calculations for testing
pub mod pressure_baselines {
    /// Standard atmospheric pressure at sea level (Pa)
    pub const STANDARD_ATMOSPHERIC_PRESSURE_PA: f64 = 101325.0;
    
    /// Pressure increase per km of depth (rough geological estimate)
    pub const PRESSURE_INCREASE_PER_KM_PA: f64 = 2.7e7;  // ~270 bar/km
    
    /// Calculate pressure at depth using lithostatic pressure
    pub fn pressure_at_depth_pa(depth_km: f64, surface_pressure_pa: f64) -> f64 {
        surface_pressure_pa + (depth_km * PRESSURE_INCREASE_PER_KM_PA)
    }
    
    /// Calculate pressure-adjusted melting point
    pub fn pressure_adjusted_melting_point_k(base_melting_point_k: f64, depth_km: f64) -> f64 {
        // Pressure effect: ~50K increase per 100km depth (rough geological estimate)
        let pressure_increase_k = (depth_km / 100.0) * 50.0;
        base_melting_point_k + pressure_increase_k
    }
}

/// Check if a temperature is within a reasonable range for a material type
pub fn is_temperature_reasonable(temperature_k: f64, material_type: &str, tolerance_percent: f64) -> bool {
    let (min_reasonable, max_reasonable) = match material_type.to_lowercase().as_str() {
        "basalt" | "basaltic" => (
            temperature_baselines::basalt_melting_min_k() * (1.0 - tolerance_percent / 100.0),
            temperature_baselines::basalt_melting_max_k() * (1.0 + tolerance_percent / 100.0)
        ),
        "silicate" | "peridotite" => (
            temperature_baselines::peridotite_melting_min_k() * (1.0 - tolerance_percent / 100.0),
            temperature_baselines::peridotite_melting_max_k() * (1.0 + tolerance_percent / 100.0)
        ),
        "lithosphere" => (
            temperature_baselines::lithosphere_peak_formation_temp_k() * (1.0 - tolerance_percent / 100.0),
            temperature_baselines::lithosphere_formation_temp_k() * (1.0 + tolerance_percent / 100.0)
        ),
        "asthenosphere" => (
            temperature_baselines::asthenosphere_equilibrium_temp_k() * (1.0 - tolerance_percent / 100.0),
            temperature_baselines::asthenosphere_surface_start_temp_k() * (1.0 + tolerance_percent / 100.0)
        ),
        _ => (0.0, f64::INFINITY) // Unknown material type - allow any temperature
    };

    temperature_k >= min_reasonable && temperature_k <= max_reasonable
}

/// Assert that a temperature is within reasonable range for a material type
/// 
/// This macro checks if a temperature falls within scientifically reasonable
/// bounds for the specified material type, with configurable tolerance.
#[macro_export]
macro_rules! assert_temperature_reasonable {
    ($temperature:expr, $material_type:expr, $tolerance_percent:expr) => {
        {
            let temp = $temperature;
            let material = $material_type;
            let tolerance = $tolerance_percent;
            
            if !examples::test_utils_1km3::is_temperature_reasonable(temp, material, tolerance) {
                panic!(
                    "assertion failed: temperature {:.1}K is not reasonable for material '{}' (tolerance: {:.1}%)",
                    temp, material, tolerance
                );
            }
        }
    };
    ($temperature:expr, $material_type:expr, $tolerance_percent:expr, $($arg:tt)+) => {
        {
            let temp = $temperature;
            let material = $material_type;
            let tolerance = $tolerance_percent;
            
            if !examples::test_utils_1km3::is_temperature_reasonable(temp, material, tolerance) {
                panic!(
                    "assertion failed: temperature {:.1}K is not reasonable for material '{}' (tolerance: {:.1}%): {}",
                    temp, material, tolerance, format_args!($($arg)+)
                );
            }
        }
    };
}

/// Assert that a temperature is within a specific baseline range
/// 
/// This macro checks if a temperature falls within a specified range
/// with configurable tolerance percentage.
#[macro_export]
macro_rules! assert_temperature_baseline {
    ($actual:expr, $baseline:expr, $tolerance_percent:expr) => {
        {
            let actual_temp = $actual;
            let baseline_temp = $baseline;
            let tolerance = $tolerance_percent;
            let deviation = atmo_asth_rust::math_utils::deviation(actual_temp, baseline_temp);
            
            if deviation > tolerance {
                panic!(
                    "assertion failed: temperature {:.1}K deviates {:.2}% from baseline {:.1}K (tolerance: {:.1}%)",
                    actual_temp, deviation, baseline_temp, tolerance
                );
            }
        }
    };
    ($actual:expr, $baseline:expr, $tolerance_percent:expr, $($arg:tt)+) => {
        {
            let actual_temp = $actual;
            let baseline_temp = $baseline;
            let tolerance = $tolerance_percent;
            let deviation = atmo_asth_rust::math_utils::deviation(actual_temp, baseline_temp);
            
            if deviation > tolerance {
                panic!(
                    "assertion failed: temperature {:.1}K deviates {:.2}% from baseline {:.1}K (tolerance: {:.1}%): {}",
                    actual_temp, deviation, baseline_temp, tolerance, format_args!($($arg)+)
                );
            }
        }
    };
}

/// Test helper for validating temperature gradients in thermal experiments
pub fn validate_temperature_gradient(
    actual_temps: &[f64],
    surface_temp: f64,
    foundry_temp: f64,
    tolerance_percent: f64,
) -> Result<(), String> {
    let num_nodes = actual_temps.len();
    
    for (i, &temp) in actual_temps.iter().enumerate() {
        let expected_temp = lerp(surface_temp, foundry_temp, i as f64 / num_nodes as f64);
        let dev = deviation(temp, expected_temp);
        
        if dev > tolerance_percent {
            return Err(format!(
                "Node {} temperature {:.1}K deviates {:.2}% from expected {:.1}K (tolerance: {:.1}%)",
                i, temp, dev, expected_temp, tolerance_percent
            ));
        }
    }
    
    Ok(())
}

/// Test helper for validating monotonic temperature increase
pub fn validate_monotonic_temperature(temps: &[f64]) -> Result<(), String> {
    for i in 1..temps.len() {
        if temps[i] <= temps[i-1] {
            return Err(format!(
                "Temperature should increase monotonically: node {} ({:.1}K) <= node {} ({:.1}K)",
                i, temps[i], i-1, temps[i-1]
            ));
        }
    }
    Ok(())
}

/// Test helper for validating boundary conditions
pub fn validate_boundary_conditions(
    surface_temp: f64,
    foundry_temp: f64,
    expected_surface: f64,
    expected_foundry: f64,
    tolerance_percent: f64,
) -> Result<(), String> {
    let surface_dev = deviation(surface_temp, expected_surface);
    let foundry_dev = deviation(foundry_temp, expected_foundry);
    
    if surface_dev > tolerance_percent {
        return Err(format!(
            "Surface temperature {:.1}K deviates {:.2}% from expected {:.1}K (tolerance: {:.1}%)",
            surface_temp, surface_dev, expected_surface, tolerance_percent
        ));
    }
    
    if foundry_dev > tolerance_percent {
        return Err(format!(
            "Foundry temperature {:.1}K deviates {:.2}% from expected {:.1}K (tolerance: {:.1}%)",
            foundry_temp, foundry_dev, expected_foundry, tolerance_percent
        ));
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temperature_baselines_from_material_libraries() {
        // Test that temperature baselines are correctly retrieved from material libraries

        // Water freezing point should be 273.15K from Icy material
        assert_eq!(temperature_baselines::water_freezing_k(), 273.15);

        // Basalt melting points should match MaterialComposite values (from JSON)
        assert_eq!(temperature_baselines::basalt_melting_min_k(), 1473.0);  // melt_temp from JSON
        assert_eq!(temperature_baselines::basalt_melting_max_k(), 2900.0);   // boil_temp from JSON
        assert_eq!(temperature_baselines::basalt_melting_avg_k(), 2186.5);   // (1473 + 2900) / 2

        // Peridotite (silicate) melting points should match MaterialComposite values (from JSON)
        assert_eq!(temperature_baselines::peridotite_melting_min_k(), 1600.0);  // melt_temp from JSON
        assert_eq!(temperature_baselines::peridotite_melting_max_k(), 3200.0);  // boil_temp from JSON
        assert_eq!(temperature_baselines::peridotite_melting_avg_k(), 2400.0);  // (1600 + 3200) / 2

        // Iron melting point should match MaterialComposite values (from JSON steel data)
        assert_eq!(temperature_baselines::iron_melting_k(), 2470.5);  // (1808 + 3133) / 2

        // Lithosphere formation temperatures should match constants
        assert_eq!(temperature_baselines::lithosphere_formation_temp_k(), LITHOSPHERE_FORMATION_TEMP_K);
        assert_eq!(temperature_baselines::lithosphere_peak_formation_temp_k(), LITHOSPHERE_PEAK_FORMATION_TEMP_K);

        // Asthenosphere temperatures should match constants
        assert_eq!(temperature_baselines::asthenosphere_surface_start_temp_k(), ASTHENOSPHERE_SURFACE_START_TEMP_K);
        assert_eq!(temperature_baselines::asthenosphere_equilibrium_temp_k(), ASTHENOSPHERE_EQUILIBRIUM_TEMP_K);

        // Phase transition threshold should use basalt melting point
        assert_eq!(temperature_baselines::phase_transition_threshold_k(), temperature_baselines::basalt_melting_avg_k());
    }

    #[test]
    fn test_is_temperature_reasonable_with_material_libraries() {
        // Test that temperature reasonableness checks work with material library values

        // Basalt temperatures (range: 1473K - 2900K from JSON)
        assert!(is_temperature_reasonable(temperature_baselines::basalt_melting_avg_k(), "basalt", 5.0));
        assert!(is_temperature_reasonable(1500.0, "basaltic", 10.0)); // Within range
        assert!(is_temperature_reasonable(2000.0, "basalt", 10.0)); // Within range (below 2900K boiling point)
        assert!(!is_temperature_reasonable(3500.0, "basalt", 10.0)); // Too hot (above boiling point)
        assert!(!is_temperature_reasonable(1000.0, "basalt", 10.0)); // Too cold (below melting point)

        // Silicate temperatures (range: 1600K - 3200K from JSON)
        assert!(is_temperature_reasonable(temperature_baselines::peridotite_melting_avg_k(), "silicate", 5.0));
        assert!(is_temperature_reasonable(1750.0, "peridotite", 10.0)); // Within range
        assert!(is_temperature_reasonable(2500.0, "silicate", 10.0)); // Within range

        // Lithosphere temperatures
        assert!(is_temperature_reasonable(temperature_baselines::lithosphere_peak_formation_temp_k(), "lithosphere", 5.0));

        // Asthenosphere temperatures
        assert!(is_temperature_reasonable(temperature_baselines::asthenosphere_equilibrium_temp_k(), "asthenosphere", 5.0));
    }

    #[test]
    fn test_pressure_baselines() {
        // Test pressure calculations
        let surface_pressure = pressure_baselines::STANDARD_ATMOSPHERIC_PRESSURE_PA;
        let pressure_at_1km = pressure_baselines::pressure_at_depth_pa(1.0, surface_pressure);
        let pressure_at_10km = pressure_baselines::pressure_at_depth_pa(10.0, surface_pressure);

        assert!(pressure_at_1km > surface_pressure);
        assert!(pressure_at_10km > pressure_at_1km);

        // Test pressure-adjusted melting point
        let base_melting = temperature_baselines::basalt_melting_avg_k();
        let adjusted_100km = pressure_baselines::pressure_adjusted_melting_point_k(base_melting, 100.0);
        let adjusted_200km = pressure_baselines::pressure_adjusted_melting_point_k(base_melting, 200.0);

        assert!(adjusted_100km > base_melting);
        assert!(adjusted_200km > adjusted_100km);
        assert_eq!(adjusted_100km, base_melting + 50.0); // 50K increase per 100km
    }
}

pub fn main() {
    
}