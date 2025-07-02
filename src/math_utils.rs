/// Mathematical utility functions for the thermal simulation
///
/// This module provides common mathematical operations used throughout
/// the thermal simulation system.

/// Assert that the deviation between two values is less than a threshold
///
/// This macro combines deviation calculation with assertion for cleaner test code.
/// It calculates the percentage deviation between `actual` and `expected`, then
/// asserts that this deviation is less than the specified `max_deviation`.
///
/// # Examples
/// See the test cases below for usage examples.
#[macro_export]
macro_rules! assert_deviation {
    ($actual:expr, $expected:expr, $max_deviation:expr) => {
        {
            let actual_val = $actual;
            let expected_val = $expected;
            let max_dev = $max_deviation;
            let actual_deviation = $crate::math_utils::deviation(actual_val, expected_val);

            if actual_deviation >= max_dev {
                panic!(
                    "assertion failed: deviation {:.2}% >= {:.2}%\n  actual: {:?},\n  expected: {:?}",
                    actual_deviation, max_dev, actual_val, expected_val
                );
            }
        }
    };
    ($actual:expr, $expected:expr, $max_deviation:expr, $($arg:tt)+) => {
        {
            let actual_val = $actual;
            let expected_val = $expected;
            let max_dev = $max_deviation;
            let actual_deviation = $crate::math_utils::deviation(actual_val, expected_val);

            if actual_deviation >= max_dev {
                panic!(
                    "assertion failed: deviation {:.2}% >= {:.2}%: {}\n  actual: {:?},\n  expected: {:?}",
                    actual_deviation, max_dev, format_args!($($arg)+), actual_val, expected_val
                );
            }
        }
    };
}



/// Linear interpolation between two values
/// 
/// # Arguments
/// * `a` - Start value
/// * `b` - End value  
/// * `ratio` - Interpolation ratio (0.0 = a, 1.0 = b)
/// 
/// # Returns
/// The interpolated value between a and b
/// 
/// # Examples
/// ```
/// use atmo_asth_rust::math_utils::lerp;
/// 
/// // Basic interpolation
/// assert_eq!(lerp(0.0, 10.0, 0.5), 5.0);
/// assert_eq!(lerp(100.0, 200.0, 0.25), 125.0);
/// 
/// // Temperature gradient example
/// let surface_temp = 300.0;
/// let foundry_temp = 7200.0;
/// let quarter_temp = lerp(surface_temp, foundry_temp, 0.25);
/// ```
pub fn lerp(a: f64, b: f64, ratio: f64) -> f64 {
    a + (b - a) * ratio
}

pub fn un_lerp(a: f64, b: f64, value: f64) -> f64 {
     if b > a {
         un_lerp(b, a, value)
     } else if b == a {
         0.5
     } else {
         (value - a) / b - a
     }
}
/// Linear interpolation with index-based ratio calculation
/// 
/// Convenience function for interpolating based on array indices.
/// Automatically calculates the ratio as index / total_count.
/// 
/// # Arguments
/// * `a` - Start value
/// * `b` - End value
/// * `index` - Current index
/// * `total_count` - Total number of elements
/// 
/// # Returns
/// The interpolated value at the given index position
/// 
/// # Examples
/// ```
/// use atmo_asth_rust::math_utils::lerp_indexed;
/// 
/// // Temperature at 25% through 60 nodes
/// let temp = lerp_indexed(300.0, 7200.0, 15, 60);
/// ```
pub fn lerp_indexed(a: f64, b: f64, index: usize, total_count: usize) -> f64 {
    let ratio = index as f64 / total_count as f64;
    lerp(a, b, ratio)
}

/// Inverse linear interpolation - find the ratio for a given value
/// 
/// Given a value between a and b, returns the ratio (0.0 to 1.0)
/// that would produce that value via linear interpolation.
/// 
/// # Arguments
/// * `a` - Start value
/// * `b` - End value
/// * `value` - The value to find the ratio for
/// 
/// # Returns
/// The ratio (0.0 to 1.0) that produces the given value
/// 
/// # Examples
/// ```
/// use atmo_asth_rust::math_utils::inverse_lerp;
/// 
/// let ratio = inverse_lerp(100.0, 200.0, 150.0);
/// assert_eq!(ratio, 0.5);
/// ```
pub fn inverse_lerp(a: f64, b: f64, value: f64) -> f64 {
    if (b - a).abs() < f64::EPSILON {
        0.0 // Avoid division by zero
    } else {
        (value - a) / (b - a)
    }
}

/// Clamp a value between minimum and maximum bounds
/// 
/// # Arguments
/// * `value` - The value to clamp
/// * `min` - Minimum allowed value
/// * `max` - Maximum allowed value
/// 
/// # Returns
/// The clamped value
/// 
/// # Examples
/// ```
/// use atmo_asth_rust::math_utils::clamp;
/// 
/// assert_eq!(clamp(5.0, 0.0, 10.0), 5.0);
/// assert_eq!(clamp(-5.0, 0.0, 10.0), 0.0);
/// assert_eq!(clamp(15.0, 0.0, 10.0), 10.0);
/// ```
pub fn clamp(value: f64, min: f64, max: f64) -> f64 {
    if value < min {
        min
    } else if value > max {
        max
    } else {
        value
    }
}

/// Remap a value from one range to another
///
/// # Arguments
/// * `value` - The value to remap
/// * `from_min` - Minimum of the source range
/// * `from_max` - Maximum of the source range
/// * `to_min` - Minimum of the target range
/// * `to_max` - Maximum of the target range
///
/// # Returns
/// The remapped value
///
/// # Examples
/// ```
/// use atmo_asth_rust::math_utils::remap;
///
/// // Remap 50 from range [0,100] to range [0,1000]
/// assert_eq!(remap(50.0, 0.0, 100.0, 0.0, 1000.0), 500.0);
/// ```
pub fn remap(value: f64, from_min: f64, from_max: f64, to_min: f64, to_max: f64) -> f64 {
    let ratio = inverse_lerp(from_min, from_max, value);
    lerp(to_min, to_max, ratio)
}

/// Calculate the percentage deviation between two values
///
/// Returns the percentage difference of `actual` from `expected`.
/// Uses the expected value as the reference (base) for the percentage calculation.
///
/// # Arguments
/// * `actual` - The actual measured value
/// * `expected` - The expected reference value
///
/// # Returns
/// The percentage deviation as a positive f64 (absolute difference)
///
/// # Examples
/// ```
/// use atmo_asth_rust::math_utils::deviation;
///
/// // 105 is 5% higher than 100
/// assert_eq!(deviation(105.0, 100.0), 5.0);
///
/// // 95 is 5% lower than 100
/// assert_eq!(deviation(95.0, 100.0), 5.0);
///
/// // Temperature test example
/// let actual_temp = 1523.5;
/// let expected_temp = 1500.0;
/// let dev = deviation(actual_temp, expected_temp);
/// assert!(dev < 2.0); // Within 2% tolerance
/// ```
pub fn deviation(actual: f64, expected: f64) -> f64 {
    if expected.abs() < f64::EPSILON {
        // Avoid division by zero - if expected is 0, return 0 if actual is also 0
        if actual.abs() < f64::EPSILON {
            0.0
        } else {
            f64::INFINITY // Infinite deviation if expected is 0 but actual is not
        }
    } else {
        ((actual - expected).abs() / expected.abs()) * 100.0
    }
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lerp() {
        assert_eq!(lerp(0.0, 10.0, 0.0), 0.0);
        assert_eq!(lerp(0.0, 10.0, 1.0), 10.0);
        assert_eq!(lerp(0.0, 10.0, 0.5), 5.0);
        assert_eq!(lerp(100.0, 200.0, 0.25), 125.0);
    }

    #[test]
    fn test_lerp_indexed() {
        assert_eq!(lerp_indexed(0.0, 100.0, 0, 4), 0.0);
        assert_eq!(lerp_indexed(0.0, 100.0, 2, 4), 50.0);
        assert_eq!(lerp_indexed(0.0, 100.0, 4, 4), 100.0);
    }

    #[test]
    fn test_inverse_lerp() {
        assert_eq!(inverse_lerp(0.0, 10.0, 5.0), 0.5);
        assert_eq!(inverse_lerp(100.0, 200.0, 150.0), 0.5);
        assert_eq!(inverse_lerp(100.0, 200.0, 100.0), 0.0);
        assert_eq!(inverse_lerp(100.0, 200.0, 200.0), 1.0);
    }

    #[test]
    fn test_clamp() {
        assert_eq!(clamp(5.0, 0.0, 10.0), 5.0);
        assert_eq!(clamp(-5.0, 0.0, 10.0), 0.0);
        assert_eq!(clamp(15.0, 0.0, 10.0), 10.0);
    }

    #[test]
    fn test_remap() {
        assert_eq!(remap(50.0, 0.0, 100.0, 0.0, 1000.0), 500.0);
        assert_eq!(remap(25.0, 0.0, 100.0, 0.0, 10.0), 2.5);
    }

    #[test]
    fn test_deviation() {
        // Basic percentage calculations
        assert_eq!(deviation(105.0, 100.0), 5.0);
        assert_eq!(deviation(95.0, 100.0), 5.0);
        assert_eq!(deviation(100.0, 100.0), 0.0);

        // Temperature examples
        assert_eq!(deviation(1500.0, 1500.0), 0.0);
        assert!((deviation(1530.0, 1500.0) - 2.0).abs() < 0.001);
        assert!((deviation(1470.0, 1500.0) - 2.0).abs() < 0.001);

        // Edge cases
        assert_eq!(deviation(0.0, 0.0), 0.0);
        assert_eq!(deviation(10.0, 0.0), f64::INFINITY);
    }

    #[test]
    fn test_assert_deviation_macro() {
        // Basic usage - should pass
        assert_deviation!(105.0, 100.0, 10.0);  // 5% < 10%
        assert_deviation!(95.0, 100.0, 10.0);   // 5% < 10%
        assert_deviation!(100.0, 100.0, 1.0);   // 0% < 1%

        // With expressions
        assert_deviation!(2.0 * 52.5, 100.0, 10.0);  // 5% < 10%

        // With custom message
        assert_deviation!(1530.0, 1500.0, 5.0, "Temperature should be within 5%");
    }

    #[test]
    #[should_panic(expected = "assertion failed: deviation")]
    fn test_assert_deviation_macro_fails() {
        assert_deviation!(120.0, 100.0, 10.0);  // 20% >= 10%, should panic
    }

}
