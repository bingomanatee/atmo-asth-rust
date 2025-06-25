// src/sim_op_lithosphere

use crate::asth_cell::AsthCellLithosphere;
use crate::h3_utils::H3Utils;
use crate::material::MaterialType;
use crate::sim::Simulation;
use crate::sim::sim_op::{SimOp, SimOpHandle};
use noise::{NoiseFn, Perlin};

#[derive(Debug, Clone)]
pub struct LithosphereOp {
    pub material_distribution: Vec<(MaterialType, f64)>, // must sum to 1.0
    pub seed: u32,
    pub scale: f64,
}

impl LithosphereOp {
    pub fn new(material_distribution: Vec<(MaterialType, f64)>, seed: u32, scale: f64) -> Self {
        let normalized_distribution = Self::normalize_distribution(material_distribution);
        Self {
            material_distribution: normalized_distribution,
            seed,
            scale,
        }
    }

    /// Normalize the material distribution so that all weights sum to 1.0
    fn normalize_distribution(
        distribution: Vec<(MaterialType, f64)>,
    ) -> Vec<(MaterialType, f64)> {
        if distribution.is_empty() {
            return distribution;
        }

        // Calculate the sum of all weights
        let total_weight: f64 = distribution.iter().map(|(_, weight)| weight).sum();

        // If total is 0 or very close to 0, distribute equally
        if total_weight <= f64::EPSILON {
            let equal_weight = 1.0 / distribution.len() as f64;
            return distribution
                .into_iter()
                .map(|(material, _)| (material, equal_weight))
                .collect();
        }

        // Normalize each weight by dividing by the total
        distribution
            .into_iter()
            .map(|(material, weight)| (material, weight / total_weight))
            .collect()
    }

    pub fn handle(
        material_distribution: Vec<(MaterialType, f64)>,
        seed: u32,
        scale: f64,
        growth_rate_km_per_step: f64,
        formation_threshold_energy: f64,
    ) -> SimOpHandle {
        SimOpHandle::new(Box::new(Self::new(material_distribution, seed, scale)))
    }

    fn pick_material(&self, value: f64) -> MaterialType {
        let mut acc = 0.0;
        for (ty, weight) in &self.material_distribution {
            acc += weight;
            if value <= acc {
                return *ty;
            }
        }
        self.material_distribution.last().unwrap().0
    }
}

impl SimOp for LithosphereOp {
    fn init_sim(&mut self, sim: &mut Simulation) {
        let noise = Perlin::new(self.seed);

        for (cell_id, column) in sim.cells.iter_mut() {
            let point = H3Utils::cell_to_3d_point(*cell_id, sim.planet.radius_km);
            let scaled_point = point * self.scale as f32;
            let point_array = scaled_point.to_array().map(|x| x as f64);
            let random_value = noise.get(point_array);
            let scaled_val = (random_value + 1.0) / 2.0; // normalize to 0.0–1.0

            let material = self.pick_material(scaled_val);
            column.lithospheres_next.push(AsthCellLithosphere::new(0.0, material, 0.0));
        }
    }

    fn update_sim(&mut self, sim: &mut Simulation) {
        // Add lithosphere growth during simulation steps based on material properties
        for column in sim.cells.values_mut() {
            // First, get the top layer energy to avoid borrowing conflicts
            let top = {
                let (_, top_layer) = column.layer(0);
                top_layer.clone()
            };
            let area = column.area();
            // Now get lithosphere (panics if none exists - should be created during init)
            let (_, next_lithosphere, profile) = column.lithosphere();
            
            let kelvin = top.kelvin();

            let growth_km_per_year = profile.growth_at_kelvin(kelvin);
            let growth_km_per_step = growth_km_per_year * sim.years_per_step as f64;

            if growth_km_per_step > 0.0 {
                next_lithosphere.height_km += growth_km_per_step;
                let new_volume = area * next_lithosphere.height_km;
                next_lithosphere.set_volume_km3(new_volume);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_distribution() {
        // Test with weights that don't sum to 1
        let distribution = vec![
            (MaterialType::Silicate, 2.0),
            (MaterialType::Basaltic, 3.0),
            (MaterialType::Metallic, 5.0),
        ];

        let normalized = LithosphereOp::normalize_distribution(distribution);

        // Check that weights sum to 1.0
        let total: f64 = normalized.iter().map(|(_, weight)| weight).sum();
        assert!(
            (total - 1.0).abs() < f64::EPSILON,
            "Total weight should be 1.0, got {}",
            total
        );

        // Check individual weights are correct (2/10, 3/10, 5/10)
        assert!((normalized[0].1 - 0.2).abs() < f64::EPSILON);
        assert!((normalized[1].1 - 0.3).abs() < f64::EPSILON);
        assert!((normalized[2].1 - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_normalize_distribution_zero_weights() {
        // Test with all zero weights - should distribute equally
        let distribution = vec![
            (MaterialType::Silicate, 0.0),
            (MaterialType::Basaltic, 0.0),
            (MaterialType::Metallic, 0.0),
        ];

        let normalized = LithosphereOp::normalize_distribution(distribution);

        // Check that weights sum to 1.0
        let total: f64 = normalized.iter().map(|(_, weight)| weight).sum();
        assert!(
            (total - 1.0).abs() < f64::EPSILON,
            "Total weight should be 1.0, got {}",
            total
        );

        // Check that each weight is 1/3
        for (_, weight) in normalized {
            assert!(
                (weight - 1.0 / 3.0).abs() < f64::EPSILON,
                "Each weight should be 1/3, got {}",
                weight
            );
        }
    }

    #[test]
    fn test_normalize_distribution_already_normalized() {
        // Test with weights that already sum to 1
        let distribution = vec![
            (MaterialType::Silicate, 0.3),
            (MaterialType::Basaltic, 0.4),
            (MaterialType::Metallic, 0.3),
        ];

        let normalized = LithosphereOp::normalize_distribution(distribution.clone());

        // Check that weights sum to 1.0
        let total: f64 = normalized.iter().map(|(_, weight)| weight).sum();
        assert!(
            (total - 1.0).abs() < f64::EPSILON,
            "Total weight should be 1.0, got {}",
            total
        );

        // Check that weights are unchanged (within floating point precision)
        for (i, (_, weight)) in normalized.iter().enumerate() {
            assert!((weight - distribution[i].1).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn test_lithosphere_op_normalization() {
        // Test that the constructor normalizes the distribution
        let distribution = vec![
            (MaterialType::Silicate, 10.0),
            (MaterialType::Basaltic, 20.0),
        ];

        let op = LithosphereOp::new(distribution, 42, 0.1);

        // Check that the stored distribution is normalized
        let total: f64 = op
            .material_distribution
            .iter()
            .map(|(_, weight)| weight)
            .sum();
        assert!(
            (total - 1.0).abs() < f64::EPSILON,
            "Total weight should be 1.0, got {}",
            total
        );

        // Check individual weights (10/30 = 1/3, 20/30 = 2/3)
        assert!((op.material_distribution[0].1 - 1.0 / 3.0).abs() < f64::EPSILON);
        assert!((op.material_distribution[1].1 - 2.0 / 3.0).abs() < f64::EPSILON);

    }

    // Helper function to create a test simulation with specific surface temperature in Kelvin
    fn create_test_simulation_with_surface_kelvin(surface_kelvin: f64) -> crate::sim::Simulation {
        use crate::sim::{Simulation, SimProps};
        use crate::planet::Planet;
        use h3o::Resolution;

        let props = SimProps {
            planet: Planet::new(6371.0, Resolution::Two),
            name: "lithosphere_test",
            ops: vec![], // No ops for basic testing
            res: Resolution::Two,
            layer_count: 10,
            layer_height: 10.0,
            layer_height_km: 10.0,
            sim_steps: 100,
            years_per_step: 1000,
            alert_freq: 10,
            starting_surface_temp_k: surface_kelvin,
            debug: false,
        };

        let mut sim = Simulation::new(props);
        sim.make_cells();
        sim
    }

    fn setup_silicate_only_op() -> LithosphereOp {
        LithosphereOp::new(
            vec![(MaterialType::Silicate, 1.0)], // Only Silicate
            42,
            1.0,
        )
    }

    #[test]
    fn test_lithosphere_at_formation_temperature() {
        // Test at formation temperature (1873.15 K) - should have minimal growth
        let formation_kelvin = 1873.15;
        let mut sim = create_test_simulation_with_surface_kelvin(formation_kelvin);
        let mut op = setup_silicate_only_op();

        // Initialize and run one step
        op.init_sim(&mut sim);
        op.update_sim(&mut sim);

        // Check lithosphere growth
        for column in sim.cells.values() {
            let (_, _, profile) = &mut column.clone().lithosphere();
            assert_eq!(profile.kind, MaterialType::Silicate);

            // At formation temp, growth should be minimal (close to 0)
            let actual_height = column.lithospheres_next[0].height_km;
            assert!(actual_height < 0.0001, "Height should be near 0 at formation temp, got {}", actual_height);
        }
    }

    #[test]
    fn test_lithosphere_halfway_between_formation_and_peak() {
        // Test halfway between formation (1873.15) and peak (1673.15) = 1773.15 K
        let halfway_kelvin = (1873.15 + 1673.15) / 2.0; // 1773.15 K
        let mut sim = create_test_simulation_with_surface_kelvin(halfway_kelvin);
        let mut op = setup_silicate_only_op();

        op.init_sim(&mut sim);
        op.update_sim(&mut sim);

        for column in sim.cells.values() {
            let mut column_clone = column.clone();
            let (current_layer, _) = column_clone.layer(0);
            let actual_temp = current_layer.kelvin();
            let (_, _, profile) = column_clone.lithosphere();

            // At halfway temp, should have partial growth
            let expected_growth_rate = profile.growth_at_kelvin(actual_temp);
            let expected_height = expected_growth_rate * sim.years_per_step as f64;

            let actual_height = column.lithospheres_next[0].height_km;
            let tolerance = 0.001;
            assert!((actual_height - expected_height).abs() < tolerance,
                   "Expected height ~{:.3}, got {:.3}", expected_height, actual_height);
        }
    }

    #[test]
    fn test_lithosphere_at_peak_growth_temperature() {
        // Test at peak growth temperature (1673.15 K) - should have maximum growth
        let peak_growth_kelvin = 1673.15;
        let mut sim = create_test_simulation_with_surface_kelvin(peak_growth_kelvin);
        let mut op = setup_silicate_only_op();

        op.init_sim(&mut sim);

        // Debug: Check what temperature we actually get
        for column in sim.cells.values() {
            let mut column_clone = column.clone();
            let (current_layer, _) = column_clone.layer(0);
            let actual_temp = current_layer.kelvin();
            println!("Surface temp set to: 1673.15 K, actual layer temp: {:.2} K", actual_temp);

            let (_, _, profile) = column_clone.lithosphere();
            let growth_rate = profile.growth_at_kelvin(actual_temp);
            println!("Growth rate at {:.2} K: {:.6} km/year", actual_temp, growth_rate);
            break; // Just check first cell
        }

        op.update_sim(&mut sim);

        for column in sim.cells.values() {
            let (_, _, profile) = &mut column.clone().lithosphere();

            // Now the implementation correctly scales by years_per_step
            let expected_height = profile.max_lith_growth_km_per_year * sim.years_per_step as f64;
            let actual_height = column.lithospheres_next[0].height_km;

            let tolerance = 0.001;
            assert!((actual_height - expected_height).abs() < tolerance,
                   "Expected height {:.3} km, got {:.3} km", expected_height, actual_height);
            assert!((actual_height - 1.0).abs() < tolerance, "Should be ~1 km at peak temp");
        }
    }

    #[test]
    fn test_lithosphere_below_peak_temperature() {
        // Test below peak temperature (1573.15 K) - should have partial growth
        let below_peak_kelvin = 1573.15;
        let mut sim = create_test_simulation_with_surface_kelvin(below_peak_kelvin);
        let mut op = setup_silicate_only_op();

        op.init_sim(&mut sim);
        op.update_sim(&mut sim);

        for column in sim.cells.values() {
            let (_, _, profile) = &mut column.clone().lithosphere();

            // Below peak temp, should have maximum growth
            let expected_height = profile.max_lith_growth_km_per_year * sim.years_per_step as f64;
            let actual_height = column.lithospheres_next[0].height_km;

            let tolerance = 0.001;
            assert!((actual_height - expected_height).abs() < tolerance,
                   "Expected height {:.3} km, got {:.3} km", expected_height, actual_height);
            assert!((actual_height - 1.0).abs() < tolerance, "Should be ~1 km below peak temp");
        }
    }

    #[test]
    fn test_lithosphere_above_formation_temperature() {
        // Test above formation temperature (1973.15 K) - should have no growth
        let above_formation_kelvin = 1973.15;
        let mut sim = create_test_simulation_with_surface_kelvin(above_formation_kelvin);
        let mut op = setup_silicate_only_op();

        op.init_sim(&mut sim);
        op.update_sim(&mut sim);

        for column in sim.cells.values() {
            // Above formation temp, should have no growth (stays at initial 0.0)
            let actual_height = column.lithospheres_next[0].height_km;
            assert!(actual_height < 0.5, "Should have no growth above formation temp, got {}", actual_height);
        }
    }

    #[test]
    fn test_lithosphere_multi_step_growth() {
        // Test growth over multiple simulation steps at peak temperature
        let peak_growth_kelvin = 1673.15;
        let mut sim = create_test_simulation_with_surface_kelvin(peak_growth_kelvin);
        let mut op = setup_silicate_only_op();

        op.init_sim(&mut sim);

        // Run 3 simulation steps
        for _ in 0..3 {
            op.update_sim(&mut sim);
            // Commit changes on each column
            for column in sim.cells.values_mut() {
                column.commit_next_layers();
            }
        }

        for column in sim.cells.values() {
            let (_, _, profile) = &mut column.clone().lithosphere();

            // After 3 steps at peak temp, should have 3x the growth
            let expected_height = profile.max_lith_growth_km_per_year * sim.years_per_step as f64 * 3.0;
            let actual_height = column.lithospheres[0].height_km;

            let tolerance = 0.001;
            assert!((actual_height - expected_height).abs() < tolerance,
                   "Expected height {:.3} km after 3 steps, got {:.3} km", expected_height, actual_height);
            assert!((actual_height - 3.0).abs() < tolerance, "Should be ~3 km after 3 steps");
        }
    }
}
