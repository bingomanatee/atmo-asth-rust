// src/sim_op_init_lithosphere.rs

use crate::sim::sim_op::{SimOp, SimOpHandle};
use crate::sim::{Simulation};
use crate::lithosphere::LithosphereType;
use noise::{NoiseFn, Perlin};
use crate::h3_utils::H3Utils;
use crate::asth_cell::AsthCellLithosphere;

#[derive(Debug, Clone)]
pub struct LithosphereOp {
    pub material_distribution: Vec<(LithosphereType, f64)>, // must sum to 1.0
    pub seed: u32,
    pub scale: f64,
    pub growth_rate_km_per_step: f64,
    pub formation_threshold_energy: f64,
}

impl LithosphereOp {
    pub fn new(
        material_distribution: Vec<(LithosphereType, f64)>,
        seed: u32,
        scale: f64,
        growth_rate_km_per_step: f64,
        formation_threshold_energy: f64,
    ) -> Self {
        let normalized_distribution = Self::normalize_distribution(material_distribution);
        Self {
            material_distribution: normalized_distribution,
            seed,
            scale,
            growth_rate_km_per_step,
            formation_threshold_energy,
        }
    }

    /// Normalize the material distribution so that all weights sum to 1.0
    fn normalize_distribution(distribution: Vec<(LithosphereType, f64)>) -> Vec<(LithosphereType, f64)> {
        if distribution.is_empty() {
            return distribution;
        }

        // Calculate the sum of all weights
        let total_weight: f64 = distribution.iter().map(|(_, weight)| weight).sum();

        // If total is 0 or very close to 0, distribute equally
        if total_weight <= f64::EPSILON {
            let equal_weight = 1.0 / distribution.len() as f64;
            return distribution.into_iter()
                .map(|(material, _)| (material, equal_weight))
                .collect();
        }

        // Normalize each weight by dividing by the total
        distribution.into_iter()
            .map(|(material, weight)| (material, weight / total_weight))
            .collect()
    }

    pub fn handle(
        material_distribution: Vec<(LithosphereType, f64)>,
        seed: u32,
        scale: f64,
        growth_rate_km_per_step: f64,
        formation_threshold_energy: f64,
    ) -> SimOpHandle {
        SimOpHandle::new(Box::new(Self::new(
            material_distribution,
            seed,
            scale,
            growth_rate_km_per_step,
            formation_threshold_energy,
        )))
    }

    fn pick_material(&self, value: f64) -> LithosphereType {
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
            column.lithosphere_next.push(AsthCellLithosphere {
                height_km: 0.0,
                volume_km3: 0.0,
                material,
            });
        }
    }

    fn update_sim(&mut self, sim: &mut Simulation) {
        // Add lithosphere growth during simulation steps
        for column in sim.cells.values_mut() {
            // Check if the top layer has enough energy to form lithosphere
            if let Some(top_layer) = column.layers_next.get(0) {
                if top_layer.energy_joules > self.formation_threshold_energy {
                    // Find existing lithosphere layers or create new ones
                    for lithosphere_layer in &mut column.lithosphere_next {
                        // Add growth to existing layers
                        lithosphere_layer.height_km += self.growth_rate_km_per_step;

                        // Update volume based on height (simplified calculation)
                        // In a real simulation, this would use proper cell area calculations
                        lithosphere_layer.volume_km3 = lithosphere_layer.height_km * 100.0; // Placeholder
                    }

                    // If no lithosphere exists yet, create a new layer
                    if column.lithosphere_next.is_empty() {
                        let noise = Perlin::new(self.seed);
                        let point = H3Utils::cell_to_3d_point(column.cell_index, sim.planet.radius_km);
                        let scaled_point = point * self.scale as f32;
                        let nval = noise.get(scaled_point.to_array().map(|x| x as f64));
                        let scaled_val = (nval + 1.0) / 2.0;

                        let material = self.pick_material(scaled_val);
                        column.lithosphere_next.push(AsthCellLithosphere {
                            height_km: self.growth_rate_km_per_step,
                            volume_km3: self.growth_rate_km_per_step * 100.0, // Placeholder
                            material,
                        });
                    }
                }
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
            (LithosphereType::Silicate, 2.0),
            (LithosphereType::Basaltic, 3.0),
            (LithosphereType::Metallic, 5.0),
        ];

        let normalized = LithosphereOp::normalize_distribution(distribution);

        // Check that weights sum to 1.0
        let total: f64 = normalized.iter().map(|(_, weight)| weight).sum();
        assert!((total - 1.0).abs() < f64::EPSILON, "Total weight should be 1.0, got {}", total);

        // Check individual weights are correct (2/10, 3/10, 5/10)
        assert!((normalized[0].1 - 0.2).abs() < f64::EPSILON);
        assert!((normalized[1].1 - 0.3).abs() < f64::EPSILON);
        assert!((normalized[2].1 - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_normalize_distribution_zero_weights() {
        // Test with all zero weights - should distribute equally
        let distribution = vec![
            (LithosphereType::Silicate, 0.0),
            (LithosphereType::Basaltic, 0.0),
            (LithosphereType::Metallic, 0.0),
        ];

        let normalized = LithosphereOp::normalize_distribution(distribution);

        // Check that weights sum to 1.0
        let total: f64 = normalized.iter().map(|(_, weight)| weight).sum();
        assert!((total - 1.0).abs() < f64::EPSILON, "Total weight should be 1.0, got {}", total);

        // Check that each weight is 1/3
        for (_, weight) in normalized {
            assert!((weight - 1.0/3.0).abs() < f64::EPSILON, "Each weight should be 1/3, got {}", weight);
        }
    }

    #[test]
    fn test_normalize_distribution_already_normalized() {
        // Test with weights that already sum to 1
        let distribution = vec![
            (LithosphereType::Silicate, 0.3),
            (LithosphereType::Basaltic, 0.4),
            (LithosphereType::Metallic, 0.3),
        ];

        let normalized = LithosphereOp::normalize_distribution(distribution.clone());

        // Check that weights sum to 1.0
        let total: f64 = normalized.iter().map(|(_, weight)| weight).sum();
        assert!((total - 1.0).abs() < f64::EPSILON, "Total weight should be 1.0, got {}", total);

        // Check that weights are unchanged (within floating point precision)
        for (i, (_, weight)) in normalized.iter().enumerate() {
            assert!((weight - distribution[i].1).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn test_lithosphere_op_normalization() {
        // Test that the constructor normalizes the distribution
        let distribution = vec![
            (LithosphereType::Silicate, 10.0),
            (LithosphereType::Basaltic, 20.0),
        ];

        let op = LithosphereOp::new(distribution, 42, 0.1, 0.01, 1e20);

        // Check that the stored distribution is normalized
        let total: f64 = op.material_distribution.iter().map(|(_, weight)| weight).sum();
        assert!((total - 1.0).abs() < f64::EPSILON, "Total weight should be 1.0, got {}", total);

        // Check individual weights (10/30 = 1/3, 20/30 = 2/3)
        assert!((op.material_distribution[0].1 - 1.0/3.0).abs() < f64::EPSILON);
        assert!((op.material_distribution[1].1 - 2.0/3.0).abs() < f64::EPSILON);

        // Check that new fields are set correctly
        assert!((op.growth_rate_km_per_step - 0.01).abs() < f64::EPSILON);
        assert!((op.formation_threshold_energy - 1e20).abs() < f64::EPSILON);
    }
}
