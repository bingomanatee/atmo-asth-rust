use crate::asth_cell::AsthCellLithosphere;
use crate::h3_utils::H3Utils;
use crate::material::MaterialType;
use crate::sim::sim_op::{SimOp, SimOpHandle};
use crate::sim::simulation::Simulation;
use noise::{NoiseFn, Perlin};

/// Unified Lithosphere Operator
///
/// This operator handles both lithosphere formation and melting in a single, unified system:
///
/// **Formation**: When asthenosphere temperature drops below material formation threshold:
/// - Grows lithosphere based on material-specific growth rates
/// - Respects maximum height limits per material
///
/// **Melting**: When asthenosphere temperature exceeds formation threshold:
/// - Melts lithosphere back into magma using `melt_from_below_km_per_year`
/// - Adds melted energy back to asthenosphere
///
/// This creates the complete thermal cycle:
/// 1. Cool → Lithosphere forms → Insulation → Less cooling → Temperature rises
/// 2. Hot → Lithosphere melts → Less insulation → More cooling → Temperature drops
/// 3. Natural equilibrium between formation and melting
pub struct LithosphereUnifiedOp {
    /// Material distribution for lithosphere formation
    pub materials: Vec<(MaterialType, f64)>,

    /// Random seed for material distribution
    pub seed: u32,

    /// Noise scale for material variation
    pub scale: f64,
}

impl LithosphereUnifiedOp {
    /// Create a new unified lithosphere operator
    ///
    /// # Arguments
    /// * `materials` - Vector of (MaterialType, frequency) pairs for lithosphere formation
    /// * `seed` - Random seed for material distribution
    /// * `scale` - Noise scale for material variation
    pub fn new(materials: Vec<(MaterialType, f64)>, seed: u32, scale: f64) -> Self {
        Self {
            materials,
            seed,
            scale,
        }
    }

    /// Create a handle for the unified lithosphere operator
    pub fn handle(materials: Vec<(MaterialType, f64)>, seed: u32, scale: f64) -> SimOpHandle {
        SimOpHandle::new(Box::new(Self::new(materials, seed, scale)))
    }

    /// Pick a material based on noise value and material frequencies
    fn pick_material(&self, noise_value: f64) -> MaterialType {
        // Normalize frequencies to sum to 1.0
        let total_freq: f64 = self.materials.iter().map(|(_, freq)| freq).sum();

        if total_freq == 0.0 {
            return self
                .materials
                .first()
                .map(|(mat, _)| *mat)
                .unwrap_or(MaterialType::Silicate);
        }

        let mut cumulative = 0.0;
        let normalized_noise = noise_value * total_freq;

        for (material, frequency) in &self.materials {
            cumulative += frequency;
            if normalized_noise <= cumulative {
                return *material;
            }
        }

        // Fallback to first material
        self.materials
            .first()
            .map(|(mat, _)| *mat)
            .unwrap_or(MaterialType::Silicate)
    }
}

impl SimOp for LithosphereUnifiedOp {
    fn name(&self) -> &str {
        "LithosphereUnifiedOp"
    }

    fn init_sim(&mut self, sim: &mut Simulation) {
        let noise = Perlin::new(self.seed);

        for (cell_id, column) in sim.cells.iter_mut() {
            let point = H3Utils::cell_to_3d_point(*cell_id, sim.planet.radius_km);
            let scaled_point = point * self.scale as f32;
            let point_array = scaled_point.to_array().map(|x| x as f64);
            let random_value = noise.get(point_array);
            let scaled_val = (random_value + 1.0) / 2.0; // normalize to 0.0–1.0

            let material = self.pick_material(scaled_val);

            // Get formation temperature from the surface asthenosphere layer
            let formation_temp_k = if let Some((surface_layer, _)) = column.asth_layers_t.first() {
                surface_layer.kelvin()
            } else {
                1673.15 // Default formation temperature for Silicate
            };

            let new_layer = AsthCellLithosphere::new(0.0, material, 0.0, formation_temp_k);
            column.lith_layers_t.push((new_layer.clone(), new_layer));
        }
    }

    fn update_sim(&mut self, sim: &mut Simulation) {
        for column in sim.cells.values_mut() {
            let surface_temp_k = column.asth_layers_t[0].0.kelvin();

            // Process lithosphere changes at the column level
            let energy_released = column.process_lithosphere_layer_change(
                surface_temp_k,
                sim.years_per_step,
                sim.lith_layer_height_km
            );

            // Add any released energy back to the top asthenosphere layer
            if energy_released > 0.0 {
                let (_, top_asth_layer) = &mut column.layer_mut(0);
                top_asth_layer.add_energy(energy_released);
            }
            // column.cleanup_empty_lithosphere_layers();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::{ASTHENOSPHERE_SURFACE_START_TEMP_K, EARTH_RADIUS_KM};
    use crate::planet::Planet;
    use crate::sim::simulation::{SimProps, Simulation};
    use approx::assert_abs_diff_eq;
    use h3o::Resolution;

    fn create_test_simulation() -> Simulation {
        Simulation::new(SimProps {
            name: "lithosphere_unified_test",
            planet: Planet {
                radius_km: EARTH_RADIUS_KM as f64,
                resolution: Resolution::One,
            },
            ops: vec![],
            res: Resolution::Two,
            layer_count: 4,
            asth_layer_height_km: 100.0,
            lith_layer_height_km: 50.0,
            sim_steps: 1,
            years_per_step: 1000,
            debug: false,
            alert_freq: 1,
            starting_surface_temp_k: ASTHENOSPHERE_SURFACE_START_TEMP_K,
        })
    }

    #[test]
    fn test_formation_at_low_temperature() {
        let mut sim = create_test_simulation();
        let mut op = LithosphereUnifiedOp::new(vec![(MaterialType::Silicate, 1.0)], 42, 0.1);

        // Initialize lithosphere
        op.init_sim(&mut sim);

        // Set low temperature for formation
        for column in sim.cells.values_mut() {
            column.asth_layers_t[0].0.set_temp_kelvin(1600.0); // Below silicate formation temp
        }

        // Record initial heights (should be 0 or very small initially)
        // Check the first (bottom) layer since that's where growth happens
        let initial_heights: Vec<f64> = sim
            .cells
            .values()
            .map(|column| {
                if column.lith_layers_t.is_empty() {
                    0.0
                } else {
                    column.lith_layers_t.first().unwrap().0.height_km
                }
            })
            .collect();

        // Run unified operator
        op.update_sim(&mut sim);

        // Commit changes from next to current
        for column in sim.cells.values_mut() {
            column.commit_next_layers();
        }

        // Heights should increase (formation occurred)
        // Check the first (bottom) layer since that's where growth happens
        for (column, &initial_height) in sim.cells.values().zip(initial_heights.iter()) {
            let final_height = if column.lith_layers_t.is_empty() {
                0.0
            } else {
                column.lith_layers_t.first().unwrap().0.height_km
            };

            // Calculate total lithosphere height
            let total_lith_height: f64 = column.lith_layers_t.iter()
                .map(|(current, _)| current.height_km)
                .sum();

            assert!(
                final_height > initial_height,
                "Lithosphere should form at low temperature. Initial: {}, Final: {}",
                initial_height,
                final_height
            );

            // Also check that total lithosphere height is reasonable
            assert!(
                total_lith_height > 0.5,
                "Total lithosphere height should be substantial after formation. Got: {} km",
                total_lith_height
            );
        }
    }

    #[test]
    fn test_melting_at_high_temperature() {
        let mut sim = create_test_simulation();
        let mut op = LithosphereUnifiedOp::new(vec![(MaterialType::Silicate, 1.0)], 42, 0.1);

        // Initialize
        op.init_sim(&mut sim);

        // Step 1: First create lithosphere by setting formation temperature
        for column in sim.cells.values_mut() {
            column.asth_layers_t[0].0.set_temp_kelvin(1600.0); // Below silicate formation temp (1750K)
        }

        // Run formation to create lithosphere
        op.update_sim(&mut sim);

        // Commit changes from next to current
        for column in sim.cells.values_mut() {
            column.commit_next_layers();
        }

        // Verify lithosphere was created
        let formation_heights: Vec<f64> = sim
            .cells
            .values()
            .map(|column| {
                if column.lith_layers_t.is_empty() {
                    0.0
                } else {
                    column.lith_layers_t.first().unwrap().0.height_km // Check current state of first (bottom) layer
                }
            })
            .collect();

        // Ensure we have some lithosphere to melt
        assert!(
            formation_heights.iter().any(|&h| h > 0.0),
            "Formation should have created lithosphere"
        );

        // Step 2: Now set very high temperature for melting
        for column in sim.cells.values_mut() {
            column.asth_layers_t[0].0.set_temp_kelvin(2500.0); // Well above melting point
        }

        // Record initial heights after formation
        let initial_heights: Vec<f64> = sim
            .cells
            .values()
            .map(|column| {
                if column.lith_layers_t.is_empty() {
                    0.0
                } else {
                    column.lith_layers_t.first().unwrap().0.height_km // Check current state of first (bottom) layer
                }
            })
            .collect();

        // Run melting
        op.update_sim(&mut sim);

        // Commit changes from next to current
        for column in sim.cells.values_mut() {
            column.commit_next_layers();
        }

        // Heights should decrease (melting occurred)
        let mut melting_occurred = false;
        for (column, &initial_height) in sim.cells.values().zip(initial_heights.iter()) {
            let final_height = if column.lith_layers_t.is_empty() {
                0.0
            } else {
                column.lith_layers_t.first().unwrap().0.height_km // Check current state of first (bottom) layer
            };

            if initial_height > 0.0 && final_height < initial_height {
                melting_occurred = true;
            }
        }

        assert!(
            melting_occurred,
            "Lithosphere should melt at high temperature. Formation heights: {:?}, Initial heights: {:?}",
            formation_heights,
            initial_heights
        );
    }

    #[test]
    fn test_equilibrium_behavior() {
        let mut sim = create_test_simulation();
        let mut op = LithosphereUnifiedOp::new(vec![(MaterialType::Silicate, 1.0)], 42, 0.1);

        // Initialize lithosphere
        op.init_sim(&mut sim);

        // Set temperature right at formation threshold
        for column in sim.cells.values_mut() {
            column.asth_layers_t[0].0.set_temp_kelvin(1873.15); // Exactly at silicate formation temp
        }

        // Record initial heights
        let initial_heights: Vec<f64> = sim
            .cells
            .values()
            .map(|column| column.lith_layers_t.last().unwrap().1.height_km)
            .collect();

        // Run unified operator
        op.update_sim(&mut sim);

        // Heights should remain approximately the same (equilibrium)
        for (column, &initial_height) in sim.cells.values().zip(initial_heights.iter()) {
            let final_height = column.lith_layers_t.last().unwrap().1.height_km;
            assert_abs_diff_eq!(final_height, initial_height, epsilon = 0.1);
        }
    }
}
