use crate::sim::sim_op::{SimOp, SimOpHandle};
use crate::sim::Simulation;
use crate::material::MaterialType;
use crate::asth_cell::AsthCellLithosphere;
use crate::h3_utils::H3Utils;
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
            return self.materials.first().map(|(mat, _)| *mat).unwrap_or(MaterialType::Silicate);
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
        self.materials.first().map(|(mat, _)| *mat).unwrap_or(MaterialType::Silicate)
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
            column.lithospheres_next.push(AsthCellLithosphere::new(0.0, material, 0.0));
        }
    }

    fn update_sim(&mut self, sim: &mut Simulation) {
        for column in sim.cells.values_mut() {
            // Get the surface layer temperature (asthenosphere temperature)
            let surface_temp_k = if let Some(surface_layer) = column.layers.first() {
                surface_layer.kelvin()
            } else {
                continue; // Skip if no surface layer
            };

            // Calculate area once to avoid borrowing issues
            let area = column.area();

            // Process each lithosphere layer
            for lithosphere in column.lithospheres_next.iter_mut() {
                let profile = lithosphere.profile();
                
                if surface_temp_k <= profile.max_lith_formation_temp_kv {
                    // FORMATION: Temperature is cool enough for lithosphere growth
                    if lithosphere.height_km < profile.max_lith_height_km {
                        // Calculate growth rate based on temperature
                        let growth_km_per_year = if surface_temp_k <= profile.peak_lith_growth_temp_kv {
                            // At or below peak growth temperature - maximum growth
                            profile.max_lith_growth_km_per_year
                        } else {
                            // Between peak and formation temperature - reduced growth
                            let temp_range = profile.max_lith_formation_temp_kv - profile.peak_lith_growth_temp_kv;
                            let temp_above_peak = surface_temp_k - profile.peak_lith_growth_temp_kv;
                            let growth_factor = 1.0 - (temp_above_peak / temp_range);
                            profile.max_lith_growth_km_per_year * growth_factor.max(0.0)
                        };

                        let growth_km_per_step = growth_km_per_year * sim.years_per_step as f64;

                        if growth_km_per_step > 0.0 {
                            // Calculate new height but cap at maximum allowed height
                            let new_height = lithosphere.height_km + growth_km_per_step;
                            let capped_height = new_height.min(profile.max_lith_height_km);
                            
                            lithosphere.height_km = capped_height;
                            let new_volume = area * lithosphere.height_km;
                            lithosphere.set_volume_km3(new_volume);
                        }
                    }
                } else {
                    // MELTING: Temperature is too hot - melt lithosphere
                    if lithosphere.height_km > 0.0 {
                        // Calculate melting rate using the existing melt_from_below_km_per_year function
                        let melt_rate_km_per_year = lithosphere.melt_from_below_km_per_year(surface_temp_k);
                        
                        if melt_rate_km_per_year > 0.0 {
                            // Convert to melting per simulation step
                            let melt_rate_km_per_step = melt_rate_km_per_year * sim.years_per_step as f64;
                            
                            // Apply melting (reduce lithosphere height)
                            let original_height = lithosphere.height_km;
                            lithosphere.height_km = (lithosphere.height_km - melt_rate_km_per_step).max(0.0);
                            
                            // Update volume to match new height
                            if lithosphere.height_km > 0.0 {
                                let new_volume = area * lithosphere.height_km;
                                lithosphere.set_volume_km3(new_volume);
                            } else {
                                // Lithosphere completely melted
                                lithosphere.set_volume_km3(0.0);
                            }

                            // Add the melted energy back to the surface layer
                            // The melted lithosphere becomes hot magma, adding energy
                            if let Some(surface_layer) = column.layers_next.first_mut() {
                                let melted_height = original_height - lithosphere.height_km;
                                if melted_height > 0.0 {
                                    // Calculate energy from melted lithosphere
                                    // Use the asthenosphere temperature as the energy to add
                                    let melted_volume_km3 = area * melted_height;
                                    let melted_mass_kg = melted_volume_km3 * 1e9 * lithosphere.density();
                                    let energy_to_add = melted_mass_kg * lithosphere.specific_heat() * surface_temp_k;
                                    
                                    surface_layer.add_energy(energy_to_add);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::{ASTHENOSPHERE_SURFACE_START_TEMP_K, EARTH_RADIUS_KM};
    use crate::planet::Planet;
    use crate::sim::{SimProps, Simulation};
    use h3o::Resolution;
    use approx::assert_abs_diff_eq;

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
            layer_height_km: 10.0,
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
        let mut op = LithosphereUnifiedOp::new(
            vec![(MaterialType::Silicate, 1.0)],
            42,
            0.1,
        );

        // Initialize lithosphere
        op.init_sim(&mut sim);

        // Set low temperature for formation
        for column in sim.cells.values_mut() {
            column.layers[0].set_temp_kelvin(1600.0); // Below silicate formation temp
        }

        // Record initial heights
        let initial_heights: Vec<f64> = sim.cells.values()
            .map(|column| column.lithospheres_next.last().unwrap().height_km)
            .collect();

        // Run unified operator
        op.update_sim(&mut sim);

        // Heights should increase (formation occurred)
        for (column, &initial_height) in sim.cells.values().zip(initial_heights.iter()) {
            let final_height = column.lithospheres_next.last().unwrap().height_km;
            assert!(final_height > initial_height, 
                    "Lithosphere should form at low temperature. Initial: {}, Final: {}", 
                    initial_height, final_height);
        }
    }

    #[test]
    fn test_melting_at_high_temperature() {
        let mut sim = create_test_simulation();
        let mut op = LithosphereUnifiedOp::new(
            vec![(MaterialType::Silicate, 1.0)],
            42,
            0.1,
        );

        // Initialize and add some lithosphere
        op.init_sim(&mut sim);
        for column in sim.cells.values_mut() {
            column.lithospheres_next.last_mut().unwrap().height_km = 10.0;
            let area = column.area();
            let volume = area * 10.0;
            column.lithospheres_next.last_mut().unwrap().set_volume_km3(volume);
        }

        // Set high temperature for melting
        for column in sim.cells.values_mut() {
            column.layers[0].set_temp_kelvin(2500.0); // Well above silicate formation temp
        }

        // Record initial heights
        let initial_heights: Vec<f64> = sim.cells.values()
            .map(|column| column.lithospheres_next.last().unwrap().height_km)
            .collect();

        // Run unified operator
        op.update_sim(&mut sim);

        // Heights should decrease (melting occurred)
        for (column, &initial_height) in sim.cells.values().zip(initial_heights.iter()) {
            let final_height = column.lithospheres_next.last().unwrap().height_km;
            assert!(final_height < initial_height, 
                    "Lithosphere should melt at high temperature. Initial: {}, Final: {}", 
                    initial_height, final_height);
        }
    }

    #[test]
    fn test_equilibrium_behavior() {
        let mut sim = create_test_simulation();
        let mut op = LithosphereUnifiedOp::new(
            vec![(MaterialType::Silicate, 1.0)],
            42,
            0.1,
        );

        // Initialize lithosphere
        op.init_sim(&mut sim);

        // Set temperature right at formation threshold
        for column in sim.cells.values_mut() {
            column.layers[0].set_temp_kelvin(1873.15); // Exactly at silicate formation temp
        }

        // Record initial heights
        let initial_heights: Vec<f64> = sim.cells.values()
            .map(|column| column.lithospheres_next.last().unwrap().height_km)
            .collect();

        // Run unified operator
        op.update_sim(&mut sim);

        // Heights should remain approximately the same (equilibrium)
        for (column, &initial_height) in sim.cells.values().zip(initial_heights.iter()) {
            let final_height = column.lithospheres_next.last().unwrap().height_km;
            assert_abs_diff_eq!(final_height, initial_height, epsilon = 0.1);
        }
    }
}
