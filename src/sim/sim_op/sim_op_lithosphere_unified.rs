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
            column
                .lithospheres_next
                .push(AsthCellLithosphere::new(0.0, material, 0.0));
        }
    }

    fn update_sim(&mut self, sim: &mut Simulation) {
        for column in sim.cells.values_mut() {
            let (surface_layer, _) = column.asth_layer(0);
            let surface_temp_k = surface_layer.kelvin();
            let area = column.area();

            let current_total_height = column.total_lithosphere_height_next();
            let (_, bottom_layer, profile) = column.lithosphere(0);

            if surface_temp_k <= profile.max_lith_formation_temp_kv {
                if current_total_height < profile.max_lith_height_km {
                    // Calculate growth rate based on temperature
                    let growth_height =
                        bottom_layer.growth_per_year(surface_temp_k) * sim.years_per_step as f64;
                    if (growth_height > 0.0) {
                        let excess_mass =
                            bottom_layer.grow(growth_height, area, sim.lith_layer_height_km);
                        if (excess_mass > 0.0) {
                            let mt = bottom_layer.material_type();
                            column.add_bottom_lithosphere(mt, excess_mass / area);
                        }
                    }
                }
            } else if (surface_temp_k > profile.melting_point_min_k) {
                // MELTING: Temperature is too hot - melt lithosphere from bottom layer only
                // We need to handle this without multiple mutable borrows
                if bottom_layer.height_km > 0.0 {
                    let energy = bottom_layer.process_melting(surface_temp_k, sim.years_per_step);
                    if (energy > 0.0) {
                        let (_, top_asth_layer) = column.asth_layer(0);
                        top_asth_layer.add_energy(energy);
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
            column.asth_layers[0].set_temp_kelvin(1600.0); // Below silicate formation temp
        }

        // Record initial heights
        let initial_heights: Vec<f64> = sim
            .cells
            .values()
            .map(|column| column.lithospheres_next.last().unwrap().height_km)
            .collect();

        // Run unified operator
        op.update_sim(&mut sim);

        // Heights should increase (formation occurred)
        for (column, &initial_height) in sim.cells.values().zip(initial_heights.iter()) {
            let final_height = column.lithospheres_next.last().unwrap().height_km;
            assert!(
                final_height > initial_height,
                "Lithosphere should form at low temperature. Initial: {}, Final: {}",
                initial_height,
                final_height
            );
        }
    }

    #[test]
    fn test_melting_at_high_temperature() {
        let mut sim = create_test_simulation();
        let mut op = LithosphereUnifiedOp::new(vec![(MaterialType::Silicate, 1.0)], 42, 0.1);

        // Initialize and add some lithosphere
        op.init_sim(&mut sim);
        for column in sim.cells.values_mut() {
            column.lithospheres_next.last_mut().unwrap().height_km = 10.0;
            let area = column.area();
            let volume = area * 10.0;
            column
                .lithospheres_next
                .last_mut()
                .unwrap()
                .set_volume_km3(volume);
        }

        // Set high temperature for melting
        for column in sim.cells.values_mut() {
            column.asth_layers[0].set_temp_kelvin(2500.0); // Well above silicate formation temp
        }

        // Record initial heights
        let initial_heights: Vec<f64> = sim
            .cells
            .values()
            .map(|column| column.lithospheres_next.last().unwrap().height_km)
            .collect();

        // Run unified operator
        op.update_sim(&mut sim);

        // Heights should decrease (melting occurred)
        for (column, &initial_height) in sim.cells.values().zip(initial_heights.iter()) {
            let final_height = column.lithospheres_next.last().unwrap().height_km;
            assert!(
                final_height < initial_height,
                "Lithosphere should melt at high temperature. Initial: {}, Final: {}",
                initial_height,
                final_height
            );
        }
    }

    #[test]
    fn test_equilibrium_behavior() {
        let mut sim = create_test_simulation();
        let mut op = LithosphereUnifiedOp::new(vec![(MaterialType::Silicate, 1.0)], 42, 0.1);

        // Initialize lithosphere
        op.init_sim(&mut sim);

        // Set temperature right at formation threshold
        for column in sim.cells.values_mut() {
            column.asth_layers[0].set_temp_kelvin(1873.15); // Exactly at silicate formation temp
        }

        // Record initial heights
        let initial_heights: Vec<f64> = sim
            .cells
            .values()
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
