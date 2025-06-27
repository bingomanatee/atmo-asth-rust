use super::serialize_cell_indexes::{
    deserialize_cell_index, deserialize_neighbor_cells, serialize_cell_index,
    serialize_neighbor_cells,
};
use crate::asth_cell::AsthCellLithosphere;
pub(crate) use crate::asth_cell::asth_cell_asth_layer::AsthCellAsthLayer;
use crate::asth_cell::energy_at_layer::energy_at_layer;
use crate::constants::{
    ASTHENOSPHERE_SURFACE_START_TEMP_K, EARTH_RADIUS_KM, SPECIFIC_HEAT_CAPACITY_MANTLE_J_PER_KG_K,
};
use crate::energy_mass::{EnergyMass, StandardEnergyMass};
use crate::h3_utils::H3Utils;
use crate::material::MaterialType;
use h3o::CellIndex;
use serde::{Deserialize, Serialize};

/// This is a class that represents a stack of the planet at a specific h3o location
/// layers projec downwards, layer[0] is at the top
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AsthCellColumn {
    pub energy_joules: f64,
    pub volume_km3: f64,
    pub asth_layers_t: Vec<(AsthCellAsthLayer, AsthCellAsthLayer)>,
    pub lith_layers_t: Vec<(AsthCellLithosphere, AsthCellLithosphere)>,
    pub layer_height_km: f64,
    pub planet_radius_km: f64,
    pub layer_count: usize,

    #[serde(
        serialize_with = "serialize_neighbor_cells",
        deserialize_with = "deserialize_neighbor_cells"
    )]
    pub neighbor_cell_ids: Vec<CellIndex>,

    #[serde(
        serialize_with = "serialize_cell_index",
        deserialize_with = "deserialize_cell_index"
    )]
    pub cell_index: CellIndex,
}

pub struct AsthCellParams {
    pub cell_index: CellIndex,
    pub volume: f64,
    pub energy: f64,
    pub layer_count: usize,
    pub layer_height_km: f64,
    pub planet_radius_km: f64,
    pub surface_temp_k: f64,
}

impl AsthCellColumn {
    pub fn new(params: AsthCellParams) -> AsthCellColumn {
        // Use the volume as provided (should already be area × layer_height_km)
        let layer_volume = params.volume;

        let initial_asth_layer = AsthCellAsthLayer::new_with_material(
            MaterialType::Silicate,
            params.surface_temp_k,
            layer_volume,
            0,
        );
        let initial_lith_layer = crate::asth_cell::asth_cell_lithosphere::AsthCellLithosphere::new(
            0.0,
            MaterialType::Silicate,
            0.0,
            params.surface_temp_k, // Use surface temperature as formation temperature
        );

        let cell_column = AsthCellColumn {
            energy_joules: 0.0,
            volume_km3: 0.0,
            neighbor_cell_ids: H3Utils::neighbors_for(params.cell_index),
            // Create layer 0 at surface temperature - projection will handle geothermal gradient for deeper layers
            asth_layers_t: vec![(initial_asth_layer.clone(), initial_asth_layer)],
            cell_index: params.cell_index,
            layer_height_km: params.layer_height_km,
            layer_count: params.layer_count,
            planet_radius_km: params.planet_radius_km,
            lith_layers_t: vec![(initial_lith_layer.clone(), initial_lith_layer)],
        };

        cell_column.project(
            params.layer_count,
            params.planet_radius_km,
            params.surface_temp_k,
        )
    }

    /// Get the total height of all current lithosphere layers
    pub fn total_lithosphere_height(&self) -> f64 {
        self.lith_layers_t.iter().map(|(current, _)| current.height_km).sum()
    }

    /// Get the total height of all next lithosphere layers
    pub fn total_lithosphere_height_next(&self) -> f64 {
        self.lith_layers_t.iter().map(|(_, next)| next.height_km).sum()
    }

    /// Get the area of this cell column in km²
    /// Uses the cell's resolution and the provided planet radius
    pub fn area(&self) -> f64 {
        H3Utils::cell_area(self.cell_index.resolution(), self.planet_radius_km)
    }



    /// Get mutable reference to lithosphere layer tuple (current, next)
    /// If accessing index 0 and no lithosphere exists, creates an empty layer with the same material type as the top asthenosphere layer
    pub fn lithosphere_mut(&mut self, layer_index: usize) -> &mut (AsthCellLithosphere, AsthCellLithosphere) {
        // Special case: if accessing index 0 and no lithosphere layers exist, create an empty layer
        if layer_index == 0 && self.lith_layers_t.is_empty() {
            // Get material type from top asthenosphere layer
            let material_type = if !self.asth_layers_t.is_empty() {
                self.asth_layers_t[0].0.energy_mass.material_type()
            } else {
                // Fallback to Silicate if no asthenosphere layers exist
                MaterialType::Silicate
            };

            // Create empty lithosphere layer tuple with surface temperature
            let formation_temp_k = if let Some((surface_layer, _)) = self.asth_layers_t.first() {
                surface_layer.kelvin()
            } else {
                1673.15 // Default formation temperature for Silicate
            };
            let empty_layer = AsthCellLithosphere::new(0.0, material_type, 0.0, formation_temp_k);
            self.lith_layers_t.push((empty_layer.clone(), empty_layer));
        }

        // Ensure the layer exists
        if layer_index >= self.lith_layers_t.len() {
            panic!(
                "Lithosphere layer {} does not exist. Current layers: {}",
                layer_index,
                self.lith_layers_t.len()
            );
        }

        &mut self.lith_layers_t[layer_index]
    }

    /// Get immutable reference to lithosphere layer tuple (current, next)
    pub fn lithosphere(&self, layer_index: usize) -> &(AsthCellLithosphere, AsthCellLithosphere) {
        if layer_index >= self.lith_layers_t.len() {
            panic!(
                "Lithosphere layer {} does not exist. Current layers: {}",
                layer_index,
                self.lith_layers_t.len()
            );
        }

        &self.lith_layers_t[layer_index]
    }

    /// Get mutable reference to asthenosphere layer tuple (current, next)
    /// Panics if no asthenosphere layer exists
    pub fn layer_mut(&mut self, layer_index: usize) -> &mut (AsthCellAsthLayer, AsthCellAsthLayer) {
        // Ensure the layer exists
        if layer_index >= self.asth_layers_t.len() {
            panic!(
                "Asthenosphere layer {} does not exist. Current layers: {}",
                layer_index,
                self.asth_layers_t.len()
            );
        }

        &mut self.asth_layers_t[layer_index]
    }

    /// Get immutable reference to asthenosphere layer tuple (current, next)
    pub fn layer(&self, layer_index: usize) -> &(AsthCellAsthLayer, AsthCellAsthLayer) {
        if layer_index >= self.asth_layers_t.len() {
            panic!(
                "Asthenosphere layer {} does not exist. Current layers: {}",
                layer_index,
                self.asth_layers_t.len()
            );
        }

        &self.asth_layers_t[layer_index]
    }

    /// Create a lithosphere with the specified material
    /// Used when operations need to create new lithosphere
    pub fn create_lithosphere(&mut self, material: MaterialType, height_km: f64, volume_km3: f64) {
        // Get formation temperature from the surface asthenosphere layer
        let formation_temp_k = if let Some((surface_layer, _)) = self.asth_layers_t.first() {
            surface_layer.kelvin()
        } else {
            1673.15 // Default formation temperature for Silicate
        };

        let lithosphere = AsthCellLithosphere::new(height_km, material, volume_km3, formation_temp_k);

        // Add as tuple (current, next) - both start the same
        self.lith_layers_t.push((lithosphere.clone(), lithosphere));
    }

    /// Add a new lithosphere layer at the bottom (becomes new layer 0)
    /// All existing layers shift up in index
    pub fn add_bottom_lithosphere(&mut self, material: MaterialType, height_km: f64) {
        let area = self.area();
        let volume_km3 = area * height_km;

        // Get formation temperature from the surface asthenosphere layer
        let formation_temp_k = if let Some((surface_layer, _)) = self.asth_layers_t.first() {
            surface_layer.kelvin()
        } else {
            1673.15 // Default formation temperature
        };

        let lithosphere = AsthCellLithosphere::new(height_km, material, volume_km3, formation_temp_k);

        // Insert at index 0 (bottom) - all other layers shift up
        self.lith_layers_t.insert(0, (lithosphere.clone(), lithosphere));
    }

    /// Remove empty lithosphere layers (height = 0 or volume = 0)
    /// This prevents accumulation of empty layers that can cause ordering issues
    pub fn cleanup_empty_lithosphere_layers(&mut self) {
        // Remove empty layers from lith_layers_t
        self.lith_layers_t.retain(|(current, next)| {
            current.height_km > 0.001 && current.volume_km3() > 0.001 &&
            next.height_km > 0.001 && next.volume_km3() > 0.001
        });

        // Ensure we always have at least one lithosphere layer
        if self.lith_layers_t.is_empty() {
            // Get formation temperature from the surface asthenosphere layer
            let formation_temp_k = if let Some((surface_layer, _)) = self.asth_layers_t.first() {
                surface_layer.kelvin()
            } else {
                1673.15 // Default formation temperature for Silicate
            };
            let empty_layer = AsthCellLithosphere::new(0.0, crate::material::MaterialType::Silicate, 0.0, formation_temp_k);
            self.lith_layers_t.push((empty_layer.clone(), empty_layer));
        }
    }

    /// Insert a lithosphere layer with the given height
    /// - Only allows empty layers when the stack is completely empty (initial state)
    /// - Otherwise, only allows non-empty layers (growth creates material, melting shrinks existing)
    pub fn insert_layer_height(&mut self, material_type: MaterialType, height_km: f64, _max_layer_height_km: f64) -> bool {
        let is_empty = height_km <= 0.001;

        // Get formation temperature from the surface asthenosphere layer
        let formation_temp_k = if let Some((surface_layer, _)) = self.asth_layers_t.first() {
            surface_layer.kelvin()
        } else {
            1673.15 // Default formation temperature for Silicate
        };

        // Only allow empty layers when there are no layers at all (initial state)
        if is_empty {
            if self.lith_layers_t.is_empty() {
                let new_layer = AsthCellLithosphere::new(height_km, material_type, 0.0, formation_temp_k);
                self.lith_layers_t.insert(0, (new_layer.clone(), new_layer));
                return true;
            } else {
                // Don't insert empty layers when there are existing layers
                return false;
            }
        }

        // Non-empty layers are always acceptable (growth/formation creates material)
        let new_layer = AsthCellLithosphere::new(height_km, material_type, 0.0, formation_temp_k);
        self.lith_layers_t.insert(0, (new_layer.clone(), new_layer));
        true
    }

    /// Insert a lithosphere layer with the given volume and energy, following acceptable patterns
    pub fn insert_layer_volume_energy(&mut self, material_type: MaterialType, volume_km3: f64, energy_j: f64, max_layer_height_km: f64) -> bool {
        // Calculate height from volume
        let height_km = volume_km3 / self.area();

        // Use the height-based logic to determine if insertion is acceptable
        if !self.insert_layer_height(material_type, height_km, max_layer_height_km) {
            return false;
        }

        // Replace the layer that was just inserted with one that has correct volume and energy
        if let Some((current, next)) = self.lith_layers_t.first_mut() {
            let energy_mass = StandardEnergyMass::new_with_material_energy(material_type, energy_j, volume_km3);
            let new_layer = AsthCellLithosphere {
                height_km,
                energy_mass,
            };
            *current = new_layer.clone();
            *next = new_layer;
        }

        true
    }



    /// Get the material type for lithosphere formation (from top asthenosphere layer)
    pub fn formation_material_type(&self) -> MaterialType {
        if !self.asth_layers_t.is_empty() {
            self.asth_layers_t[0].0.energy_mass.material_type()
        } else {
            MaterialType::Silicate // Fallback
        }
    }

    /// Process lithosphere formation/melting at the layer level
    /// Returns the energy released from melting (if any)
    pub fn process_lithosphere_layer_change(&mut self, surface_temp_k: f64, years_per_step: u32, max_layer_height_km: f64) -> f64 {
        let material_type = self.formation_material_type();
        let profile = crate::material::get_profile(material_type).unwrap();

        let current_total_height = self.total_lithosphere_height();
        let mut energy_released = 0.0;

        if surface_temp_k <= profile.max_lith_formation_temp_kv {
            // FORMATION: Temperature below formation threshold
            if current_total_height < profile.max_lith_height_km {
                // Use the existing layer growth_per_year method
                let growth_rate_km_per_year = if let Some((bottom_layer, _)) = self.lith_layers_t.first() {
                    bottom_layer.growth_per_year(surface_temp_k)
                } else {
                    // Fallback to profile calculation if no layers exist
                    profile.growth_at_kelvin(surface_temp_k)
                };

                let growth_km = growth_rate_km_per_year * years_per_step as f64;

                if growth_km > 0.0 {
                    self.grow_lithosphere_stack(growth_km, max_layer_height_km, surface_temp_k);
                }
            }
        } else {
            // MELTING: Temperature above formation threshold
            if current_total_height > 0.0 {
                // Use the existing layer melt_from_below_km_per_year method
                let melt_rate_km_per_year = if let Some((bottom_layer, _)) = self.lith_layers_t.first() {
                    bottom_layer.melt_from_below_km_per_year(surface_temp_k)
                } else {
                    0.0
                };

                let melt_km = melt_rate_km_per_year * years_per_step as f64;

                if melt_km > 0.0 {
                    energy_released = self.melt_lithosphere_stack(melt_km);
                }
            }
        }

        energy_released
    }

    /// Grow the lithosphere stack by the specified total height
    /// Manages layer creation and distribution from high index to low index
    pub fn grow_lithosphere_stack(&mut self, growth_km: f64, max_layer_height_km: f64, formation_temp_k: f64) {
        let material_type = self.formation_material_type();
        let area = self.area();
        let mut remaining_growth = growth_km;

        // Start from the bottom layer (index 0) and work upward
        let mut layer_index = 0;

        while remaining_growth > 0.001 && layer_index < 20 { // Safety limit
            // Ensure layer exists
            if layer_index >= self.lith_layers_t.len() {
                // Create new layer at the top (highest index) with formation temperature
                let new_layer = AsthCellLithosphere::new(0.0, material_type, 0.0, formation_temp_k);
                self.lith_layers_t.push((new_layer.clone(), new_layer));
            }

            let (_, next_layer) = &mut self.lith_layers_t[layer_index];

            // Use the existing layer grow() method which handles excess properly
            let excess_height = next_layer.grow(remaining_growth, area, max_layer_height_km, formation_temp_k);

            // Calculate how much growth was actually applied to this layer
            let growth_applied = remaining_growth - excess_height;
            remaining_growth = excess_height;

            // If this layer is now full and we still have growth remaining, move to next layer
            if excess_height > 0.001 {
                layer_index += 1;
            } else {
                // All growth was absorbed by this layer
                break;
            }
        }
    }

    /// Melt the lithosphere stack by the specified total height
    /// Removes layers from front of array (high index to low index) and returns energy released
    pub fn melt_lithosphere_stack(&mut self, melt_km: f64) -> f64 {
        let mut remaining_melt = melt_km;
        let mut total_energy_released = 0.0;
        let area = self.area(); // Calculate area once to avoid borrowing issues

        // Start from the bottom layer (index 0) and work upward
        while remaining_melt > 0.001 && !self.lith_layers_t.is_empty() {
            let (_, next_layer) = &mut self.lith_layers_t[0]; // Always work on bottom layer

            if next_layer.height_km <= 0.001 {
                // Remove empty layer and continue
                self.lith_layers_t.remove(0);
                continue;
            }

            // Use the existing layer melt() method
            let original_height = next_layer.height_km;
            let original_energy = next_layer.energy_joules();

            let actual_melted = next_layer.melt(remaining_melt);

            if actual_melted > 0.0 {
                // Calculate energy released proportional to melted fraction
                let melted_fraction = actual_melted / original_height;
                let energy_released = original_energy * melted_fraction;
                total_energy_released += energy_released;

                // Remove the melted energy from the layer
                next_layer.remove_energy(energy_released);

                // Update volume to match new height
                let new_volume = area * next_layer.height_km;
                next_layer.set_volume_km3(new_volume);

                remaining_melt -= actual_melted;
            }

            // If layer is now empty, remove it
            if next_layer.height_km <= 0.001 {
                self.lith_layers_t.remove(0);
            } else {
                // Layer still has material but we couldn't melt more - stop
                break;
            }
        }

        // Ensure we always have at least one lithosphere layer (even if empty)
        if self.lith_layers_t.is_empty() {
            let material_type = self.formation_material_type();
            // Get formation temperature from the surface asthenosphere layer
            let formation_temp_k = if let Some((surface_layer, _)) = self.asth_layers_t.first() {
                surface_layer.kelvin()
            } else {
                1673.15 // Default formation temperature for Silicate
            };
            let empty_layer = AsthCellLithosphere::new(0.0, material_type, 0.0, formation_temp_k);
            self.lith_layers_t.push((empty_layer.clone(), empty_layer));
        }

        total_energy_released
    }



    pub fn commit_next_layers(&mut self) {
        // Copy next state to current state for all layers
        for (current, next) in self.asth_layers_t.iter_mut() {
            *current = next.clone();
        }
        for (current, next) in self.lith_layers_t.iter_mut() {
            *current = next.clone();
        }
    }

    pub fn default_volume(&self) -> f64 {
        H3Utils::cell_area(self.cell_index.resolution(), self.planet_radius_km)
            * self.layer_height_km
    }
    fn project(
        &self,
        level_count: usize,
        planet_radius: f64,
        surface_temp_k: f64,
    ) -> AsthCellColumn {
        let mut layers_t: Vec<(AsthCellAsthLayer, AsthCellAsthLayer)> = self.asth_layers_t.clone();

        // iterate over all the absent _current_ cells
        for index in layers_t.len()..level_count {
            // Always use the correct volume calculation: area × layer_height_km
            let volume = self.default_volume();

            // Calculate temperature for this layer using geothermal gradient
            let depth_km = (index as f64 + 0.5) * self.layer_height_km;
            let layer_temp =
                surface_temp_k + crate::constants::GEOTHERMAL_GRADIENT_K_PER_KM * depth_km;

            // Create layer with target temperature and let EnergyMass calculate energy
            let layer = AsthCellAsthLayer::new_with_material(
                MaterialType::Silicate,
                layer_temp,
                volume,
                index,
            );

            // Add as tuple (current, next) - both start the same
            layers_t.push((layer.clone(), layer));
        }

        AsthCellColumn {
            asth_layers_t: layers_t,
            layer_count: level_count,
            ..self.clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::asth_cell::asth_cell_column::{AsthCellColumn, AsthCellParams};
    use crate::constants::{
        ASTHENOSPHERE_SURFACE_START_TEMP_K, DEFAULT_LAYER_HEIGHT_KM,
        SPECIFIC_HEAT_CAPACITY_MANTLE_J_PER_KG_K,
    };
    use crate::temp_utils::volume_kelvin_to_joules;
    use approx::assert_abs_diff_eq;
    use h3o::CellIndex;

    #[test]
    fn creation() {
        let first = CellIndex::base_cells().next().unwrap();

        let volume: f64 = 1000.0;
        let energy = volume_kelvin_to_joules(
            volume,
            ASTHENOSPHERE_SURFACE_START_TEMP_K,
            SPECIFIC_HEAT_CAPACITY_MANTLE_J_PER_KG_K,
        );

        const LEVEL_COUNT: usize = 4;

        let cell = AsthCellColumn::new(AsthCellParams {
            cell_index: first,
            volume: volume,
            energy: energy,
            layer_count: LEVEL_COUNT,
            layer_height_km: DEFAULT_LAYER_HEIGHT_KM,
            planet_radius_km: 6371.0,
            surface_temp_k: ASTHENOSPHERE_SURFACE_START_TEMP_K,
        });

        assert_eq!(cell.neighbor_cell_ids.iter().len(), 6);
        assert_eq!(cell.asth_layers_t.iter().len(), LEVEL_COUNT);
        assert_eq!(cell.asth_layers_t[0].0.volume_km3(), volume);
        assert_abs_diff_eq!(
            cell.asth_layers_t[0].0.energy_joules(),
            energy,
            epsilon = 1e10
        );
    }

    #[test]
    fn test_layer_method() {
        use crate::constants::EARTH_RADIUS_KM;
        use h3o::CellIndex;

        let cell_index = CellIndex::base_cells().next().unwrap();
        let mut cell = AsthCellColumn::new(AsthCellParams {
            cell_index,
            energy: 1000.0,
            volume: 100.0,
            layer_height_km: 10.0,
            layer_count: 2,
            planet_radius_km: EARTH_RADIUS_KM as f64,
            surface_temp_k: ASTHENOSPHERE_SURFACE_START_TEMP_K,
        });

        // Test accessing existing layer
        {
            let (current, next) = &mut cell.layer_mut(0);
            assert_eq!(current.level, 0);
            assert_eq!(next.level, 0);

            // Modify the next layer
            next.set_energy_joules(2000.0);
        }
        assert_eq!(cell.asth_layers_t[0].1.energy_joules(), 2000.0);

        // Test accessing layer 1 (should exist from initialization)
        {
            let (current_1, next_1) = &mut cell.layer_mut(1);
            assert_eq!(current_1.level, 1);
            assert_eq!(next_1.level, 1);

            // Energy should be calculated using energy_at_layer function
            assert!(current_1.energy_joules() > 0.0);
        }

        // Test that the layer method properly clones current to next when needed
        let initial_next_len = cell.asth_layers_t.len();

        // Access a layer that might not be in next yet
        {
            let (_, next_layer) = &mut cell.layer_mut(1);
            next_layer.set_energy_joules(1500.0);
        }

        // Should have cloned current layers to next if needed
        assert!(cell.asth_layers_t.len() >= 2);
        assert_eq!(cell.asth_layers_t[1].1.energy_joules(), 1500.0);

        // Test accessing multiple layers separately
        {
            let (_, next_0) = &mut cell.layer_mut(0);
            next_0.set_energy_joules(3000.0);
        }

        assert_abs_diff_eq!(cell.asth_layers_t[0].1.energy_joules(), 3000.0, epsilon = 1e-6);
        assert_abs_diff_eq!(cell.asth_layers_t[1].1.energy_joules(), 1500.0, epsilon = 1e-6);
    }

    #[test]
    fn test_surface_temperature_to_layer_kelvin() {
        use crate::constants::EARTH_RADIUS_KM;
        use h3o::CellIndex;

        let cell_index = CellIndex::base_cells().next().unwrap();

        // Test various surface temperatures
        let test_cases = vec![
            (1000.0, "Low temperature"),
            (1200.0, "Medium-low temperature"),
            (1400.0, "Medium temperature"),
            (1600.0, "Medium-high temperature"),
            (1800.0, "High temperature"),
            (2000.0, "Very high temperature"),
        ];

        for (surface_temp_k, description) in test_cases {
            let cell = AsthCellColumn::new(AsthCellParams {
                cell_index,
                energy: 1000.0,
                volume: 100.0,
                layer_height_km: 10.0,
                layer_count: 3,
                planet_radius_km: EARTH_RADIUS_KM as f64,
                surface_temp_k,
            });

            // Check layer 0 (surface layer)
            let layer_0_temp = cell.asth_layers_t[0].0.kelvin();
            println!(
                "{}: Surface temp {:.2} K -> Layer 0 temp {:.2} K (diff: {:.2} K)",
                description,
                surface_temp_k,
                layer_0_temp,
                layer_0_temp - surface_temp_k
            );

            // Check layer 1 (deeper layer)
            let layer_1_temp = cell.asth_layers_t[1].0.kelvin();
            println!(
                "  Layer 1 temp: {:.2} K (diff from surface: {:.2} K)",
                layer_1_temp,
                layer_1_temp - surface_temp_k
            );

            // Check layer 2 (deepest layer)
            let layer_2_temp = cell.asth_layers_t[2].0.kelvin();
            println!(
                "  Layer 2 temp: {:.2} K (diff from surface: {:.2} K)",
                layer_2_temp,
                layer_2_temp - surface_temp_k
            );

            // Verify temperature increases with depth (geothermal gradient)
            assert!(
                layer_1_temp > layer_0_temp,
                "Layer 1 should be hotter than layer 0"
            );
            assert!(
                layer_2_temp > layer_1_temp,
                "Layer 2 should be hotter than layer 1"
            );
        }
    }

    #[test]
    fn test_specific_temperature_targets() {
        use crate::constants::EARTH_RADIUS_KM;
        use h3o::CellIndex;

        let cell_index = CellIndex::base_cells().next().unwrap();

        // Test to find what surface temperature gives us specific layer temperatures
        let target_layer_temps = vec![
            (1673.15, "Silicate peak growth temp"),
            (1773.15, "Silicate halfway temp"),
            (1873.15, "Silicate formation temp"),
        ];

        for (target_temp, description) in target_layer_temps {
            // Try different surface temperatures to see which gets closest to target
            for surface_offset in [-600.0, -500.0, -400.0, -300.0, -200.0] {
                let surface_temp = target_temp + surface_offset;

                let cell = AsthCellColumn::new(AsthCellParams {
                    cell_index,
                    energy: 1000.0,
                    volume: 100.0,
                    layer_height_km: 10.0,
                    layer_count: 2,
                    planet_radius_km: EARTH_RADIUS_KM as f64,
                    surface_temp_k: surface_temp,
                });

                let actual_layer_temp = cell.asth_layers_t[0].0.kelvin();
                let diff = (actual_layer_temp - target_temp).abs();

                println!(
                    "{} (target {:.2} K): Surface {:.2} K -> Layer {:.2} K (diff: {:.2} K)",
                    description, target_temp, surface_temp, actual_layer_temp, diff
                );

                // If we're within 10K, that's close enough for testing
                if diff < 10.0 {
                    println!(
                        "  *** GOOD MATCH: Use surface temp {:.2} K for target {:.2} K ***",
                        surface_temp, target_temp
                    );
                }
            }
            println!();
        }
    }
}
