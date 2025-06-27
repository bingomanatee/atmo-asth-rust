use super::serialize_cell_indexes::{
    deserialize_cell_index, deserialize_neighbor_cells, serialize_cell_index,
    serialize_neighbor_cells,
};
use crate::material::MaterialType;
pub(crate) use crate::asth_cell::asth_cell_asth_layer::AsthCellAsthLayer;
use crate::asth_cell::energy_at_layer::energy_at_layer;
use crate::asth_cell::AsthCellLithosphere;
use crate::constants::{ASTHENOSPHERE_SURFACE_START_TEMP_K, EARTH_RADIUS_KM, SPECIFIC_HEAT_CAPACITY_MANTLE_J_PER_KG_K};
use crate::h3_utils::H3Utils;
use h3o::CellIndex;
use serde::{Deserialize, Serialize};

/// This is a class that represents a stack of the planet at a specific h3o location
/// layers projec downwards, layer[0] is at the top
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AsthCellColumn {
    pub energy_joules: f64,
    pub volume_km3: f64,
    pub asth_layers: Vec<AsthCellAsthLayer>,
    pub asth_layers_next: Vec<AsthCellAsthLayer>,
    pub lithospheres: Vec<AsthCellLithosphere>,
    pub lithospheres_next: Vec<AsthCellLithosphere>,
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
        // Calculate correct layer volume: area × layer_height_km
        let layer_volume = params.volume * params.layer_height_km;

        let cell_column = AsthCellColumn {
            energy_joules: 0.0,
            volume_km3: 0.0,
            neighbor_cell_ids: H3Utils::neighbors_for(params.cell_index),
            // Create layer 0 at surface temperature - projection will handle geothermal gradient for deeper layers
            asth_layers: vec![AsthCellAsthLayer::new_with_material(MaterialType::Silicate, params.surface_temp_k, layer_volume, 0)],
            cell_index: params.cell_index,
            layer_height_km: params.layer_height_km,
            layer_count: params.layer_count,
            asth_layers_next: vec![],
            planet_radius_km: params.planet_radius_km,
            lithospheres: vec![
                crate::asth_cell::asth_cell_lithosphere::AsthCellLithosphere::new(
                    0.0,
                    MaterialType::Silicate,
                    0.0,
                ),
            ],
            lithospheres_next: vec![
                crate::asth_cell::asth_cell_lithosphere::AsthCellLithosphere::new(
                    0.0,
                    MaterialType::Silicate,
                    0.0,
                ),
            ],
        };

        cell_column.project(
            params.layer_count,
            params.planet_radius_km,
            params.surface_temp_k,
        )
    }

    /// Get the total height of all lithosphere layers
    pub fn total_lithosphere_height(&self) -> f64 {
        self.lithospheres.iter().map(|layer| layer.height_km).sum()
    }

    /// Get the total height of all next lithosphere layers
    pub fn total_lithosphere_height_next(&self) -> f64 {
        self.lithospheres_next
            .iter()
            .map(|layer| layer.height_km)
            .sum()
    }

    /// Get the area of this cell column in km²
    /// Uses the cell's resolution and the provided planet radius
    pub fn area(&self) -> f64 {
        H3Utils::cell_area(self.cell_index.resolution(), self.planet_radius_km)
    }

    /// Get the bottom lithosphere layer, returning (current, next, profile) where next is mutable
    /// Panics if no lithosphere exists - all lithospheres should be created during initialization
    /// If next lithosphere is absent, clone the current lithosphere into next
    /// Returns the profile of the next lithosphere as the third argument
    /// Panics if the lithosphere profile cannot be found (game-ending error)
    pub fn lithosphere(
        &mut self,
        layer_index: usize
    ) -> (
        &AsthCellLithosphere,
        &mut AsthCellLithosphere,
        &'static crate::material::MaterialProfile,
    ) {
        // Ensure the current layer exists (should be created by project method)
        if layer_index >= self.lithospheres.len() {
            panic!(
                "Current lithosphere {} does not exist. All layers should be created during initialization. Current layers: {}",
                layer_index,
                self.asth_layers.len()
            );
        }

        // Ensure the next layers vector has this layer - clone from current if missing
        while self.asth_layers_next.len() <= layer_index {
            let current_layer = &self.asth_layers[self.asth_layers_next.len()];
            self.asth_layers_next.push(current_layer.clone());
        }

        // Return references to the layers
        (
            &self.lithospheres[layer_index],
            &mut self.lithospheres_next[layer_index],
            self.lithospheres[layer_index].profile()
        )
    }

    /// Create a lithosphere with the specified material
    /// Used when operations need to create new lithosphere
    pub fn create_lithosphere(
        &mut self,
        material: MaterialType,
        height_km: f64,
        volume_km3: f64,
    ) {
        let lithosphere = AsthCellLithosphere::new(height_km, material, volume_km3);

        // Add to both current and next
        self.lithospheres.push(lithosphere.clone());
        self.lithospheres_next.push(lithosphere);
    }



    /// Add a new lithosphere layer at the bottom (becomes new layer 0)
    /// All existing layers shift up in index
    pub fn add_bottom_lithosphere(&mut self, material: MaterialType, height_km: f64) {
        let area = self.area();
        let volume_km3 = area * height_km;
        let lithosphere = AsthCellLithosphere::new(height_km, material, volume_km3);

        // Insert at index 0 (bottom) - all other layers shift up
        self.lithospheres_next.insert(0, lithosphere);
    }



    /// Get a layer by index, returning (current, next) where next is mutable
    /// If next layer is absent, clone the current layer into next
    pub fn asth_layer(&mut self, layer_index: usize) -> (&AsthCellAsthLayer, &mut AsthCellAsthLayer) {
        // Ensure the current layer exists (should be created by project method)
        if layer_index >= self.asth_layers.len() {
            panic!(
                "Current layer {} does not exist. All layers should be created during initialization. Current layers: {}",
                layer_index,
                self.asth_layers.len()
            );
        }

        // Ensure the next layers vector has this layer - clone from current if missing
        while self.asth_layers_next.len() <= layer_index {
            let current_layer = &self.asth_layers[self.asth_layers_next.len()];
            self.asth_layers_next.push(current_layer.clone());
        }

        // Return references to the layers
        (
            &self.asth_layers[layer_index],
            &mut self.asth_layers_next[layer_index],
        )
    }

    pub fn commit_next_layers(&mut self) {
        self.asth_layers.clone_from_slice(&self.asth_layers_next);
        self.lithospheres = self.lithospheres_next.clone();
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
        let mut layers: Vec<AsthCellAsthLayer> = self.asth_layers.clone();
        let mut layers_next = self.asth_layers_next.clone();

        // iterate over all the absent _current_ cells
        for index in layers.len()..level_count {
            // Always use the correct volume calculation: area × layer_height_km
            let volume = self.default_volume();

            // Calculate temperature for this layer using geothermal gradient
            let depth_km = (index as f64 + 0.5) * self.layer_height_km;
            let layer_temp = surface_temp_k + crate::constants::GEOTHERMAL_GRADIENT_K_PER_KM * depth_km;

            // Create layer with target temperature and let EnergyMass calculate energy
            layers.push(AsthCellAsthLayer::new_with_material(MaterialType::Silicate, layer_temp, volume, index));
        }
        // Ensure layers_next has the correct size and copy layers
        layers_next.clear();
        layers_next.extend(layers.iter().cloned());

        AsthCellColumn {
            asth_layers: layers,
            asth_layers_next: layers_next,
            layer_count: level_count,
            ..self.clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::asth_cell::asth_cell_column::{AsthCellColumn, AsthCellParams};
    use crate::constants::{ASTHENOSPHERE_SURFACE_START_TEMP_K, DEFAULT_LAYER_HEIGHT_KM, SPECIFIC_HEAT_CAPACITY_MANTLE_J_PER_KG_K};
    use crate::temp_utils::volume_kelvin_to_joules;
    use approx::assert_abs_diff_eq;
    use h3o::CellIndex;

    #[test]
    fn creation() {
        let first = CellIndex::base_cells().next().unwrap();

        let volume: f64 = 1000.0;
        let energy = volume_kelvin_to_joules(volume, ASTHENOSPHERE_SURFACE_START_TEMP_K, SPECIFIC_HEAT_CAPACITY_MANTLE_J_PER_KG_K);

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
        assert_eq!(cell.asth_layers.iter().len(), LEVEL_COUNT);
        assert_eq!(cell.asth_layers_next.iter().len(), LEVEL_COUNT);
        assert_eq!(cell.asth_layers[0].volume_km3(), volume);
        assert_abs_diff_eq!(cell.asth_layers[0].energy_joules(), 7.08e21, epsilon = 4.0e20);
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
            let (current, next) = cell.asth_layer(0);
            assert_eq!(current.level, 0);
            assert_eq!(next.level, 0);

            // Modify the next layer
            next.set_energy_joules(2000.0);
        }
        assert_eq!(cell.asth_layers_next[0].energy_joules(), 2000.0);

        // Test accessing layer 1 (should exist from initialization)
        {
            let (current_1, next_1) = cell.asth_layer(1);
            assert_eq!(current_1.level, 1);
            assert_eq!(next_1.level, 1);

            // Energy should be calculated using energy_at_layer function
            assert!(current_1.energy_joules() > 0.0);
        }

        // Test that the layer method properly clones current to next when needed
        let initial_next_len = cell.asth_layers_next.len();

        // Access a layer that might not be in next yet
        {
            let (_, next_layer) = cell.asth_layer(1);
            next_layer.set_energy_joules(1500.0);
        }

        // Should have cloned current layers to next if needed
        assert!(cell.asth_layers_next.len() >= 2);
        assert_eq!(cell.asth_layers_next[1].energy_joules(), 1500.0);

        // Test accessing multiple layers separately
        {
            let (_, next_0) = cell.asth_layer(0);
            next_0.set_energy_joules(3000.0);
        }

        assert_eq!(cell.asth_layers_next[0].energy_joules(), 3000.0);
        assert_eq!(cell.asth_layers_next[1].energy_joules(), 1500.0);
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
            let layer_0_temp = cell.asth_layers[0].kelvin();
            println!("{}: Surface temp {:.2} K -> Layer 0 temp {:.2} K (diff: {:.2} K)",
                     description, surface_temp_k, layer_0_temp, layer_0_temp - surface_temp_k);

            // Check layer 1 (deeper layer)
            let layer_1_temp = cell.asth_layers[1].kelvin();
            println!("  Layer 1 temp: {:.2} K (diff from surface: {:.2} K)",
                     layer_1_temp, layer_1_temp - surface_temp_k);

            // Check layer 2 (deepest layer)
            let layer_2_temp = cell.asth_layers[2].kelvin();
            println!("  Layer 2 temp: {:.2} K (diff from surface: {:.2} K)",
                     layer_2_temp, layer_2_temp - surface_temp_k);

            // Verify temperature increases with depth (geothermal gradient)
            assert!(layer_1_temp > layer_0_temp, "Layer 1 should be hotter than layer 0");
            assert!(layer_2_temp > layer_1_temp, "Layer 2 should be hotter than layer 1");
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

                let actual_layer_temp = cell.asth_layers[0].kelvin();
                let diff = (actual_layer_temp - target_temp).abs();

                println!("{} (target {:.2} K): Surface {:.2} K -> Layer {:.2} K (diff: {:.2} K)",
                         description, target_temp, surface_temp, actual_layer_temp, diff);

                // If we're within 10K, that's close enough for testing
                if diff < 10.0 {
                    println!("  *** GOOD MATCH: Use surface temp {:.2} K for target {:.2} K ***",
                             surface_temp, target_temp);
                }
            }
            println!();
        }
    }
}
