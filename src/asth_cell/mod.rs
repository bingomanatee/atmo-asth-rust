mod energy_at_layer;
mod serialize_cell_indexes;
mod asth_cell_lithosphere;

use crate::asth_cell::energy_at_layer::energy_at_layer;
use crate::h3_utils::H3Utils;
use h3o::CellIndex;
use serde::{Deserialize, Serialize};
use serialize_cell_indexes::{
    deserialize_cell_index, deserialize_neighbor_cells, serialize_cell_index,
    serialize_neighbor_cells,
};
// Re-export LithosphereType to make it available from this module
pub use crate::lithosphere::LithosphereType;

// Re-export AsthCellLithosphere to make it available from this module
pub use crate::asth_cell::asth_cell_lithosphere::AsthCellLithosphere;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AsthCellLayer {
    pub energy_joules: f64,
    pub volume_km3: f64,
    pub level: usize,
}

/// This is a class that represents a stack of the planet at a specific h3o location
/// layers projec downwards, layer[0] is at the top
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AsthCellColumn {
    pub energy_joules: f64,
    pub volume_km3: f64,
    pub layers: Vec<AsthCellLayer>,
    pub layers_next: Vec<AsthCellLayer>,
    pub lithosphere: Vec<crate::asth_cell::asth_cell_lithosphere::AsthCellLithosphere>,
    pub lithosphere_next: Vec<crate::asth_cell::asth_cell_lithosphere::AsthCellLithosphere>,
    pub layer_height_km: f64,
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
    pub planet_radius: f64,
    pub surface_temp_k: f64,
}

impl AsthCellColumn {
    pub(crate) fn new(params: AsthCellParams) -> AsthCellColumn {
        let cell_column = AsthCellColumn {
            energy_joules: 0.0,
            volume_km3: 0.0,
            neighbor_cell_ids: H3Utils::neighbors_for(params.cell_index),
            layers: vec![AsthCellLayer {
                energy_joules: params.energy,
                volume_km3: params.volume,
                level: 0,
            }],
            cell_index: params.cell_index,
            layer_height_km: params.layer_height_km,
            layer_count: params.layer_count,
            layers_next: vec![],
            lithosphere: vec![crate::asth_cell::asth_cell_lithosphere::AsthCellLithosphere::new(0.0, LithosphereType::Silicate)],
            lithosphere_next: vec![crate::asth_cell::asth_cell_lithosphere::AsthCellLithosphere::new(0.0, LithosphereType::Silicate)],
        };

        cell_column.project(params.layer_count, params.planet_radius, params.surface_temp_k)
    }

    /// Get the total height of all lithosphere layers
    pub fn total_lithosphere_height(&self) -> f64 {
        self.lithosphere.iter().map(|layer| layer.height_km).sum()
    }

    /// Get the total height of all next lithosphere layers
    pub fn total_lithosphere_height_next(&self) -> f64 {
        self.lithosphere_next.iter().map(|layer| layer.height_km).sum()
    }

    /// Get a layer by index, returning (current, next) where next is mutable
    /// If next layer is absent, clone the current layer into next
    pub fn layer(&mut self, layer_index: usize) -> (&AsthCellLayer, &mut AsthCellLayer) {
        // Ensure the current layer exists (should be created by project method)
        if layer_index >= self.layers.len() {
            panic!("Current layer {} does not exist. All layers should be created during initialization. Current layers: {}",
                   layer_index, self.layers.len());
        }

        // Ensure the next layers vector has this layer - clone from current if missing
        while self.layers_next.len() <= layer_index {
            let current_layer = &self.layers[self.layers_next.len()];
            self.layers_next.push(current_layer.clone());
        }

        // Return references to the layers
        (&self.layers[layer_index], &mut self.layers_next[layer_index])
    }

    pub fn commit_next_layers(&mut self) {
        self.layers.clone_from_slice(&self.layers_next);
        self.lithosphere = self.lithosphere_next.clone();
    }

    fn project(&self, level_count: usize, planet_radius: f64, surface_temp_k: f64) -> AsthCellColumn {
        let mut layers: Vec<AsthCellLayer> = self.layers.clone();
        let mut layers_next = self.layers_next.clone();
        let mut volume: f64 = 0.0;

        // iterate over all the absent cells
        for index in layers.len()..level_count {
            if volume == 0.0 {
                volume = if !layers.is_empty() {
                    layers[0].volume_km3
                } else {
                    H3Utils::cell_area(self.cell_index.resolution(), planet_radius)
                };
            }
            let energy = energy_at_layer(index, self.layer_height_km.into(), volume, surface_temp_k);
            layers.push(AsthCellLayer {
                energy_joules: energy,
                volume_km3: volume,
                level: index,
            });
        }
        // Ensure layers_next has the correct size and copy layers
        layers_next.clear();
        layers_next.extend(layers.iter().cloned());

        AsthCellColumn {
            layers,
            layers_next,
            layer_count: level_count,
            ..self.clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::asth_cell::{AsthCellColumn, AsthCellParams};
    use crate::constants::{ASTHENOSPHERE_SURFACE_START_TEMP_K, DEFAULT_LAYER_HEIGHT_KM};
    use crate::temp_utils::volume_kelvin_to_joules;
    use approx::assert_abs_diff_eq;
    use h3o::CellIndex;

    #[test]
    fn creation() {
        let first = CellIndex::base_cells().next().unwrap();

        let volume: f64 = 1000.0;
        let energy = volume_kelvin_to_joules(volume, ASTHENOSPHERE_SURFACE_START_TEMP_K);

        const LEVEL_COUNT: usize = 4;

        let cell = AsthCellColumn::new(AsthCellParams {
            cell_index: first,
            volume: volume,
            energy: energy,
            layer_count: LEVEL_COUNT,
            layer_height_km: DEFAULT_LAYER_HEIGHT_KM,
            planet_radius: 6371.0,
            surface_temp_k: ASTHENOSPHERE_SURFACE_START_TEMP_K
        });

        assert_eq!(cell.neighbor_cell_ids.iter().len(), 6);
        assert_eq!(cell.layers.iter().len(), LEVEL_COUNT);
        assert_eq!(cell.layers_next.iter().len(), LEVEL_COUNT);
        assert_eq!(cell.layers[0].volume_km3, volume);
        assert_abs_diff_eq!(cell.layers[0].energy_joules, 7.08e21, epsilon = 4.0e20);
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
            planet_radius: EARTH_RADIUS_KM as f64,
            surface_temp_k: ASTHENOSPHERE_SURFACE_START_TEMP_K
        });

        // Test accessing existing layer
        {
            let (current, next) = cell.layer(0);
            assert_eq!(current.level, 0);
            assert_eq!(next.level, 0);

            // Modify the next layer
            next.energy_joules = 2000.0;
        }
        assert_eq!(cell.layers_next[0].energy_joules, 2000.0);

        // Test accessing layer 1 (should exist from initialization)
        {
            let (current_1, next_1) = cell.layer(1);
            assert_eq!(current_1.level, 1);
            assert_eq!(next_1.level, 1);

            // Energy should be calculated using energy_at_layer function
            assert!(current_1.energy_joules > 0.0);
        }

        // Test that the layer method properly clones current to next when needed
        let initial_next_len = cell.layers_next.len();

        // Access a layer that might not be in next yet
        {
            let (_, next_layer) = cell.layer(1);
            next_layer.energy_joules = 1500.0;
        }

        // Should have cloned current layers to next if needed
        assert!(cell.layers_next.len() >= 2);
        assert_eq!(cell.layers_next[1].energy_joules, 1500.0);

        // Test accessing multiple layers separately
        {
            let (_, next_0) = cell.layer(0);
            next_0.energy_joules = 3000.0;
        }

        assert_eq!(cell.layers_next[0].energy_joules, 3000.0);
        assert_eq!(cell.layers_next[1].energy_joules, 1500.0);
    }
}
