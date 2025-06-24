mod energy_at_layer;
mod serialize_cell_indexes;

use crate::asth_cell::energy_at_layer::energy_at_layer;
use crate::h3_utils::H3Utils;
use h3o::CellIndex;
use serde::{Deserialize, Serialize};
use serialize_cell_indexes::{
    deserialize_cell_index, deserialize_neighbor_cells, serialize_cell_index,
    serialize_neighbor_cells,
};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AsthCellLayer {
    pub energy_joules: f64,
    pub volume_km3: f64,
    pub level: usize,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AsthCellLithosphere {
    pub height_km :f64,
    pub volume_km3: f64
}

impl AsthCellLithosphere {
    fn new(height: f64) -> AsthCellLithosphere {
       AsthCellLithosphere {
           height_km: height, // we assume height is zero 
           // but if its not we will have to compensate in the future
           volume_km3: 0.0,
       }
    }
}

/// This is a class that represents a stack of the planet at a specific h3o location
/// layers projec downwards, layer[0] is at the top
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AsthCellColumn {
    pub energy_joules: f64,
    pub volume_km3: f64,
    pub layers: Vec<AsthCellLayer>,
    pub layers_next: Vec<AsthCellLayer>,
    pub lithosphere: AsthCellLithosphere ,
    pub lithosphere_next: AsthCellLithosphere,
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
            lithosphere: AsthCellLithosphere::new(0.0),
            lithosphere_next: AsthCellLithosphere::new(0.0),
        };

        cell_column.project(params.layer_count, params.planet_radius)
    }
    
    pub fn commit_next_layers(&mut self) {
        self.layers.clone_from_slice(&self.layers_next);
        self.lithosphere = self.lithosphere_next.clone();
    }
    
    fn project(&self, level_count: usize, planet_radius: f64) -> AsthCellColumn {
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
            let energy = energy_at_layer(index, self.layer_height_km.into(), volume);
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
    use crate::temp_utils::volume_kelvin_to_joules;
    use crate::asth_cell::{AsthCellColumn, AsthCellParams};
    use approx::assert_abs_diff_eq;
    use h3o::CellIndex;
    use crate::constants::{ASTHENOSPHERE_SURFACE_START_TEMP_K, DEFAULT_LAYER_HEIGHT_KM};

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
        });

        assert_eq!(cell.neighbor_cell_ids.iter().len(), 6);
        assert_eq!(cell.layers.iter().len(), LEVEL_COUNT);
        assert_eq!(cell.layers_next.iter().len(), LEVEL_COUNT);
        assert_eq!(cell.layers[0].volume_km3, volume);
        assert_abs_diff_eq!(cell.layers[0].energy_joules, 6.4e21, epsilon = 4.0e20);
    }
}
