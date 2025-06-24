use serde::{Deserialize, Serialize};
use crate::asth_cell::LithosphereType;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AsthCellLithosphere {
    pub material: LithosphereType,
    pub height_km: f64,
    pub volume_km3: f64,
}

impl AsthCellLithosphere {
    pub fn new(height: f64, material: LithosphereType) -> AsthCellLithosphere {
        AsthCellLithosphere {
            material,
            height_km: height,
            volume_km3: 0.0,
        }
    }
}