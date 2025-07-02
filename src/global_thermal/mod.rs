/// Global H3-based thermal simulation system with unified layers
///
/// This module implements a global thermal simulation using H3 hexagonal grid cells,
/// where each cell contains unified thermal layers that can naturally transition
/// between atmospheric, liquid, and solid phases based on temperature and pressure.

pub mod thermal_layer;
pub mod global_h3_cell;

pub use thermal_layer::ThermalLayer;
pub use global_h3_cell::GlobalH3Cell;
