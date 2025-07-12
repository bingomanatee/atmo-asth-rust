/// Global H3-based thermal simulation system with unified layers
///
/// This module implements a global thermal simulation using H3 hexagonal grid cells,
/// where each cell contains unified thermal layers that can naturally transition
/// between atmospheric, liquid, and solid phases based on temperature and pressure.

pub mod thermal_layer;
pub mod sim_cell;
pub mod heat_plume;
pub mod thermal_expansion;

pub use thermal_layer::ThermalLayer;
pub use sim_cell::SimCell;
pub use heat_plume::{HeatPlume, CellPlumeCollection, PlumeStats};
pub use thermal_expansion::{ThermalExpansionCalculator, ThermalExpansionStats};
