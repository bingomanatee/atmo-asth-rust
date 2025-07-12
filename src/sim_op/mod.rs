// Current operations for global thermal simulation
pub mod atmospheric_generation_op;
pub mod heat_plume_op;
pub mod radiance_op;
pub mod space_radiation_op;
pub mod surface_energy_init_op;
pub mod temperature_reporting_op;
pub mod thermal_conduction_op;

// Re-export the main operations for easier access
pub use atmospheric_generation_op::AtmosphericGenerationOp;
pub use heat_plume_op::HeatPlumeOp;
pub use radiance_op::{RadianceOp, RadianceOpParams};
pub use space_radiation_op::{SpaceRadiationOp, SpaceRadiationOpParams, apply_space_radiation};
pub use surface_energy_init_op::{SurfaceEnergyInitOp, SurfaceEnergyInitParams};
pub use temperature_reporting_op::TemperatureReportingOp;
pub use thermal_conduction_op::{ThermalConductionOp, ThermalConductionParams};

use crate::sim::simulation::Simulation;
use std::any::Any;

pub trait SimOp: Send + Sync {
    /// The name of this operator (for identification and lookup)
    fn name(&self) -> &str;
    
    /// Allow downcasting to concrete types for specialized access
    fn as_any(&self) -> &dyn Any;

    /// Called once at the beginning of the simulation
    fn init_sim(&mut self, _sim: &mut Simulation) {
        // Default implementation does nothing
    }

    /// Called every simulation step
    fn update_sim(&mut self, _sim: &mut Simulation) {
        // Default implementation does nothing
    }

    /// Called once at the end of the simulation
    fn after_sim(&mut self, _sim: &mut Simulation) {
        // Default implementation does nothing
    }
}

pub struct SimOpHandle {
    pub op: Box<dyn SimOp>,
}

impl SimOpHandle {
    /// Create a new SimOpHandle with the given operation
    pub fn new(op: Box<dyn SimOp>) -> Self {
        SimOpHandle { op }
    }

    /// Execute the operation on the simulation
    pub fn execute(&mut self, sim: &mut Simulation) {
        self.op.update_sim(sim);
    }
}
