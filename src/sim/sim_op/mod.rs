// Deprecated operations moved to deprecated/ subfolder
// These operations expect the old AsthCellColumn structure
// mod sim_op_atmosphere;
// mod sim_op_cooling;
// mod sim_op_core_radiance;
// mod sim_op_lithosphere_unified;
// pub mod sim_op_csv_writer;
// mod sim_op_progress_reporter;
// mod sim_op_thermal_diffusion;

// pub use sim_op_atmosphere::AtmosphereOp;
// pub use sim_op_cooling::CoolingOp;
// pub use sim_op_core_radiance::CoreRadianceOp;
// pub use sim_op_lithosphere_unified::LithosphereUnifiedOp;
// pub use sim_op_csv_writer::CsvWriterOp;
// pub use sim_op_progress_reporter::ProgressReporterOp;
// pub use sim_op_thermal_diffusion::ThermalDiffusionOp;

// Current operations for global thermal simulation
pub mod atmospheric_generation_op;
pub mod heat_redistribution_op;
pub mod pressure_adjustment_op;
pub mod space_radiation_op;
pub mod surface_energy_init_op;
pub mod temperature_reporting_op;

// Re-export the main operations for easier access
pub use atmospheric_generation_op::AtmosphericGenerationOp;
pub use heat_redistribution_op::HeatRedistributionOp;
pub use pressure_adjustment_op::PressureAdjustmentOp;
pub use space_radiation_op::{SpaceRadiationOp, SpaceRadiationOpParams, apply_space_radiation};
pub use surface_energy_init_op::{SurfaceEnergyInitOp, SurfaceEnergyInitParams};
pub use temperature_reporting_op::TemperatureReportingOp;

use crate::sim::simulation::Simulation;

pub trait SimOp {
    /// The name of this operator (for identification and lookup)
    fn name(&self) -> &str;

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
