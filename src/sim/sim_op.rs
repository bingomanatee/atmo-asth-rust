mod sim_op_cooling;
mod sim_op_radiance;
mod sim_op_lithosphere;

use crate::sim::Simulation;

pub trait SimOp {
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