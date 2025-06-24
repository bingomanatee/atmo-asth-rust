use crate::sim::Simulation;

pub trait SimOp {
    fn update_sim(&mut self, sim: &mut Simulation);
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