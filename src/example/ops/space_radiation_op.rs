use crate::sim::simulation::Simulation;
use crate::sim::sim_op::{SpaceRadiationOp as CoreSpaceRadiationOp, SpaceRadiationParams, SimOp};

/// Parameters for space radiation operation
#[derive(Debug, Clone)]
pub struct SpaceRadiationOpParams {
    /// Surface emissivity (0.0 to 1.0, typically 0.95 for rocky surfaces)
    pub emissivity: f64,
    /// Space temperature in Kelvin (typically ~2.7K cosmic background)
    pub space_temperature_k: f64,
    /// Density constant for opacity calculation (density * DENSITY_CONSTANT / height)
    pub density_constant: f64,
    /// Enable detailed reporting
    pub enable_reporting: bool,
}

impl Default for SpaceRadiationOpParams {
    fn default() -> Self {
        Self {
            emissivity: 0.95,
            space_temperature_k: 2.7, // Cosmic microwave background
            density_constant: 0.001, // Adjust this to control atmospheric opacity
            enable_reporting: false,
        }
    }
}

impl SpaceRadiationOpParams {
    /// Create parameters with custom emissivity
    pub fn with_emissivity(emissivity: f64) -> Self {
        Self {
            emissivity,
            ..Default::default()
        }
    }
    
    /// Create parameters with reporting enabled
    pub fn with_reporting() -> Self {
        Self {
            enable_reporting: true,
            ..Default::default()
        }
    }
}

/// Space radiation operation for global thermal simulation
/// Radiates heat from surface layers to space using Stefan-Boltzmann law
pub struct SpaceRadiationOp {
    params: SpaceRadiationOpParams,
    core_op: CoreSpaceRadiationOp,
    step_count: usize,
}

impl SpaceRadiationOp {
    pub fn new(params: SpaceRadiationOpParams) -> Self {
        let core_params = SpaceRadiationParams {
            emissivity: params.emissivity,
            space_temperature_k: params.space_temperature_k,
            density_constant: params.density_constant,
        };
        
        Self {
            params,
            core_op: CoreSpaceRadiationOp::new(core_params),
            step_count: 0,
        }
    }
    
    pub fn new_default() -> Self {
        Self::new(SpaceRadiationOpParams::default())
    }
    
    /// Apply space radiation to the simulation
    pub fn apply(&mut self, sim: &mut Simulation, time_years: f64) {
        self.step_count += 1;
        
        // Apply radiation to all cells
        self.core_op.apply(&mut sim.cells, time_years);
        
        // Report if enabled
        if self.params.enable_reporting {
            self.report_radiation_results(time_years);
        }
    }
    
    /// Report radiation results
    fn report_radiation_results(&self, time_years: f64) {
        let total_radiated = self.core_op.total_radiated_energy();
        let cells_processed = self.core_op.cells_processed();
        let avg_per_cell = self.core_op.average_radiated_per_cell();
        
        println!("🌌 Space Radiation Step {}: {:.2e} J total, {} cells, {:.2e} J/cell avg ({:.1} years)", 
                 self.step_count, total_radiated, cells_processed, avg_per_cell, time_years);
    }
    
    /// Get total energy radiated in the last operation
    pub fn total_radiated_energy(&self) -> f64 {
        self.core_op.total_radiated_energy()
    }
    
    /// Get number of cells processed in the last operation
    pub fn cells_processed(&self) -> usize {
        self.core_op.cells_processed()
    }
    
    /// Get average energy radiated per cell
    pub fn average_radiated_per_cell(&self) -> f64 {
        self.core_op.average_radiated_per_cell()
    }
}

impl SimOp for SpaceRadiationOp {
    fn name(&self) -> &str {
        "SpaceRadiationOp"
    }

    fn init_sim(&mut self, sim: &mut Simulation) {
        if self.params.enable_reporting {
            println!("🌌 Space radiation initialized (Stefan-Boltzmann cooling)");
            println!("   - Emissivity: {:.2}", self.params.emissivity);
            println!("   - Space temperature: {:.1}K", self.params.space_temperature_k);
            println!("   - Density constant: {}", self.params.density_constant);
        }
    }

    fn update_sim(&mut self, sim: &mut Simulation) {
        let time_years = sim.years_per_step as f64;
        self.apply(sim, time_years);
    }
}
