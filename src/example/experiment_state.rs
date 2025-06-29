/// Configuration parameters for thermal diffusion experiments
#[derive(Clone, Debug)]
pub struct ExperimentState {
    // Thermal conductivity (material-specific values)
    pub solid_conductivity: f64,
    pub liquid_conductivity: f64,
    pub transition_conductivity: f64,

    // Diffusion parameters
    pub conductivity_factor: f64,
    pub pressure_baseline: f64,
    pub max_change_rate: f64,        // Maximum energy change rate per step

    // Boundary conditions
    pub foundry_temperature_k: f64,
    pub surface_temperature_k: f64,
}

impl ExperimentState {
    /// Create 4x scaled experiment configuration for scientific accuracy
    pub fn basic_experiment_state() -> Self {
        Self {
            // Thermal conductivity (4x scaled constants)
            solid_conductivity: 10.0,
            liquid_conductivity: 12.8,
            transition_conductivity: 11.2,

            // Diffusion parameters (4x scaled constants)
            conductivity_factor: 12.0,
            pressure_baseline: 4.0,
            max_change_rate: 0.08,        

            // Boundary conditions (4x enhanced for early Earth conditions)
            foundry_temperature_k: 7200.0,   
            surface_temperature_k: 300.0,
        }
    }
}