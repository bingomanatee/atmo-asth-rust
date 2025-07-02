use crate::material_composite::MaterialCompositeType;

/// Specification parameters for creating ExperimentState
#[derive(Clone, Debug)]
pub struct ExperimentSpecs {
    pub material_type: MaterialCompositeType,
    pub conductivity_factor: f64,
    pub pressure_baseline: f64,
    pub max_change_rate: f64,
    pub surface_temperature_k: f64,
    pub foundry_temperature_k: f64,
}

/// Configuration parameters for thermal diffusion experiments
/// Simplified to use the new multi-state material system
#[derive(Clone, Debug)]
pub struct ExperimentState {
    // Primary material type for the experiment
    pub material_type: MaterialCompositeType,

    // ------------ Diffusion parameters
    pub conductivity_factor: f64,
    pub pressure_baseline: f64,
    pub max_change_rate: f64, // max change per cell
    // from 0 (no cell)
    // to 1 (can change from 2x energy to 0x energy

    // ---------- thermal initialization

    // start heat at cell [0]
    pub surface_temperature_k: f64,
    // start heat at cell  [len - 1]
    pub foundry_temperature_k: f64,
}

impl ExperimentState {
    /// Create experiment configuration with specified parameters
    pub fn new_with_specs(specs: ExperimentSpecs) -> Self {
        Self {
            material_type: specs.material_type,
            conductivity_factor: specs.conductivity_factor,
            pressure_baseline: specs.pressure_baseline,
            max_change_rate: specs.max_change_rate,
            surface_temperature_k: specs.surface_temperature_k,
            foundry_temperature_k: specs.foundry_temperature_k,
        }
    }

    /// Create basic experiment configuration for 1km² experiments
    /// These values are locked for the 1km experiment to prevent config drift
    pub fn basic_experiment_state() -> Self {
        Self::new_with_specs(ExperimentSpecs {
            // Use Silicate as the primary material type (mantle material)
            material_type: MaterialCompositeType::Silicate,

            // Diffusion parameters (4x scaled constants) - locked for 1km² experiment
            conductivity_factor: 12.0,
            pressure_baseline: 4.0,
            max_change_rate: 0.08,

            // Boundary conditions (4x enhanced for early Earth conditions) - locked for 1km² experiment
            foundry_temperature_k: 5000.0,
            surface_temperature_k: 300.0,
        })
    }

    /// Get thermal conductivity for a specific material phase
    /// Uses the material system instead of hardcoded values
    pub fn get_thermal_conductivity(&self, phase: crate::material_composite::MaterialPhase) -> f64 {
        use crate::material_composite::get_profile_fast;
        get_profile_fast(&self.material_type, &phase).thermal_conductivity_w_m_k
    }

    /// Get melting point for the experiment material
    pub fn get_melting_point(&self) -> f64 {
        use crate::material_composite::get_melting_point_k;
        get_melting_point_k(&self.material_type)
    }
}