use crate::material_composite::MaterialCompositeType;

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
    /// Create 4x scaled experiment configuration for scientific accuracy
    pub fn basic_experiment_state() -> Self {
        Self {
            // Use Silicate as the primary material type (mantle material)
            material_type: MaterialCompositeType::Silicate,

            // Diffusion parameters (4x scaled constants)
            conductivity_factor: 12.0,
            pressure_baseline: 4.0,
            max_change_rate: 0.08,

            // Boundary conditions (4x enhanced for early Earth conditions)
            foundry_temperature_k: 7200.0,
            surface_temperature_k: 300.0,
        }
    }

    /// Get thermal conductivity for a specific material phase
    /// Uses the material system instead of hardcoded values
    pub fn get_thermal_conductivity(&self, phase: crate::material_composite::MaterialPhase) -> f64 {
        use crate::material_composite::get_profile_fast;
        get_profile_fast(&self.material_type, &phase).thermal_conductivity_w_m_k
    }

    /// Get melting point for the experiment material
    pub fn get_melting_point(&self) -> f64 {
        use crate::material_composite::get_material_core;
        get_material_core(&self.material_type).melting_point_avg_k
    }
}