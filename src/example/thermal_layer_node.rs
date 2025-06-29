use super::experiment_state::ExperimentState;
use crate::energy_mass_composite::{EnergyMassComposite, EnergyMassParams, StandardEnergyMassComposite};
use crate::material_composite::{get_profile_fast, MaterialCompositeType, MaterialPhase};
use crate::temp_utils::joules_volume_to_kelvin;

/// Parameters for creating ThermalLayerNode
pub struct ThermalLayerNodeParams {
    pub material_type: MaterialCompositeType,
    pub energy_joules: f64,
    pub volume_km3: f64,
    pub depth_km: f64,
    pub height_km: f64,
}

/// Thermal node with enhanced state tracking
#[derive(Clone, Debug)]
pub struct ThermalLayerNode {
    pub energy_mass: StandardEnergyMassComposite,
    pub thermal_state: i32,
    pub depth_km: f64,
    pub height_km: f64,

    /// Thermal history for analysis
    pub initial_temperature: f64,
    pub max_temperature: f64,
    // this is a debugging tracker for the highest temperature reached not a physically limiting constant
    pub min_temperature: f64,
    // same
    /// Outgassing tracking
    pub total_outgassed_mass: f64,
    pub outgassing_rate: f64,
}

impl ThermalLayerNode {
    pub fn new(params: ThermalLayerNodeParams) -> Self {
        let profile = get_profile_fast(&params.material_type, &MaterialPhase::Liquid);

        let temperature_k = joules_volume_to_kelvin(
            params.energy_joules,
            params.volume_km3,
            profile.specific_heat_capacity_j_per_kg_k
        );
        let energy_mass = StandardEnergyMassComposite::new_with_material_state_and_energy(
            EnergyMassParams {
                material_type: params.material_type,
                initial_phase: MaterialPhase::Solid,
                energy_joules: params.energy_joules,
                volume_km3: params.volume_km3,
                height_km: params.height_km,
            }
        );

        Self {
            energy_mass,
            thermal_state: 100, // Start as solid
            depth_km: params.depth_km,
            height_km: params.height_km,
            initial_temperature: temperature_k,
            max_temperature: temperature_k,
            min_temperature: temperature_k,
            total_outgassed_mass: 0.0,
            outgassing_rate: 0.0,
        }
    }

    pub fn temp_kelvin(&self) -> f64 {
        self.energy_mass.kelvin()
    }

    pub fn volume_km3(&self) -> f64 {
        self.energy_mass.volume()
    }

    pub fn mass_kg(&self) -> f64 {
        self.energy_mass.mass_kg()
    }

    pub fn height_km(&self) -> f64 {
        self.height_km
    }

    pub fn depth_km(&self) -> f64 {
        self.depth_km
    }

    /// Get thermal conductivity based on current material state and type
    pub fn thermal_conductivity(&self) -> f64 {
        self.energy_mass
            .material_composite_profile()
            .thermal_conductivity_w_m_k
    }

    pub fn add_energy(&mut self, energy_j: f64) {
        self.energy_mass.add_joules(energy_j);
        self.log_extent();
    }

    pub fn remove_energy(&mut self, energy_j: f64) {
        if energy_j > 0.0 && energy_j <= self.energy_mass.energy() {
            self.energy_mass.remove_joules(energy_j);
            self.log_extent();
        }
    }

    /// Calculate outgassing based on temperature
    pub fn calculate_outgassing(&mut self, config: &ExperimentState, years: f64) -> f64 {
        0.0
    }
}

/// Extent logging for debugging

impl ThermalLayerNode {
    fn log_extent(&mut self) {
        let temp = self.temp_kelvin();
        self.max_temperature = self.max_temperature.max(temp);
        self.min_temperature = self.min_temperature.min(temp);
    }
}

/// Material and thermal State

impl ThermalLayerNode {
    /// Format thermal state for display
    pub fn format_thermal_state(&self) -> String {
        let temp = self.thermal_state;
        match self.material_state() {
            MaterialPhase::Solid => {
                format!("<{}> 🗻 Solid/Lithosphere", temp)
            }
            MaterialPhase::Liquid => {
                format!("<{}> 💧 Liquid/Magma", temp.abs())
            }
            MaterialPhase::Gas => {
                format!("<{}> ☁️ Gas/Atmosphere", temp.abs())
            }
        }
    }

    pub fn material_state(&self) -> MaterialPhase {
        match self.thermal_state {
            std::i32::MIN..=-66 => MaterialPhase::Gas, // anything ≤ −66
            -65..=65 => MaterialPhase::Liquid,         // −65 through 65
            66..=std::i32::MAX => MaterialPhase::Solid, // 66 through max
        }
    }
}
