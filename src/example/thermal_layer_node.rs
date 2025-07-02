use super::experiment_state::ExperimentState;
use crate::energy_mass_composite::{EnergyMassComposite, EnergyMassParams, StandardEnergyMassComposite};
use crate::material_composite::{get_profile_fast, MaterialCompositeType, MaterialPhase, MaterialStateProfile};
use crate::temp_utils::joules_volume_to_kelvin;

/// Parameters for creating ThermalLayerNode
pub struct ThermalLayerNodeParams {
    pub material_type: MaterialCompositeType,
    pub energy_joules: f64,
    pub volume_km3: f64,
    pub depth_km: f64,
    pub height_km: f64,
}

/// Parameters for creating ThermalLayerNode with direct temperature setting
pub struct ThermalLayerNodeTempParams {
    pub material_type: MaterialCompositeType,
    pub temperature_k: f64,
    pub volume_km3: f64,
    pub depth_km: f64,
    pub height_km: f64,
}

/// Thermal node with enhanced state tracking
#[derive(Clone, Debug)]
pub struct ThermalLayerNode {
    energy_mass: StandardEnergyMassComposite, // Private - access through trait methods
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
                pressure_gpa: 0.0, // Default surface pressure
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

    /// Create a new ThermalLayerNode with direct temperature setting (simplified)
    /// This avoids the energy conversion round-trip and sets temperature directly
    pub fn new_with_temperature(params: ThermalLayerNodeTempParams) -> Self {
        // Create energy mass with minimal energy, then set temperature directly
        let mut energy_mass = StandardEnergyMassComposite::new_with_material_state_and_energy(
            EnergyMassParams {
                material_type: params.material_type,
                initial_phase: MaterialPhase::Solid,
                energy_joules: 1.0, // Minimal energy, will be overridden
                volume_km3: params.volume_km3,
                height_km: params.height_km,
                pressure_gpa: 0.0, // Default surface pressure
            }
        );

        // Set temperature directly using the energy_mass set_kelvin method
        energy_mass.set_kelvin(params.temperature_k);

        Self {
            energy_mass,
            thermal_state: 100, // Start as solid
            depth_km: params.depth_km,
            height_km: params.height_km,
            initial_temperature: params.temperature_k,
            max_temperature: params.temperature_k,
            min_temperature: params.temperature_k,
            total_outgassed_mass: 0.0,
            outgassing_rate: 0.0,
        }
    }

    /// Convenience method for getting temperature (alias for kelvin())
    pub fn temp_kelvin(&self) -> f64 {
        self.kelvin()
    }

    /// Convenience method for getting volume (alias for volume())
    pub fn volume_km3(&self) -> f64 {
        self.volume()
    }

    /// Convenience method for getting mass (delegates to energy_mass)
    pub fn mass_kg_convenience(&self) -> f64 {
        self.energy_mass.mass_kg()
    }

    /// Get the depth in km (specific to ThermalLayerNode)
    pub fn depth_km(&self) -> f64 {
        self.depth_km
    }

    /// Setter methods for controlled access to internal properties

    /// Set the material composite type (changes the underlying material)
    pub fn set_material_type(&mut self, material_type: MaterialCompositeType) {
        // Create new energy mass with same energy but different material
        let current_energy = self.energy_mass.energy();
        let current_volume = self.energy_mass.volume();

        self.energy_mass = StandardEnergyMassComposite::new_with_material_state_and_energy(
            EnergyMassParams {
                material_type,
                initial_phase: MaterialPhase::Solid,
                energy_joules: current_energy,
                volume_km3: current_volume,
                height_km: self.height_km,
                pressure_gpa: self.energy_mass.pressure_gpa(), // Preserve current pressure
            }
        );
        self.log_extent();
    }

    /// Set the material phase (solid, liquid, gas)
    /// Note: This method no longer forces temperature changes.
    /// Phase transitions are now handled automatically by the energy bank system.
    pub fn set_material_phase(&mut self, phase: MaterialPhase) {
        // Phase transitions are now handled automatically by the energy mass system
        // This method is kept for compatibility but doesn't force temperature changes
        // The actual phase will be determined by the current temperature and energy bank state
        self.log_extent();
    }

    /// Set the volume (adjusts energy proportionally to maintain temperature)
    pub fn set_volume(&mut self, new_volume_km3: f64) {
        let current_temp = self.kelvin();
        self.energy_mass.scale(new_volume_km3 / self.volume());
        // Restore temperature by adjusting energy
        self.set_kelvin(current_temp);
        self.log_extent();
    }

    /// Additional getter methods for properties accessed by experiments

    /// Get the thermal capacity (J/K) - used for energy transfer calculations
    pub fn thermal_capacity(&self) -> f64 {
        self.energy_mass.thermal_capacity()
    }

    /// Get the material phase (for phase change logic)
    pub fn phase(&self) -> MaterialPhase {
        self.energy_mass.phase
    }

    /// Set the material phase (for phase change logic)
    pub fn set_phase(&mut self, phase: MaterialPhase) {
        self.set_material_phase(phase);
    }

    /// Update the material phase based on current temperature
    /// This encapsulates the phase transition logic and should be called after temperature changes
    pub fn update_phase_from_kelvin(&mut self) {
        let temp = self.kelvin();
        let profile = self.material_composite_profile();

        // Determine the appropriate phase based on temperature using JSON material data
        let new_phase = if temp < profile.melt_temp {  // Use actual melting point (1600K for silicate)
            MaterialPhase::Solid
        } else if temp < profile.boil_temp {  // Use actual boiling point (3200K for silicate)
            MaterialPhase::Liquid
        } else {
            MaterialPhase::Gas
        };

        // Update thermal_state to reflect the phase (for display/analysis purposes)
        // This follows the existing thermal state scale: 100=solid, 0=liquid, -100=gas
        self.thermal_state = match new_phase {
            MaterialPhase::Solid => 100,
            MaterialPhase::Liquid => 0,
            MaterialPhase::Gas => -100,
        };

        self.log_extent();
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

/// Implement EnergyMassComposite trait for ThermalLayerNode
/// This allows ThermalLayerNode to be used anywhere an EnergyMassComposite is expected
impl EnergyMassComposite for ThermalLayerNode {
    /// Get the current temperature in Kelvin
    fn kelvin(&self) -> f64 {
        self.energy_mass.kelvin()
    }

    /// Set the temperature in Kelvin, updating energy accordingly (volume stays constant)
    /// note -- in general you add or remove energy in a simulation not mandate the temperature
    /// 
    fn set_kelvin(&mut self, kelvin: f64) {
        self.energy_mass.set_kelvin(kelvin);
        self.update_phase_from_kelvin(); // Automatically update phase when temperature changes
        self.log_extent();
    }

    /// Get the current energy in Joules (read-only)
    fn energy(&self) -> f64 {
        self.energy_mass.energy()
    }

    /// Get the current volume in km³ (read-only)
    fn volume(&self) -> f64 {
        self.energy_mass.volume()
    }

    /// Get the height in km (for layer-based calculations)
    fn height_km(&self) -> f64 {
        self.height_km
    }

    /// Get the mass in kg (derived from volume and density)
    fn mass_kg(&self) -> f64 {
        self.energy_mass.mass_kg()
    }

    /// Get the density in kg/m³
    fn density_kgm3(&self) -> f64 {
        self.energy_mass.density_kgm3()
    }

    /// Get the specific heat in J/(kg·K)
    fn specific_heat_j_kg_k(&self) -> f64 {
        self.energy_mass.specific_heat_j_kg_k()
    }

    /// Get the thermal conductivity in W/(m·K)
    fn thermal_conductivity(&self) -> f64 {
        self.energy_mass.thermal_conductivity()
    }

    /// Get the material composite type
    fn material_composite_type(&self) -> MaterialCompositeType {
        self.energy_mass.material_composite_type()
    }

    /// Get the material composite profile
    fn material_composite_profile(&self) -> &'static MaterialStateProfile {
        self.energy_mass.material_composite_profile()
    }

    // material_composite() method removed - MaterialComposite struct no longer exists
    // Use material_composite_type() and get_profile_fast() instead

    /// Scale the entire EnergyMass by a factor (useful for splitting/combining)
    fn scale(&mut self, factor: f64) {
        self.energy_mass.scale(factor);
        self.log_extent();
    }

    /// Add energy to this energy mass
    fn add_energy(&mut self, energy_joules: f64) {
        self.energy_mass.add_energy(energy_joules);
        self.update_phase_from_kelvin(); // Update phase after energy change
        self.log_extent();
    }

    /// Remove energy from this energy mass
    fn remove_energy(&mut self, energy_joules: f64) {
        self.energy_mass.remove_energy(energy_joules);
        self.update_phase_from_kelvin(); // Update phase after energy change
        self.log_extent();
    }

    /// Send energy to another energy mass composite in a single atomic operation
    fn send_energy(&mut self, energy_joules: f64, recipient: &mut dyn EnergyMassComposite) {
        self.energy_mass.send_energy(energy_joules, recipient);
        self.update_phase_from_kelvin(); // Update phase after energy change
        self.log_extent();
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    /// Remove heat energy (temperature will decrease, enforces zero minimum)
    fn remove_joules(&mut self, heat_joules: f64) {
        self.energy_mass.remove_joules(heat_joules);
        self.update_phase_from_kelvin(); // Update phase after energy change
        self.log_extent();
    }

    /// Add energy (temperature will increase)
    fn add_joules(&mut self, energy_joules: f64) {
        self.energy_mass.add_joules(energy_joules);
        self.update_phase_from_kelvin(); // Update phase after energy change
        self.log_extent();
    }

    /// Radiate energy to another EnergyMass using conductive transfer
    fn radiate_to(&mut self, other: &mut dyn EnergyMassComposite, distance_km: f64, area_km2: f64, time_years: f64) -> f64 {
        let result = self.energy_mass.radiate_to(other, distance_km, area_km2, time_years);
        self.log_extent();
        result
    }

    /// Radiate energy to space using Stefan-Boltzmann law
    fn radiate_to_space(&mut self, area_km2: f64, time_years: f64) -> f64 {
        let result = self.energy_mass.radiate_to_space(area_km2, time_years);
        self.log_extent();
        result
    }

    /// Radiate energy to space using Stefan-Boltzmann law with thermal skin depth limiting
    fn radiate_to_space_with_skin_depth(&mut self, area_km2: f64, time_years: f64, energy_throttle: f64) -> f64 {
        let result = self.energy_mass.radiate_to_space_with_skin_depth(area_km2, time_years, energy_throttle);
        self.log_extent();
        result
    }

    /// Compute thermal-diffusive skin depth in kilometres for this material
    fn skin_depth_km(&self, time_years: f64) -> f64 {
        self.energy_mass.skin_depth_km(time_years)
    }

    /// Remove volume (enforces zero minimum, maintains temperature)
    fn remove_volume_internal(&mut self, volume_to_remove: f64) {
        self.energy_mass.remove_volume_internal(volume_to_remove);
        self.log_extent();
    }

    /// Merge another EnergyMass into this one
    fn merge_em(&mut self, other: &dyn EnergyMassComposite) {
        self.energy_mass.merge_em(other);
        self.log_extent();
    }

    /// Remove a specified volume from this EnergyMass, returning a new EnergyMass with that volume
    fn remove_volume(&mut self, volume_to_remove: f64) -> Box<dyn EnergyMassComposite> {
        let result = self.energy_mass.remove_volume(volume_to_remove);
        self.log_extent();
        result
    }

    /// Split this EnergyMass into two parts by volume fraction
    fn split_by_fraction(&mut self, fraction: f64) -> Box<dyn EnergyMassComposite> {
        let result = self.energy_mass.split_by_fraction(fraction);
        self.log_extent();
        result
    }

    /// Get the R0 thermal transmission coefficient for this material
    fn thermal_transmission_r0(&self) -> f64 {
        self.energy_mass.thermal_transmission_r0()
    }

    /// Get the current pressure in GPa
    fn pressure_gpa(&self) -> f64 {
        self.energy_mass.pressure_gpa()
    }

    /// Set the pressure in GPa and update phase accordingly
    fn set_pressure_gpa(&mut self, pressure_gpa: f64) {
        self.energy_mass.set_pressure_gpa(pressure_gpa);
        self.log_extent();
    }

    /// Get the current material phase (pressure-aware)
    fn phase(&self) -> MaterialPhase {
        self.energy_mass.phase()
    }
}
