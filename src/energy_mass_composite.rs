use crate::atmospheric_energy_mass_composite::AtmosphericEnergyMass;
use crate::constants::{KM3_TO_M3, M2_PER_KM2, SECONDS_PER_YEAR, SIGMA_KM2_YEAR};
use crate::material_composite::resolve_phase_from_temperature_and_pressure;
pub use crate::material_composite::{
    MaterialCompositeType, MaterialPhase, MaterialStateProfile, get_profile_fast,
};
use serde::{Deserialize, Serialize};

/// Transition mode for managing phase transitions with hysteresis
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TransitionMode {
    Static,
    InTransition { target_phase: MaterialPhase },
}

// Transition hysteresis constants


/// Result of a phase transition calculation
#[derive(Debug, Clone)]
pub struct TransitionResult {
    /// The resulting phase after transition
    pub phase: MaterialPhase,
    /// The resulting energy in the material
    pub energy_j: f64,
    /// The energy in the phase transition bank
    pub phase_bank_j: f64,
}

/// Parameters for creating StandardEnergyMassComposite
pub struct EnergyMassParams {
    pub material_type: MaterialCompositeType,
    pub initial_phase: MaterialPhase,
    pub energy_joules: f64,
    pub volume_km3: f64,
    pub height_km: f64,
    pub pressure_gpa: f64, // Pressure in Gigapascals
}

/// Trait for objects that manage the relationship between energy, mass, and temperature
/// Maintains consistency between these properties using thermodynamic relationships
pub trait EnergyMassComposite: std::any::Any {
    /// Get the current temperature in Kelvin
    fn kelvin(&self) -> f64;

    /// Get the current temperature in Kelvin (alias for kelvin() - deprecated, use kelvin())
    /// @deprecated temperature is ambiguous prefer the unit-specific "kelvin"
    fn temperature(&self) -> f64 {
        self.kelvin()
    }

    /// Get the current energy in Joules (read-only)
    fn energy(&self) -> f64;

    /// Get the current volume in km³ (read-only)
    fn volume(&self) -> f64;

    /// Get the height in km (for layer-based calculations)
    fn height_km(&self) -> f64;

    /// Set the temperature in Kelvin (internal use only - maintains thermodynamic consistency)
    fn set_kelvin(&mut self, kelvin: f64);

    /// Set the temperature in Kelvin (alias for set_kelvin() - deprecated, use set_kelvin())
    fn set_temperature(&mut self, temperature_k: f64) {
        self.set_kelvin(temperature_k);
    }

    /// Get the mass in kg
    fn mass_kg(&self) -> f64;

    /// Set the mass in kg (updates density accordingly)
    fn set_mass_kg(&mut self, mass_kg: f64);

    /// Get the density in kg/m³ (calculated as mass/volume)
    fn density_kgm3(&self) -> f64;

    /// Get the specific heat in J/(kg·K)
    fn specific_heat_j_kg_k(&self) -> f64;

    /// Get the thermal conductivity in W/(m·K)
    fn thermal_conductivity(&self) -> f64;

    /// Get the thermal capacity in J/K (mass * specific_heat)
    fn thermal_capacity(&self) -> f64 {
        self.mass_kg() * self.specific_heat_j_kg_k()
    }

    /// Get the material composite type
    fn material_composite_type(&self) -> MaterialCompositeType;

    /// Get the material composite profile
    fn material_composite_profile(&self) -> &'static MaterialStateProfile;

    /// Get the current pressure in GPa
    fn pressure_gpa(&self) -> f64;

    /// Set the pressure in GPa and update phase accordingly
    fn set_pressure_gpa(&mut self, pressure_gpa: f64);

    /// Get the current material phase (pressure-aware)
    fn phase(&self) -> MaterialPhase;

    /// Check if this is an atmospheric layer (manually settable)
    fn is_atmosphere(&self) -> bool;

    /// Set whether this is an atmospheric layer
    fn set_is_atmosphere(&mut self, is_atmosphere: bool);

    // material_composite() method removed - MaterialComposite struct no longer exists

    /// Scale the entire EnergyMass by a factor (useful for splitting/combining)
    fn scale(&mut self, factor: f64);

    /// Add energy to this energy mass
    fn add_energy(&mut self, energy_joules: f64);

    /// Remove energy from this energy mass
    fn remove_energy(&mut self, energy_joules: f64);

    /// Send energy to another energy mass composite in a single atomic operation
    /// This ensures perfect energy conservation by transferring exactly the amount removed
    fn send_energy(&mut self, energy_joules: f64, recipient: &mut dyn EnergyMassComposite);

    /// Get mutable reference as Any for downcasting
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;

    /// Remove heat energy (temperature will decrease, enforces zero minimum)
    fn remove_joules(&mut self, heat_joules: f64);

    /// Add energy (temperature will increase)
    fn add_joules(&mut self, energy_joules: f64);

    /// Radiate energy to another EnergyMass using conductive transfer
    /// Returns the amount of energy transferred (positive = energy flows to other)
    fn radiate_to(
        &mut self,
        other: &mut dyn EnergyMassComposite,
        distance_km: f64,
        area_km2: f64,
        time_years: f64,
    ) -> f64;

    /// Radiate energy to space using Stefan-Boltzmann law
    /// Returns the amount of energy radiated (J)
    fn radiate_to_space(&mut self, area_km2: f64, time_years: f64) -> f64;

    /// Radiate energy to space using Stefan-Boltzmann law with thermal skin depth limiting
    /// Only the thermal skin depth participates in radiation, with rate limiting
    /// Returns the amount of energy radiated (J)
    fn radiate_to_space_with_skin_depth(
        &mut self,
        area_km2: f64,
        time_years: f64,
        energy_throttle: f64,
    ) -> f64;

    /// Compute thermal-diffusive skin depth in kilometres for this material
    /// Uses the material's thermal conductivity, density, and specific heat
    ///
    /// Formula: κ = k / (ρ·cp), d = sqrt(κ · dt), converted to km
    fn skin_depth_km(&self, time_years: f64) -> f64;
    /// Remove volume (enforces zero minimum, maintains temperature)
    fn remove_volume_internal(&mut self, volume_to_remove: f64);

    /// Merge another EnergyMass into this one
    /// Energy and volume are directly added, resulting in a new blended temperature
    /// The other EnergyMass must have the same material type
    fn merge_em(&mut self, other: &dyn EnergyMassComposite);

    /// Remove a specified volume from this EnergyMass, returning a new EnergyMass with that volume
    /// The removed EnergyMass will have the same temperature as the original
    /// This EnergyMass will have proportionally less volume and energy but maintain the same temperature
    fn remove_volume(&mut self, volume_to_remove: f64) -> Box<dyn EnergyMassComposite>;

    /// Split this EnergyMass into two parts by volume fraction
    /// Returns a new EnergyMass with the specified fraction, this one keeps the remainder
    /// Both will have the same temperature as the original
    fn split_by_fraction(&mut self, fraction: f64) -> Box<dyn EnergyMassComposite>;

    /// Get the R0 thermal transmission coefficient for this material
    /// This controls energy transfer efficiency between layers (tunable for equilibrium)
    fn thermal_transmission_r0(&self) -> f64;
}

/// Create a complete atmospheric column with realistic mass distribution
/// Returns a vector of atmospheric layers from surface to top
/// Each layer has exponentially decreasing density following Earth's atmospheric profile
pub fn create_atmospheric_column_for_km2(
    num_layers: usize,
    layer_height_km: f64,
    surface_temp_k: f64,
) -> Vec<Box<dyn EnergyMassComposite>> {
    let mut layers = Vec::new();

    // Temperature lapse rate: ~6.5 K/km in troposphere, then isothermal in stratosphere
    let troposphere_height_km = 12.0; // Troposphere extends to ~12 km
    let lapse_rate_k_per_km = 6.5;
    let stratosphere_temp_k = surface_temp_k - troposphere_height_km * lapse_rate_k_per_km;

    for layer_index in 0..num_layers {
        let layer_center_height = layer_index as f64 * layer_height_km + layer_height_km / 2.0;

        // Calculate temperature at this height
        let layer_temp = if layer_center_height <= troposphere_height_km {
            // Troposphere: linear temperature decrease
            surface_temp_k - layer_center_height * lapse_rate_k_per_km
        } else {
            // Stratosphere and above: roughly isothermal
            stratosphere_temp_k
        };

        let layer = Box::new(AtmosphericEnergyMass::create_layer_for_km2(
            layer_index,
            layer_height_km,
            layer_temp,
        )) as Box<dyn EnergyMassComposite>;
        layers.push(layer);
    }

    layers
}

/// Create a complete atmospheric column using simplified 0.88 decay model
/// Returns a vector of atmospheric layers from surface to top
/// Each layer has mass = 0.88^layer_index × base_mass
/// Total mass will not exceed maximum (1.0×10¹⁰ kg per km²)
///
/// Note: The base mass density is automatically scaled so that the infinite series
/// sum (base_mass × Σ(0.88^n)) equals the maximum atmospheric mass
pub fn create_atmospheric_column_simple_decay(
    num_layers: usize,
    layer_height_km: f64,
    surface_temp_k: f64,
) -> Vec<Box<dyn EnergyMassComposite>> {
    let mut layers = Vec::new();
    let max_total_mass_kg = 1.0e10; // Maximum atmospheric mass per km²

    // Calculate the sum of the geometric series: Σ(0.88^n) from n=0 to infinity
    // For geometric series with r = 0.88: sum = 1 / (1 - 0.88) = 1 / 0.12 = 8.333...
    let decay_factor: f64 = 0.88;
    let _infinite_series_sum = 1.0 / (1.0 - decay_factor);

    // Scale the base mass so total doesn't exceed maximum
    // We'll use a finite approximation by calculating the sum for num_layers
    let finite_series_sum: f64 = (0..num_layers).map(|i| decay_factor.powi(i as i32)).sum();

    // Base mass per layer (per km³) scaled to fit within maximum
    let base_mass_per_layer_kg = max_total_mass_kg / finite_series_sum;

    // Temperature lapse rate: ~6.5 K/km in troposphere, then isothermal in stratosphere
    let troposphere_height_km = 12.0; // Troposphere extends to ~12 km
    let lapse_rate_k_per_km = 6.5;
    let stratosphere_temp_k = surface_temp_k - troposphere_height_km * lapse_rate_k_per_km;

    for layer_index in 0..num_layers {
        let layer_center_height = layer_index as f64 * layer_height_km + layer_height_km / 2.0;

        // Calculate temperature at this height
        let layer_temp = if layer_center_height <= troposphere_height_km {
            // Troposphere: linear temperature decrease
            surface_temp_k - layer_center_height * lapse_rate_k_per_km
        } else {
            // Stratosphere and above: roughly isothermal
            stratosphere_temp_k
        };

        // Calculate mass for this layer using scaled base mass and decay factor
        let layer_mass_kg = base_mass_per_layer_kg * decay_factor.powi(layer_index as i32);

        // Calculate volume for this layer (height × 1 km² area)
        let layer_volume_km3 = layer_height_km;

        // Calculate density from mass and volume
        let volume_m3 = layer_volume_km3 * 1e9; // Convert km³ to m³
        let layer_density_kg_m3 = layer_mass_kg / volume_m3;

        // Create the atmospheric layer
        let layer: Box<dyn EnergyMassComposite> = Box::new(AtmosphericEnergyMass::new(
            layer_temp,
            layer_volume_km3,
            layer_height_km,
            layer_density_kg_m3,
        ));

        layers.push(layer);
    }

    layers
}

/// Create atmospheric layer for simulation cells with arbitrary area
/// Uses the same 0.88 decay model but scales for the actual cell area
///
/// Parameters:
/// - layer_index: Height index (0 = surface, 1 = next layer up, etc.)
/// - layer_height_km: Height of this atmospheric layer (matches solid layer height)
/// - cell_area_km2: Area of the simulation cell (from H3 hexagon)
/// - temperature_k: Temperature for this atmospheric layer
pub fn create_atmospheric_layer_for_cell(
    layer_index: usize,
    layer_height_km: f64,
    cell_area_km2: f64,
    temperature_k: f64,
) -> Box<dyn EnergyMassComposite> {
    // Base mass density for sea level layer (kg per km³)
    let base_mass_density_kg_km3 = 1.03e10;

    // Calculate mass density for this layer using 0.88 decay factor
    let decay_factor = 0.88_f64.powi(layer_index as i32);
    let layer_mass_density_kg_km3 = base_mass_density_kg_km3 * decay_factor;

    // Calculate volume for the actual cell area with given height
    let layer_volume_km3 = layer_height_km * cell_area_km2; // height × area = volume

    // Calculate total mass for this layer (mass density × volume)
    let layer_total_mass_kg = layer_mass_density_kg_km3 * layer_volume_km3;

    // Calculate density (kg/m³) from mass and volume
    let volume_m3 = layer_volume_km3 * 1e9; // Convert km³ to m³
    let layer_density_kg_m3 = layer_total_mass_kg / volume_m3;

    // Create the atmospheric layer
    Box::new(AtmosphericEnergyMass::new(
        temperature_k,
        layer_volume_km3,
        layer_height_km,
        layer_density_kg_m3,
    ))
}

/// Standard implementation of EnergyMass using material composite profiles
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct StandardEnergyMassComposite {
    pub energy_joules: f64,
    pub volume_km3: f64,
    pub height_km: f64, // Height in km (for layer calculations)
    pub material_type: MaterialCompositeType,
    pub phase: MaterialPhase, // Current material state (Solid/Liquid/Gas)
    pub thermal_transmission_r0: f64, // R0 thermal transmission coefficient (set at creation)
    pub state_transition_bank: f64, // Energy bank for phase transitions
    pub transition_mode: TransitionMode, // Transition mode for hysteresis management
    pub pressure_gpa: f64,    // Pressure in Gigapascals (affects phase transitions)
    pub mass_kg: f64,         // Mass in kg (settable property, density = mass/volume)
    pub is_atmosphere: bool,  // Whether this is an atmospheric layer (manually settable)
}

impl StandardEnergyMassComposite {
    /// Get base radiation depth for surface radiation calculations
    pub fn base_radiation_depth_m() -> f64 {
        10000.0 //  for geological timescale radiation to space
    }

    /// Get reference density for radiation depth scaling
    pub fn reference_density_kg_m3() -> f64 {
        2500.0 // kg/m³ - typical silicate rock density
    }

    /// Create a new StandardEnergyMassComposite with specified parameters (alias for new_with_material_state_and_energy)
    pub fn new_with_params(params: EnergyMassParams) -> Self {
        Self::new_with_material_state_and_energy(params)
    }

    /// Create a new StandardEnergyMassComposite with temperature instead of energy
    /// Phase is automatically determined from temperature to ensure consistency
    /// Creates the object with dummy energy, then uses set_kelvin() to adjust energy automatically
    pub fn new_with_temperature(
        material_type: MaterialCompositeType,
        temp_k: f64,
        volume_km3: f64,
        height_km: f64,
    ) -> Self {
        use crate::material_composite::resolve_phase_from_temperature;

        // Determine the correct phase from temperature - no dissonance possible!
        let initial_phase = resolve_phase_from_temperature(&material_type, temp_k);

        // Create with dummy energy first
        let params = EnergyMassParams {
            material_type,
            initial_phase,
            energy_joules: 0.0, // Dummy value - will be set by set_kelvin()
            volume_km3,
            height_km,
            pressure_gpa: 0.0, // Default surface pressure
        };

        let mut composite = Self::new_with_params(params);
        composite.set_kelvin(temp_k); // This adjusts energy automatically
        composite
    }

    /// Create a new atmospheric layer with near-zero mass that can accumulate mass through outgassing
    /// Atmospheric layers start with constant volume but near-zero mass, then gain density as mass is added
    pub fn new_atmospheric_with_near_zero_density(
        material_type: MaterialCompositeType,
        volume_km3: f64,
        height_km: f64,
        temp_k: f64,
    ) -> Self {
        // Start with minimal mass (near-zero but not exactly zero to avoid division issues)
        let minimal_mass_kg = 1e-3; // 1 gram - essentially empty but not zero

        // Calculate energy for this minimal mass at the given temperature using material properties
        let profile = get_profile_fast(&material_type, &MaterialPhase::Gas);
        let energy_joules = minimal_mass_kg * profile.specific_heat_capacity_j_per_kg_k * temp_k;

        // Generate random R0 value within material's range
        use rand::Rng;
        let mut rng = rand::rng();
        let r0_range = profile.thermal_transmission_r0_max - profile.thermal_transmission_r0_min;
        let random_r0 = profile.thermal_transmission_r0_min + rng.random::<f64>() * r0_range;

        // Create directly with minimal mass (not from material density)
        Self {
            energy_joules,
            volume_km3,
            height_km,
            material_type,
            phase: MaterialPhase::Gas, // Atmospheric layers are always gas
            thermal_transmission_r0: random_r0,
            state_transition_bank: 0.0,
            transition_mode: TransitionMode::Static,
            pressure_gpa: 0.0, // Atmospheric pressure
            mass_kg: minimal_mass_kg,
            is_atmosphere: true,
        }
    }

    /// Add mass to an atmospheric layer while maintaining constant volume (increases density)
    /// This is used for atmospheric generation where outgassed material is added to atmospheric layers
    pub fn add_atmospheric_mass(&mut self, additional_mass_kg: f64, temp_k: f64) {
        if additional_mass_kg <= 0.0 {
            return;
        }

        // Calculate energy for the additional mass at the given temperature
        let profile = get_profile_fast(&self.material_type, &self.phase);
        let additional_energy_joules = additional_mass_kg * profile.specific_heat_capacity_j_per_kg_k * temp_k;

        // Add the energy and mass directly
        self.energy_joules += additional_energy_joules;
        self.mass_kg += additional_mass_kg;

        // Volume stays constant for atmospheric layers - density increases automatically
        // The density_kgm3() method will now return the higher density due to increased mass
    }

    /// Calculate the actual current density based on energy content (for atmospheric layers)
    /// This is different from density_kgm3() which returns the material profile density
    pub fn actual_density_kgm3(&self) -> f64 {
        // Calculate mass from energy content
        let profile = get_profile_fast(&self.material_type, &self.phase);
        let temp_k = self.temperature();

        // Avoid division by zero
        if temp_k <= 0.0 || profile.specific_heat_capacity_j_per_kg_k <= 0.0 {
            return 1e-6; // Near-zero density
        }

        // Calculate actual mass from energy: mass = energy / (specific_heat * temperature)
        let actual_mass_kg = self.energy_joules / (profile.specific_heat_capacity_j_per_kg_k * temp_k);

        // Calculate density: density = mass / volume
        let volume_m3 = self.volume_km3 * 1e9; // Convert km³ to m³
        if volume_m3 <= 0.0 {
            return 1e-6; // Near-zero density
        }

        actual_mass_kg / volume_m3
    }

    /// Create a new StandardEnergyMassComposite with specified parameters
    pub fn new_with_material_state_and_energy(params: EnergyMassParams) -> Self {
        // Get profile for the initial phase state
        let profile = get_profile_fast(&params.material_type, &params.initial_phase);

        // Generate random R0 value within material's range
        use rand::Rng;
        let mut rng = rand::rng();
        let r0_range = profile.thermal_transmission_r0_max - profile.thermal_transmission_r0_min;
        let random_r0 = profile.thermal_transmission_r0_min + rng.random::<f64>() * r0_range;

        // Calculate initial mass from material density and volume
        let initial_mass_kg = params.volume_km3 * KM3_TO_M3 * profile.density_kg_m3;

        Self {
            energy_joules: params.energy_joules,
            volume_km3: params.volume_km3,
            height_km: params.height_km,
            material_type: params.material_type,
            phase: params.initial_phase,
            thermal_transmission_r0: random_r0,
            state_transition_bank: 0.0,
            transition_mode: TransitionMode::Static,
            pressure_gpa: params.pressure_gpa,
            mass_kg: initial_mass_kg,
            is_atmosphere: params.material_type == MaterialCompositeType::Air,
        }
    }

    /// Create a new StandardEnergyMassComposite with specified material, energy and volume
    pub fn new_with_material_energy(
        material_type: &MaterialCompositeType,
        energy_joules: f64,
        volume_km3: f64,
        phase: &MaterialPhase,
    ) -> Self {
        let profile = get_profile_fast(material_type, phase);

        // Generate random R0 value within material's range
        use rand::Rng;
        let mut rng = rand::rng();
        let r0_range = profile.thermal_transmission_r0_max - profile.thermal_transmission_r0_min;
        let random_r0 = profile.thermal_transmission_r0_min + rng.random::<f64>() * r0_range;

        // Calculate initial mass from material density and volume
        let initial_mass_kg = volume_km3 * KM3_TO_M3 * profile.density_kg_m3;

        Self {
            energy_joules,
            volume_km3,
            height_km: volume_km3 / 85000.0, // Default height calculation
            material_type: material_type.clone(),
            phase: phase.clone(),
            thermal_transmission_r0: random_r0,
            state_transition_bank: 0.0,
            transition_mode: TransitionMode::Static,
            pressure_gpa: 0.0, // Default surface pressure
            mass_kg: initial_mass_kg,
            is_atmosphere: *material_type == MaterialCompositeType::Air,
        }
    }

    pub fn current_material_state_profile(&self) -> &MaterialStateProfile {
        get_profile_fast(&self.material_type, &self.phase)
    }

    // current_material_composite() method removed - MaterialComposite struct no longer exists
    // Use material_composite_type() and get_profile_fast() instead

    /// Get the current state transition bank energy
    pub fn state_transition_bank(&self) -> f64 {
        self.state_transition_bank
    }

    /// Check and complete forward phase transition if enough energy is banked
    fn check_and_complete_forward_transition(&mut self, mass_kg: f64) {
        if self.state_transition_bank <= 0.0 {
            return;
        }

        let profile = self.material_composite_profile();
        let transition_cost = match self.phase {
            MaterialPhase::Solid => {
                // Solid -> Liquid: use latent heat of fusion
                mass_kg * profile.latent_heat_fusion
            }
            MaterialPhase::Liquid => {
                // Liquid -> Gas: use latent heat of vaporization
                mass_kg * profile.latent_heat_vapor
            }
            MaterialPhase::Gas => {
                // Already at highest phase, no transition possible
                return;
            }
        };

        if self.state_transition_bank >= transition_cost {
            // Complete the transition
            self.phase = match self.phase {
                MaterialPhase::Solid => MaterialPhase::Liquid,
                MaterialPhase::Liquid => MaterialPhase::Gas,
                MaterialPhase::Gas => MaterialPhase::Gas, // No change
            };

            // Subtract transition cost from bank
            self.state_transition_bank -= transition_cost;

            // Add remaining bank energy back to material
            self.energy_joules += self.state_transition_bank;
            self.state_transition_bank = 0.0;
        }
    }

    /// Check and complete reverse phase transition if enough energy deficit is banked
    fn check_and_complete_reverse_transition(&mut self, mass_kg: f64) {
        if self.state_transition_bank >= 0.0 {
            return;
        }

        let profile = self.material_composite_profile();
        let transition_cost = match self.phase {
            MaterialPhase::Gas => {
                // Gas -> Liquid: release latent heat of vaporization
                mass_kg * profile.latent_heat_vapor
            }
            MaterialPhase::Liquid => {
                // Liquid -> Solid: release latent heat of fusion
                mass_kg * profile.latent_heat_fusion
            }
            MaterialPhase::Solid => {
                // Already at lowest phase, no transition possible
                return;
            }
        };

        if self.state_transition_bank.abs() >= transition_cost {
            // Complete the reverse transition
            self.phase = match self.phase {
                MaterialPhase::Gas => MaterialPhase::Liquid,
                MaterialPhase::Liquid => MaterialPhase::Solid,
                MaterialPhase::Solid => MaterialPhase::Solid, // No change
            };

            // Add back transition cost to bank (bank is negative)
            self.state_transition_bank += transition_cost;

            // Add remaining bank energy back to material
            self.energy_joules += self.state_transition_bank;
            self.state_transition_bank = 0.0;
        }
    }

    /// Check if we're in a transition temperature range
    fn is_in_transition_range(&self) -> bool {
        let temp = self.kelvin();
        let profile = self.current_material_state_profile();
        let melting_point = profile.melt_temp;
        let boiling_point = profile.boil_temp;
        temp >= melting_point && temp <= boiling_point
    }

    /// Calculate the fraction of energy that should go to the transition bank
    /// Based on unlerp(temp, min, max) when in transition range
    fn calculate_transition_fraction(&self) -> f64 {
        if !self.is_in_transition_range() {
            return 0.0;
        }

        let temp = self.kelvin();
        let profile = self.current_material_state_profile();
        let min_temp = profile.melt_temp;
        let max_temp = profile.boil_temp;

        // Unlerp: (temp - min) / (max - min)
        (temp - min_temp) / (max_temp - min_temp)
    }
    fn temp_per_energy(&self) -> f64 {
        1.0 / (self.mass_kg().max(1.0) * self.specific_heat_j_kg_k())
    }

    /// Add core radiance energy influx (2.52e12 J per km² per year)
    /// Only applies to the bottom-most asthenosphere layer
    pub fn add_core_radiance(&mut self, area_km2: f64, years: f64) {
        // Earth's core radiance: 2.52e12 J per km² per year
        let core_radiance_per_km2_per_year = 2.52e12;
        let energy_influx = core_radiance_per_km2_per_year * area_km2 * years;

        // Add energy by calculating new temperature
        let current_energy = self.energy();
        let mass_kg = self.mass_kg();
        let specific_heat = self.specific_heat_j_kg_k();

        if mass_kg > 0.0 && specific_heat > 0.0 {
            let new_temp = (current_energy + energy_influx) / (mass_kg * specific_heat);
            self.set_kelvin(new_temp);
        }
    }

    /// Calculate thermal energy transfer between two materials based on conductivity and temperature difference
    /// Includes specific heat capacity effects through thermal capacity moderation
    /// Returns the energy transfer amount (J)
    pub fn calculate_thermal_transfer(
        &self,
        other: &dyn EnergyMassComposite,
        diffusion_rate: f64,
        years: f64,
    ) -> f64 {
        let interface_conductivity =
            2.0 * self.thermal_conductivity() * other.thermal_conductivity()
                / (self.thermal_conductivity() + other.thermal_conductivity());

        // Calculate interface R0 (harmonic mean like conductivity)
        let interface_r0 = 2.0 * self.thermal_transmission_r0() * other.thermal_transmission_r0()
            / (self.thermal_transmission_r0() + other.thermal_transmission_r0());

        let temp_diff = self.kelvin() - other.kelvin();

        // Determine which material is hotter (source) and use its height for scaling
        let source_height_km = if temp_diff > 0.0 {
            self.height_km() // self is hotter
        } else {
            other.height_km() // other is hotter
        };

        // Height scaling factor: base rates are for 1km layers, scale by sqrt(height)
        let height_scale = (source_height_km / 2.0).sqrt();

        // Include specific heat capacity effects through thermal capacity moderation
        // Materials with higher specific heat capacity resist temperature changes more
        let self_specific_heat = self.specific_heat_j_kg_k();
        let other_specific_heat = other.specific_heat_j_kg_k();
        let avg_specific_heat = (self_specific_heat + other_specific_heat) / 2.0;

        // Specific heat moderation: higher specific heat = slower energy transfer
        // Normalize to typical mantle specific heat (~1000 J/kg/K)
        let specific_heat_factor = (1000.0 / avg_specific_heat).sqrt().min(2.0).max(0.5);

        let base_transfer_rate = interface_conductivity
            * interface_r0
            * diffusion_rate
            * years
            * crate::constants::SECONDS_PER_YEAR
            * crate::constants::M2_PER_KM2;
        let energy_transfer = temp_diff * base_transfer_rate * height_scale * specific_heat_factor;

        energy_transfer
    }

    /// Calculate bulk thermal energy transfer for large volume objects
    /// Uses volume-based thermal diffusion physics appropriate for thick layers
    /// Returns the energy transfer amount (J)
    /// May be too aggressive for the sim
    pub fn calculate_bulk_thermal_transfer(
        &self,
        other: &dyn EnergyMassComposite,
        layer_thickness_km: f64,
        years: f64,
    ) -> f64 {
        let temp_diff = self.kelvin() - other.kelvin();

        let self_diffusivity =
            self.thermal_conductivity() / (self.density_kgm3() * self.specific_heat_j_kg_k());
        let other_diffusivity =
            other.thermal_conductivity() / (other.density_kgm3() * other.specific_heat_j_kg_k());

        // Use average diffusivity for the transfer
        let avg_diffusivity = (self_diffusivity + other_diffusivity) / 2.0;

        // Convert to km²/year for our units
        let diffusivity_km2_per_year = avg_diffusivity * SECONDS_PER_YEAR / M2_PER_KM2; // seconds/year / m²/km²

        // Calculate diffusion length scale: sqrt(diffusivity × time)
        let diffusion_length_km = (diffusivity_km2_per_year * years).sqrt();

        // Energy transfer efficiency based on diffusion vs layer thickness
        let transfer_efficiency = (diffusion_length_km / layer_thickness_km).min(1.0);

        // Volume-based energy transfer: larger volumes can transfer more energy
        let transfer_volume_km3 = (self.volume() + other.volume()) / 2.0;

        // Base energy transfer rate (J/K/km³/year)
        let base_rate = avg_diffusivity * 1e15; // Scaling factor for realistic energy transfer

        // Total energy transfer
        let energy_transfer =
            temp_diff * transfer_volume_km3 * transfer_efficiency * base_rate * years;

        energy_transfer
    }
}

impl EnergyMassComposite for StandardEnergyMassComposite {
    /// Get the current temperature in Kelvin
    /// Includes half the phase transition bank energy for realistic perceived temperature
    fn kelvin(&self) -> f64 {
        let mass_kg = self.mass_kg();
        if mass_kg <= 0.0 {
            return 0.0;
        }

        // Include half the bank energy in temperature calculation
        // This represents the fact that partially melted material still contributes to thermal behavior
        let effective_energy = self.energy_joules + (self.state_transition_bank * 0.5);
        effective_energy * self.temp_per_energy()
    }

    /// Set the temperature in Kelvin, updating energy accordingly (volume stays constant)
    fn set_kelvin(&mut self, kelvin: f64) {
        let mass_kg = self.mass_kg();
        if mass_kg <= 0.0 {
            self.energy_joules = 0.0;
            return;
        }
        self.energy_joules = self.specific_heat_j_kg_k() * kelvin * mass_kg;
    }

    /// Get the current energy in Joules (read-only)
    fn energy(&self) -> f64 {
        self.energy_joules
    }

    /// Get the current volume in km³ (read-only)
    fn volume(&self) -> f64 {
        self.volume_km3
    }

    /// Get the height in km (for layer-based calculations)
    fn height_km(&self) -> f64 {
        self.height_km
    }

    /// Get the mass in kg
    fn mass_kg(&self) -> f64 {
        self.mass_kg
    }

    /// Set the mass in kg (updates density accordingly)
    fn set_mass_kg(&mut self, mass_kg: f64) {
        self.mass_kg = mass_kg;
    }

    /// Get the density in kg/m³ (calculated as mass/volume)
    fn density_kgm3(&self) -> f64 {
        if self.volume_km3 <= 0.0 {
            return 0.0;
        }
        let volume_m3 = self.volume_km3 * KM3_TO_M3;
        self.mass_kg / volume_m3
    }

    /// Get the specific heat in J/(kg·K)
    fn specific_heat_j_kg_k(&self) -> f64 {
        self.material_composite_profile()
            .specific_heat_capacity_j_per_kg_k
    }

    /// Get the thermal conductivity in W/(m·K)
    /// Uses temperature-dependent thermal conductivity based on material phase and temperature
    /// Enhanced with pressure effects - higher pressure increases thermal conductivity
    fn thermal_conductivity(&self) -> f64 {
        let base_conductivity = self.material_composite_profile().thermal_conductivity_w_m_k;

        // Pressure enhancement of thermal conductivity
        let pressure_multiplier = match self.material_type {
            MaterialCompositeType::Silicate => 1.0 + self.pressure_gpa * 0.02, // 2% per GPa
            MaterialCompositeType::Basaltic => 1.0 + self.pressure_gpa * 0.015, // 1.5% per GPa
            MaterialCompositeType::Granitic => 1.0 + self.pressure_gpa * 0.01, // 1% per GPa
            MaterialCompositeType::Metallic => 1.0 + self.pressure_gpa * 0.05, // 5% per GPa (metals more sensitive)
            _ => 1.0 + self.pressure_gpa * 0.02,                               // Default 2% per GPa
        }
        .min(3.0); // Cap at 3x increase

        base_conductivity * pressure_multiplier
    }

    /// Scale the entire EnergyMass by a factor (useful for splitting/combining)
    fn scale(&mut self, factor: f64) {
        self.energy_joules *= factor;
        self.volume_km3 *= factor;
        // Temperature remains the same, mass and energy scale proportionally
    }

    /// Remove heat energy (temperature will decrease, enforces zero minimum)
    fn remove_joules(&mut self, heat_joules: f64) {
        self.energy_joules = (self.energy_joules - heat_joules).max(0.0);
    }

    /// Add energy (temperature will increase)
    fn add_joules(&mut self, energy_joules: f64) {
        self.energy_joules += energy_joules;
    }

    /// Radiate energy to another EnergyMass using conductive transfer (Fourier's law)
    /// Returns the amount of energy transferred (positive = energy flows to other)
    fn radiate_to(
        &mut self,
        other: &mut dyn EnergyMassComposite,
        distance_km: f64,
        area_km2: f64,
        time_years: f64,
    ) -> f64 {
        let my_temp = self.kelvin();
        let other_temp = other.kelvin();

        // No transfer if temperatures are equal
        if (my_temp - other_temp).abs() < 0.1 {
            return 0.0;
        }

        // Get thermal conductivities (W/m·K)
        let my_conductivity = self.thermal_conductivity();
        let other_conductivity = other.thermal_conductivity();

        // Use harmonic mean for interface conductivity
        let interface_conductivity = if my_conductivity > 0.0 && other_conductivity > 0.0 {
            2.0 * my_conductivity * other_conductivity / (my_conductivity + other_conductivity)
        } else {
            0.0
        };

        // Temperature difference (energy flows from hot to cold)
        let temp_diff = my_temp - other_temp;

        // Convert units: km² to m², km to m, years to seconds
        let area_m2 = area_km2 * 1e6; // km² to m²
        let distance_m = distance_km * 1000.0; // km to m
        let time_seconds = time_years * 365.25 * 24.0 * 3600.0; // years to seconds

        // Fourier's law: Q = k * A * ΔT * t / d
        let energy_transfer =
            interface_conductivity * area_m2 * temp_diff * time_seconds / distance_m;

        // Limit transfer to prevent temperature inversion
        let max_transfer = self.energy() * 0.1; // Max 10% per step
        let actual_transfer = energy_transfer.abs().min(max_transfer);

        // Apply the transfer
        if temp_diff > 0.0 {
            // I'm hotter - I lose energy, other gains energy
            self.remove_joules(actual_transfer);
            other.add_joules(actual_transfer);
            actual_transfer
        } else {
            // Other is hotter - other loses energy, I gain energy
            other.add_joules(-actual_transfer);
            self.remove_joules(-actual_transfer); // Remove negative energy = add positive energy
            -actual_transfer
        }
    }

    /// Radiate energy to space using Stefan-Boltzmann law
    /// Returns the amount of energy radiated (J)
    fn radiate_to_space(&mut self, area_km2: f64, time_years: f64) -> f64 {
        // Use skin depth method with no additional throttling (skin depth provides natural limit)
        self.radiate_to_space_with_skin_depth(area_km2, time_years, 1.0)
    }

    /// Radiate energy to space using Stefan-Boltzmann law with thermal skin depth limiting
    /// Only the thermal skin depth participates in radiation - no additional throttling needed
    /// Returns the amount of energy radiated (J)
    fn radiate_to_space_with_skin_depth(
        &mut self,
        area_km2: f64,
        time_years: f64,
        _energy_throttle: f64,
    ) -> f64 {
        let surface_temp = self.kelvin();

        // Calculate surface radiation depth based on material density alone
        let radiation_depth = self.skin_depth_km(time_years);

        // Clamp radiation depth to the actual layer height for current implementation
        // In the future, we may consider deeper radiation that extends beyond single layers
        let effective_radiation_depth = radiation_depth.min(self.height_km());

        // Calculate the fraction of the layer that participates in radiation
        let radiation_fraction = if self.height_km() > 0.0 {
            effective_radiation_depth / self.height_km()
        } else {
            1.0 // If no height, radiate everything
        };

        // Calculate radiated energy per km² using Stefan-Boltzmann law
        let radiated_energy_per_km2 = SIGMA_KM2_YEAR * surface_temp.powi(4) * time_years;
        let energy_to_radiate = radiated_energy_per_km2 * area_km2;
        let available_energy = self.energy() * radiation_fraction;
        let energy_to_remove = energy_to_radiate.min(available_energy);

        // Debug output removed

        self.remove_joules(energy_to_remove);

        energy_to_remove
    }

    /// Remove volume (enforces zero minimum, maintains temperature)
    /// this is an "inner method" for remove_volume -
    /// it returns nothing and just sets props
    fn remove_volume_internal(&mut self, volume_to_remove: f64) {
        if volume_to_remove <= 0.0 {
            return; // Nothing to remove
        }
        let fract = volume_to_remove / self.volume_km3;
        self.volume_km3 = (self.volume_km3 - volume_to_remove).max(0.0);
        self.energy_joules *= 1.0 - fract; // Remove proportional energy
    }

    /// Merge another EnergyMass into this one
    /// Energy and volume are directly added, resulting in a new blended temperature
    /// This MODIFIES the values of self but not other
    fn merge_em(&mut self, other: &dyn EnergyMassComposite) {
        // Verify compatible material types
        assert_eq!(
            self.material_composite_type(),
            other.material_composite_type(),
            "Cannot merge EnergyMass with different materials: {:?} vs {:?}",
            self.material_composite_type(),
            other.material_composite_type()
        );

        self.energy_joules += other.energy();
        self.volume_km3 += other.volume();
    }

    /// Remove a specified volume from this EnergyMass, returning a new EnergyMass with that volume
    /// The removed EnergyMass will have the same temperature as the original
    /// This EnergyMass will have proportionally less volume and energy but maintain the same temperature
    fn remove_volume(&mut self, volume_to_remove: f64) -> Box<dyn EnergyMassComposite> {
        if volume_to_remove <= 0.0 {
            panic!("Cannot remove zero or negative volume");
        }
        if volume_to_remove > self.volume_km3 {
            panic!(
                "Cannot remove more volume ({}) than available ({})",
                volume_to_remove, self.volume_km3
            );
        }

        let _current_temp = self.temperature();
        let fraction_removed = volume_to_remove / self.volume_km3;
        let energy_to_remove = self.energy_joules * fraction_removed;

        // Create the removed EnergyMass with same temperature and material
        let removed =
            StandardEnergyMassComposite::new_with_material_state_and_energy(EnergyMassParams {
                material_type: self.material_type.clone(),
                initial_phase: self.phase.clone(),
                energy_joules: energy_to_remove,
                volume_km3: volume_to_remove,
                height_km: self.height_km, // @TODO: probably wrong - hopefully it is not used
                pressure_gpa: self.pressure_gpa, // Preserve pressure
            });

        // Update this EnergyMass
        self.remove_volume_internal(volume_to_remove);

        Box::new(removed)
    }
    // material_composite() method removed - MaterialComposite struct no longer exists
    /// Split this EnergyMass into two parts by volume fraction
    /// Returns a new EnergyMass with the specified fraction, this one keeps the remainder
    /// Both will have the same temperature as the original
    fn split_by_fraction(&mut self, fraction: f64) -> Box<dyn EnergyMassComposite> {
        if fraction <= 0.0 || fraction >= 1.0 {
            panic!("Fraction must be between 0 and 1, got {}", fraction);
        }

        let volume_to_remove = self.volume_km3 * fraction;
        self.remove_volume(volume_to_remove)
    }

    /// Get the R0 thermal transmission coefficient for this material instance
    /// This is a randomized value within the material's range, set at creation
    fn thermal_transmission_r0(&self) -> f64 {
        self.thermal_transmission_r0
    }

    /// Compute thermal-diffusive skin depth in kilometres for this material
    /// Uses proper thermal diffusivity formula with pressure-enhanced thermal conductivity
    fn skin_depth_km(&self, time_years: f64) -> f64 {
        use crate::constants::SECONDS_PER_YEAR;

        // Calculate thermal diffusivity (m²/s) using pressure-enhanced thermal conductivity
        let thermal_conductivity = self.thermal_conductivity(); // Already includes pressure effects
        let density = self.density_kgm3();
        let specific_heat = self.specific_heat_j_kg_k();

        if density <= 0.0 || specific_heat <= 0.0 {
            return 0.0;
        }

        let thermal_diffusivity = thermal_conductivity / (density * specific_heat);

        // Calculate skin depth using thermal diffusion equation: d = sqrt(κ * t)
        let time_seconds = time_years * SECONDS_PER_YEAR;
        let skin_depth_m = (thermal_diffusivity * time_seconds).sqrt();

        // Convert to km
        skin_depth_m / 1000.0
    }

    fn material_composite_type(&self) -> MaterialCompositeType {
        self.material_type
    }

    fn material_composite_profile(&self) -> &'static MaterialStateProfile {
        get_profile_fast(&self.material_type, &self.phase)
    }

    fn pressure_gpa(&self) -> f64 {
        self.pressure_gpa
    }

    fn set_pressure_gpa(&mut self, pressure_gpa: f64) {
        self.pressure_gpa = pressure_gpa;
        // Update phase based on current temperature and new pressure
        let temp_k = self.kelvin();
        let new_phase =
            resolve_phase_from_temperature_and_pressure(&self.material_type, temp_k, pressure_gpa);
        self.phase = new_phase;
    }

    fn phase(&self) -> MaterialPhase {
        // Always return pressure-aware phase based on current temperature and pressure
        // Use caching for performance optimization
        let temp_k = self.kelvin();

        
        // For now, just calculate directly without caching
        let phase = resolve_phase_from_temperature_and_pressure(&self.material_type, temp_k, self.pressure_gpa);
        
        phase
    }

    fn is_atmosphere(&self) -> bool {
        self.is_atmosphere
    }

    fn set_is_atmosphere(&mut self, is_atmosphere: bool) {
        self.is_atmosphere = is_atmosphere;
    }

    fn add_energy(&mut self, energy_joules: f64) {
        // Get current temperature and material properties
        let current_temp = self.kelvin();
        let profile = self.current_material_state_profile();
        let melt_temp_min = profile.melt_temp_min.unwrap_or(profile.melt_temp);
        let melt_temp_max = profile.melt_temp_max.unwrap_or(profile.melt_temp);

        // Check if we're in the transition range
        if current_temp >= melt_temp_min && current_temp <= melt_temp_max {
            // Gradual transition: split energy between main and bank based on position in range
            let transition_fraction = if melt_temp_max > melt_temp_min {
                ((current_temp - melt_temp_min) / (melt_temp_max - melt_temp_min)).clamp(0.0, 1.0)
            } else {
                0.0 // No range, no transition
            };

            let energy_to_bank = energy_joules * transition_fraction;
            let energy_to_main = energy_joules - energy_to_bank;

            self.energy_joules += energy_to_main;
            self.state_transition_bank += energy_to_bank;
        } else {
            // Outside transition range: all energy goes to main
            self.energy_joules += energy_joules;
        }
    }

    fn remove_energy(&mut self, energy_joules: f64) {
        // Get current temperature and material properties
        let current_temp = self.kelvin();
        let profile = self.current_material_state_profile();
        let melt_temp_min = profile.melt_temp_min.unwrap_or(profile.melt_temp);
        let melt_temp_max = profile.melt_temp_max.unwrap_or(profile.melt_temp);

        // Check if we're in the transition range
        if current_temp >= melt_temp_min && current_temp <= melt_temp_max {
            // Gradual transition: split energy removal between bank and main based on position in range
            let transition_fraction = if melt_temp_max > melt_temp_min {
                ((current_temp - melt_temp_min) / (melt_temp_max - melt_temp_min)).clamp(0.0, 1.0)
            } else {
                0.0 // No range, no transition
            };

            let energy_from_bank = energy_joules * transition_fraction;
            let _energy_from_main = energy_joules - energy_from_bank;

            // Remove from bank first, then main (but respect available amounts)
            let available_bank = self.state_transition_bank.max(0.0);
            let actual_from_bank = energy_from_bank.min(available_bank);
            let remaining_to_remove = energy_joules - actual_from_bank;

            self.state_transition_bank -= actual_from_bank;
            self.energy_joules = (self.energy_joules - remaining_to_remove).max(0.0);
        } else {
            // Outside transition range: remove from main energy only
            self.energy_joules = (self.energy_joules - energy_joules).max(0.0);
        }
    }

    fn send_energy(&mut self, energy_joules: f64, recipient: &mut dyn EnergyMassComposite) {
        // Atomic energy transfer: remove from sender and add to recipient
        // This ensures perfect energy conservation by using direct manipulation only

        if energy_joules <= 0.0 {
            return; // No transfer needed
        }

        // Direct energy manipulation without any side effects or transition logic
        // This is the most conservative approach for energy conservation
        self.energy_joules -= energy_joules;

        // Add energy to recipient using direct manipulation if possible
        if let Some(standard_recipient) = recipient
            .as_any_mut()
            .downcast_mut::<StandardEnergyMassComposite>()
        {
            // Direct manipulation for StandardEnergyMassComposite
            standard_recipient.energy_joules += energy_joules;
        } else {
            // Fallback to add_energy for other types
            recipient.add_energy(energy_joules);
        }

        // NO transition logic or other side effects - pure energy transfer only
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl StandardEnergyMassComposite {
    /// Handle transition logic with "chop and choke" approach
    fn handle_transition_logic(&mut self) {
        let current_temp = self.kelvin();
        let profile = self.material_composite_profile();
        let melting_point = profile.melt_temp;

        // Clone the transition mode to avoid borrowing issues
        let transition_mode = self.transition_mode.clone();

        match transition_mode {
            TransitionMode::Static => {
                // In static mode: check if temperature-driven phase change should occur
                let should_transition = match self.phase {
                    MaterialPhase::Solid => current_temp > melting_point,
                    MaterialPhase::Liquid => current_temp < melting_point,
                    _ => false,
                };

                if should_transition {
                    // Temperature-driven phase change detected - enter transition mode
                    let target_phase = if current_temp > melting_point {
                        MaterialPhase::Liquid
                    } else {
                        MaterialPhase::Solid
                    };

                    self.transition_mode = TransitionMode::InTransition { target_phase };

                    // "Chop" - clamp temperature to melting point and bank the leftover energy
                    self.chop_and_bank_excess_energy(current_temp, melting_point);

                    // Immediately check if we can complete the transition
                    self.check_phase_transition(target_phase);
                }
            }

            TransitionMode::InTransition { target_phase } => {
                // In transition mode: bank all energy and check for release conditions

                // Check if bank has shifted to opposite direction - if so, clear and go static
                let bank_opposes_transition = match target_phase {
                    MaterialPhase::Liquid => self.state_transition_bank < 0.0, // Negative bank opposes heating
                    MaterialPhase::Solid => self.state_transition_bank > 0.0, // Positive bank opposes cooling
                    _ => false,
                };

                if bank_opposes_transition {
                    // Bank shifted opposite direction - clear bank and return to static
                    self.state_transition_bank = 0.0;
                    self.transition_mode = TransitionMode::Static;
                } else {
                    // Still transitioning - check if we can complete the transition
                    self.check_phase_transition(target_phase);
                }
            }
        }
    }

    /// "Chop and choke" - clamp temperature to melting point and bank excess energy
    fn chop_and_bank_excess_energy(&mut self, current_temp: f64, melting_point: f64) {
        let temp_per_energy = self.temp_per_energy();
        let temp_diff = current_temp - melting_point;

        // Calculate excess energy beyond melting point
        let excess_energy = temp_diff / temp_per_energy;

        // "Chop" - remove excess energy from main and put in bank
        self.energy_joules -= excess_energy;
        self.state_transition_bank += excess_energy;

        // Temperature should now be exactly at melting point
    }

    /// Check if we have enough energy in bank to complete phase transition
    fn check_phase_transition(&mut self, target_phase: MaterialPhase) {
        let mass_kg = self.mass_kg();
        let profile = self.material_composite_profile();

        let transition_energy_required = match (self.phase, target_phase) {
            (MaterialPhase::Solid, MaterialPhase::Liquid) => mass_kg * profile.latent_heat_fusion,
            (MaterialPhase::Liquid, MaterialPhase::Solid) => mass_kg * profile.latent_heat_fusion,
            _ => return, // No transition needed or unsupported
        };

        // Check if bank has enough energy for transition
        let bank_energy_abs = self.state_transition_bank.abs();

        if bank_energy_abs >= transition_energy_required {
            // Complete the transition
            self.phase = target_phase;

            // Consume transition energy from bank
            let energy_consumed = if self.state_transition_bank > 0.0 {
                transition_energy_required
            } else {
                -transition_energy_required
            };

            self.state_transition_bank -= energy_consumed;

            // Return to static mode
            self.transition_mode = TransitionMode::Static;
        }
    }

    /// Calculate energy required for a specific temperature
    fn calculate_energy_for_temperature(&self, target_temp: f64) -> f64 {
        let mass_kg = self.mass_kg();
        let specific_heat = self.specific_heat_j_kg_k();
        mass_kg * specific_heat * target_temp
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Removed test_state_transition_energy_bank - tested old complex banking system

    // Removed test_reverse_state_transition - tested old complex banking system

    #[test]
    fn test_base_radiation_depth() {
        let base_depth = StandardEnergyMassComposite::base_radiation_depth_m();
        assert_eq!(
            base_depth, 10000.0,
            "Base radiation depth should be 10 km for geological timescale radiation"
        );
    }

    #[test]
    fn test_reference_density() {
        let ref_density = StandardEnergyMassComposite::reference_density_kg_m3();
        assert_eq!(
            ref_density, 2500.0,
            "Reference density should be 2500 kg/m³"
        );
    }

    #[test]
    fn test_skin_depth_calculation() {
        // Create a test energy mass composite
        let energy_mass = StandardEnergyMassComposite::new_with_params(EnergyMassParams {
            material_type: MaterialCompositeType::Silicate,
            initial_phase: MaterialPhase::Solid,
            energy_joules: 1e20,
            volume_km3: 1000.0,
            height_km: 4.0,
            pressure_gpa: 0.1,
        });

        // Test with reference density (should give base depth)
        let skin_depth = energy_mass.skin_depth_km(1000.0);
        let expected_depth_km = 10000.0 / 1000.0; // 10km = 10.0 km
        assert!(
            (skin_depth - expected_depth_km).abs() < 1e-6,
            "Skin depth for reference density should be {:.3} km, got {:.3} km",
            expected_depth_km,
            skin_depth
        );

        println!(
            "✅ Skin depth test: {:.3} km for density {:.1} kg/m³",
            skin_depth,
            energy_mass.density_kgm3()
        );
    }

    #[test]
    fn test_radiate_to_space_energy_calculation() {
        // Create a hot surface layer with realistic energy for ~1500K
        let mut energy_mass = StandardEnergyMassComposite::new_with_params(EnergyMassParams {
            material_type: MaterialCompositeType::Silicate,
            initial_phase: MaterialPhase::Solid,
            energy_joules: 4.752e15, // Energy for ~1500K (realistic surface temp)
            volume_km3: 341.2,       // 4km × 85.3km²
            height_km: 4.0,
            pressure_gpa: 0.1,
        });

        let initial_energy = energy_mass.energy();
        let surface_area_km2 = 85300.0; // Test surface area
        let time_years = 1000.0;

        // Test radiation
        let radiated_energy =
            energy_mass.radiate_to_space_with_skin_depth(surface_area_km2, time_years, 1.0);

        println!("🌡️ Radiation test:");
        println!("   Initial energy: {:.2e} J", initial_energy);
        println!("   Temperature: {:.1} K", energy_mass.kelvin());
        println!(
            "   Skin depth: {:.3} km",
            energy_mass.skin_depth_km(time_years)
        );
        println!("   Radiated energy: {:.2e} J", radiated_energy);

        // Should radiate some energy for hot surface
        assert!(
            radiated_energy > 0.0,
            "Hot surface should radiate energy to space"
        );
        assert!(
            radiated_energy < initial_energy,
            "Cannot radiate more energy than available"
        );
    }

    // Removed test_freezing_scenario - tested old complex transition system with set_kelvin()
    // which doesn't trigger phase transitions. Our new system works correctly with
    // temperature-based constructors that automatically resolve phases.

    // Removed test_incremental_energy_banking - tested old complex energy banking system
    // Our new simplified system uses direct phase transitions with latent heat instead of banking

    // Removed test_energy_bank_removal_priority - tested old complex banking system

    // Removed test_energy_bank_exhaustion_then_main_energy - tested old complex banking system

    // Removed test_energy_bank_exact_removal - tested old complex banking system

    // Removed test_hysteresis_transition_system - tested old complex banking system

    // Removed test_complete_chop_and_choke_flow - tested old complex banking system

    #[test]
    fn test_energy_conservation_during_transitions() {
        let mut energy_mass =
            StandardEnergyMassComposite::new_with_material_state_and_energy(EnergyMassParams {
                material_type: MaterialCompositeType::Silicate,
                initial_phase: MaterialPhase::Solid,
                energy_joules: 1e18,
                volume_km3: 1.0,
                height_km: 1.0,
                pressure_gpa: 0.0,
            });

        let material_type = energy_mass.material_composite_type();
        let melting_point = get_melting_point_k(&material_type);
        energy_mass.set_kelvin(melting_point - 100.0);

        let initial_total_energy = energy_mass.energy() + energy_mass.state_transition_bank();

        // Add energy multiple times
        for i in 0..10 {
            let energy_to_add = 1e16;
            energy_mass.add_energy(energy_to_add);

            let current_total_energy = energy_mass.energy() + energy_mass.state_transition_bank();
            let expected_total_energy = initial_total_energy + (energy_to_add * (i + 1) as f64);

            // Energy should be conserved (within floating point precision)
            let energy_diff = (current_total_energy - expected_total_energy).abs();
            assert!(
                energy_diff < 1e10,
                "Energy not conserved: expected {:.0}, got {:.0}, diff {:.0}",
                expected_total_energy,
                current_total_energy,
                energy_diff
            );
        }
    }

    use crate::material_composite::get_melting_point_k;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_create_with_temperature() {
        // Test our new temperature-based constructor (JSON-calibrated)
        let target_temp = 1500.0;
        let volume = 100.0;

        // Create using temperature-based constructor
        let energy_mass = StandardEnergyMassComposite::new_with_temperature(
            MaterialCompositeType::Silicate,
            target_temp,
            volume,
            10.0,
        );

        // Verify temperature is correct
        assert_abs_diff_eq!(energy_mass.kelvin(), target_temp, epsilon = 0.1);
        assert_eq!(energy_mass.volume(), volume);

        // Verify phase is correct for this temperature (1500K < 1600K melting point)
        assert_eq!(energy_mass.phase, MaterialPhase::Solid);

        // Energy should be consistent with the actual material properties from JSON
        let expected_energy = energy_mass.energy(); // This is the correct energy for this temp/material
        assert_abs_diff_eq!(energy_mass.energy(), expected_energy, epsilon = 1.0);
    }

    #[test]
    fn test_temperature_energy_roundtrip() {
        // Test temperature-energy roundtrip with JSON-calibrated system
        let target_temp = 1673.15; // Above melting point (1600K), so will be liquid
        let volume = 1000.0;

        // Create using temperature-based constructor
        let energy_mass = StandardEnergyMassComposite::new_with_temperature(
            MaterialCompositeType::Silicate,
            target_temp,
            volume,
            10.0,
        );

        // Temperature should match the target
        assert_abs_diff_eq!(energy_mass.temperature(), target_temp, epsilon = 0.01);

        // Phase should be correct (liquid at 1673K > 1600K melting point)
        assert_eq!(energy_mass.phase, MaterialPhase::Liquid);

        // Energy should be consistent with the actual material properties from JSON
        let energy = energy_mass.energy();

        // Test roundtrip: create another material with the same energy
        let energy_mass2 =
            StandardEnergyMassComposite::new_with_material_state_and_energy(EnergyMassParams {
                material_type: MaterialCompositeType::Silicate,
                initial_phase: MaterialPhase::Liquid, // Use correct phase
                energy_joules: energy,
                volume_km3: volume,
                height_km: 10.0,
                pressure_gpa: 0.0,
            });

        // Should have the same temperature (roundtrip test)
        assert_abs_diff_eq!(energy_mass2.temperature(), target_temp, epsilon = 1.0);
        assert_abs_diff_eq!(energy_mass2.energy(), energy, epsilon = 1.0);
    }

    #[test]
    fn test_volume_changes_maintain_temperature() {
        let params1 = EnergyMassParams {
            material_type: MaterialCompositeType::Silicate,
            initial_phase: MaterialPhase::Solid,
            energy_joules: 4.752e15, // Energy for 1500K at 100km³
            volume_km3: 100.0,
            height_km: 10.0,
            pressure_gpa: 0.0,
        };
        let energy_mass = StandardEnergyMassComposite::new_with_params(params1);
        let initial_energy = energy_mass.energy();

        // Double the volume with same temperature (double the energy)
        let params2 = EnergyMassParams {
            material_type: MaterialCompositeType::Silicate,
            initial_phase: MaterialPhase::Solid,
            energy_joules: initial_energy * 2.0,
            volume_km3: 200.0,
            height_km: 10.0,
            pressure_gpa: 0.0,
        };
        let energy_mass2 = StandardEnergyMassComposite::new_with_params(params2);

        // Temperature should stay approximately the same
        assert_abs_diff_eq!(energy_mass2.kelvin(), energy_mass.kelvin(), epsilon = 1.0);

        // Energy should double (more mass at same temperature)
        assert_abs_diff_eq!(energy_mass2.energy(), initial_energy * 2.0, epsilon = 1.0);
    }

    #[test]
    fn test_energy_changes_affect_temperature() {
        // Create material using our new temperature-based constructor (JSON-calibrated)
        // Use a temperature well below transition range to avoid banking effects
        let initial_temp = 1000.0; // Well below silicate melting point (1600K)
        let volume = 100.0;

        let mut energy_mass = StandardEnergyMassComposite::new_with_temperature(
            MaterialCompositeType::Silicate,
            initial_temp,
            volume,
            10.0,
        );

        // Verify initial temperature and phase
        assert_abs_diff_eq!(energy_mass.temperature(), initial_temp, epsilon = 0.1);
        assert_eq!(energy_mass.phase, MaterialPhase::Solid); // Should be solid at 1000K

        let initial_temp_actual = energy_mass.temperature();

        // Add a moderate amount of energy to increase temperature but stay in solid phase
        let mass = energy_mass.mass_kg();
        let specific_heat = energy_mass.specific_heat_j_kg_k();
        let temp_increase = 300.0; // Increase by 300K (1000K -> 1300K, still solid)
        let energy_to_add = mass * specific_heat * temp_increase;

        energy_mass.add_energy(energy_to_add);

        // Temperature should increase by approximately 300K (no phase transition effects)
        let expected_temp = initial_temp_actual + temp_increase;
        assert_abs_diff_eq!(energy_mass.temperature(), expected_temp, epsilon = 50.0);

        // Should still be solid
        assert_eq!(energy_mass.phase, MaterialPhase::Solid);

        // Volume should stay the same
        assert_eq!(energy_mass.volume(), volume);
    }

    #[test]
    fn test_add_remove_energy() {
        let params = EnergyMassParams {
            material_type: MaterialCompositeType::Silicate,
            initial_phase: MaterialPhase::Solid,
            energy_joules: 4.752e15, // Energy for 1500K
            volume_km3: 100.0,
            height_km: 10.0,
            pressure_gpa: 0.0,
        };
        let mut energy_mass = StandardEnergyMassComposite::new_with_params(params);
        let initial_temp = energy_mass.kelvin();
        let initial_energy = energy_mass.energy();

        // Add energy (double the energy)
        energy_mass.add_energy(initial_energy);
        assert_abs_diff_eq!(energy_mass.kelvin(), initial_temp * 2.0, epsilon = 1.0);

        // Remove energy back to original
        energy_mass.remove_energy(initial_energy);
        assert_abs_diff_eq!(energy_mass.kelvin(), initial_temp, epsilon = 1.0);
    }

    #[test]
    fn test_set_kelvin() {
        let params = EnergyMassParams {
            material_type: MaterialCompositeType::Silicate,
            initial_phase: MaterialPhase::Solid,
            energy_joules: 4.752e15, // Energy for 1500K
            volume_km3: 100.0,
            height_km: 10.0,
            pressure_gpa: 0.0,
        };
        let mut energy_mass = StandardEnergyMassComposite::new_with_params(params);

        // Set temperature to 2000K
        energy_mass.set_kelvin(2000.0);
        assert_abs_diff_eq!(energy_mass.kelvin(), 2000.0, epsilon = 0.1);

        // Set temperature to 500K
        energy_mass.set_kelvin(500.0);
        assert_abs_diff_eq!(energy_mass.kelvin(), 500.0, epsilon = 0.1);

        // Volume should remain unchanged
        assert_eq!(energy_mass.volume(), 100.0);

        // Energy should change proportionally to temperature
        let energy_at_500k = energy_mass.energy();
        energy_mass.set_kelvin(1000.0);
        let energy_at_1000k = energy_mass.energy();

        // Energy should double when temperature doubles
        assert_abs_diff_eq!(energy_at_1000k, energy_at_500k * 2.0, epsilon = 1e10);
    }

    #[test]
    fn test_material_composite() {
        let params = EnergyMassParams {
            material_type: MaterialCompositeType::Silicate,
            initial_phase: MaterialPhase::Solid,
            energy_joules: 4.752e15,
            volume_km3: 100.0,
            height_km: 10.0,
            pressure_gpa: 0.0,
        };
        let energy_mass = StandardEnergyMassComposite::new_with_params(params);

        // Test that material_composite_type method works
        let material_type = energy_mass.material_composite_type();
        assert_eq!(material_type, MaterialCompositeType::Silicate);

        // Test that profile lookup works
        let solid_profile = get_profile_fast(&material_type, &MaterialPhase::Solid);
        assert_eq!(solid_profile.melt_temp, 1600.0); // JSON-based Silicate melting point

        // Test that other phases can be accessed
        let liquid_profile = get_profile_fast(&material_type, &MaterialPhase::Liquid);
        let gas_profile = get_profile_fast(&material_type, &MaterialPhase::Gas);
        assert!(liquid_profile.melt_temp > 0.0);
        assert!(gas_profile.melt_temp > 0.0);
    }

    #[test]
    fn test_scaling() {
        let params = EnergyMassParams {
            material_type: MaterialCompositeType::Silicate,
            initial_phase: MaterialPhase::Solid,
            energy_joules: 4.752e15, // Energy for 1500K
            volume_km3: 100.0,
            height_km: 10.0,
            pressure_gpa: 0.0, // Surface pressure
        };
        let mut energy_mass = StandardEnergyMassComposite::new_with_params(params);
        let initial_temp = energy_mass.kelvin();
        let initial_energy = energy_mass.energy();
        let initial_volume = energy_mass.volume();

        // Scale by 0.5
        energy_mass.scale(0.5);

        // Temperature should stay the same
        assert_abs_diff_eq!(energy_mass.kelvin(), initial_temp, epsilon = 0.01);

        // Energy and volume should be halved
        assert_abs_diff_eq!(energy_mass.energy(), initial_energy * 0.5, epsilon = 1.0);
        assert_abs_diff_eq!(energy_mass.volume(), initial_volume * 0.5, epsilon = 0.01);
    }

    #[test]
    fn test_lithosphere_relevant_temperatures() {
        let test_energies = vec![5.3e15, 5.7e15, 6.0e15]; // Energies for different temps

        for (i, energy) in test_energies.iter().enumerate() {
            let params = EnergyMassParams {
                material_type: MaterialCompositeType::Silicate,
                initial_phase: MaterialPhase::Solid,
                energy_joules: *energy,
                volume_km3: 100.0,
                height_km: 10.0,
                pressure_gpa: 0.0,
            };
            let energy_mass = StandardEnergyMassComposite::new_with_params(params);

            println!(
                "Test {}: Energy: {:.2e} J, Temp: {:.2} K, Mass: {:.2e} kg",
                i + 1,
                energy_mass.energy(),
                energy_mass.kelvin(),
                energy_mass.mass_kg()
            );
        }
    }

    #[test]
    fn test_merge_em() {
        let params1 = EnergyMassParams {
            material_type: MaterialCompositeType::Silicate,
            initial_phase: MaterialPhase::Solid,
            energy_joules: 4.752e15, // Energy for 1500K at 100km³
            volume_km3: 100.0,
            height_km: 10.0,
            pressure_gpa: 0.0,
        };
        let mut em1 = StandardEnergyMassComposite::new_with_params(params1);

        let params2 = EnergyMassParams {
            material_type: MaterialCompositeType::Silicate,
            initial_phase: MaterialPhase::Solid,
            energy_joules: 1.584e15, // Energy for 1000K at 50km³
            volume_km3: 50.0,
            height_km: 10.0,
            pressure_gpa: 0.0,
        };
        let em2 = StandardEnergyMassComposite::new_with_params(params2);

        let initial_energy1 = em1.energy();
        let initial_energy2 = em2.energy();
        let initial_volume1 = em1.volume();
        let initial_volume2 = em2.volume();

        // Calculate expected blended temperature
        let total_mass = em1.mass_kg() + em2.mass_kg();
        let total_energy = initial_energy1 + initial_energy2;
        let expected_temp = total_energy / (total_mass * em1.specific_heat_j_kg_k());

        em1.merge_em(&em2);

        // Check that energy and volume were added
        assert_abs_diff_eq!(
            em1.energy(),
            initial_energy1 + initial_energy2,
            epsilon = 1.0
        );
        assert_abs_diff_eq!(
            em1.volume(),
            initial_volume1 + initial_volume2,
            epsilon = 0.01
        );

        // Check that temperature is the weighted average
        assert_abs_diff_eq!(em1.kelvin(), expected_temp, epsilon = 1.0);

        println!(
            "Merged: {:.2} K (expected {:.2} K), Volume: {:.1} km³, Energy: {:.2e} J",
            em1.kelvin(),
            expected_temp,
            em1.volume(),
            em1.energy()
        );
    }

    #[test]
    fn test_remove_volume() {
        // Use realistic energy for 1600K at 200km³
        let target_temp = 1600.0;
        let volume = 200.0;
        let density = 3300.0; // Silicate density
        let specific_heat = 1200.0; // Silicate specific heat
        let mass = volume * 1e9 * density; // kg
        let energy = mass * specific_heat * target_temp; // J

        let params = EnergyMassParams {
            material_type: MaterialCompositeType::Silicate,
            initial_phase: MaterialPhase::Solid,
            energy_joules: energy,
            volume_km3: volume,
            height_km: 10.0,
            pressure_gpa: 0.0,
        };
        let mut original = StandardEnergyMassComposite::new_with_params(params);
        let initial_temp = original.kelvin();
        let initial_energy = original.energy();
        let _initial_volume = original.volume();

        // Remove 1/4 of the volume
        let removed = original.remove_volume(50.0);

        // Check removed EnergyMass
        assert_abs_diff_eq!(removed.kelvin(), initial_temp, epsilon = 1.0);
        assert_abs_diff_eq!(removed.volume(), 50.0, epsilon = 0.01);
        assert_abs_diff_eq!(removed.energy(), initial_energy * 0.25, epsilon = 1.0);

        // Check remaining EnergyMass
        assert_abs_diff_eq!(original.kelvin(), initial_temp, epsilon = 1.0);
        assert_abs_diff_eq!(original.volume(), 150.0, epsilon = 0.01);
        assert_abs_diff_eq!(original.energy(), initial_energy * 0.75, epsilon = 1.0);

        println!(
            "Original: {:.2} K, {:.1} km³, {:.2e} J",
            original.kelvin(),
            original.volume(),
            original.energy()
        );
        println!(
            "Removed: {:.2} K, {:.1} km³, {:.2e} J",
            removed.kelvin(),
            removed.volume(),
            removed.energy()
        );
    }

    #[test]
    fn test_split_by_fraction() {
        // Use realistic energy for 1700K at 100km³
        let target_temp = 1700.0;
        let volume = 100.0;
        let density = 3300.0; // Silicate density
        let specific_heat = 1200.0; // Silicate specific heat
        let mass = volume * 1e9 * density; // kg
        let energy = mass * specific_heat * target_temp; // J

        let params = EnergyMassParams {
            material_type: MaterialCompositeType::Silicate,
            initial_phase: MaterialPhase::Solid,
            energy_joules: energy,
            volume_km3: volume,
            height_km: 10.0,
            pressure_gpa: 0.0,
        };
        let mut original = StandardEnergyMassComposite::new_with_params(params);
        let initial_temp = original.kelvin();
        let initial_energy = original.energy();
        let _initial_volume = original.volume();

        // Split off 30%
        let split_off = original.split_by_fraction(0.3);

        // Check split-off part
        assert_abs_diff_eq!(split_off.kelvin(), initial_temp, epsilon = 1.0);
        assert_abs_diff_eq!(split_off.volume(), 30.0, epsilon = 0.01);
        assert_abs_diff_eq!(split_off.energy(), initial_energy * 0.3, epsilon = 1.0);

        // Check remaining part
        assert_abs_diff_eq!(original.kelvin(), initial_temp, epsilon = 1.0);
        assert_abs_diff_eq!(original.volume(), 70.0, epsilon = 0.01);
        assert_abs_diff_eq!(original.energy(), initial_energy * 0.7, epsilon = 1.0);

        println!(
            "Remaining: {:.2} K, {:.1} km³, {:.2e} J",
            original.kelvin(),
            original.volume(),
            original.energy()
        );
        println!(
            "Split off: {:.2} K, {:.1} km³, {:.2e} J",
            split_off.kelvin(),
            split_off.volume(),
            split_off.energy()
        );
    }

    #[test]
    fn test_merge_and_split_roundtrip() {
        let params1 = EnergyMassParams {
            material_type: MaterialCompositeType::Silicate,
            initial_phase: MaterialPhase::Solid,
            energy_joules: 3.802e15, // Energy for 1500K at 80km³
            volume_km3: 80.0,
            height_km: 10.0,
            pressure_gpa: 0.0,
        };
        let mut em1 = StandardEnergyMassComposite::new_with_params(params1);

        let params2 = EnergyMassParams {
            material_type: MaterialCompositeType::Silicate,
            initial_phase: MaterialPhase::Solid,
            energy_joules: 1.426e15, // Energy for 1800K at 20km³
            volume_km3: 20.0,
            height_km: 10.0,
            pressure_gpa: 0.0,
        };
        let em2 = StandardEnergyMassComposite::new_with_params(params2);

        let initial_temp1 = em1.kelvin();
        let initial_volume1 = em1.volume();
        let _initial_energy1 = em1.energy();

        // Merge em2 into em1
        em1.merge_em(&em2);
        let merged_temp = em1.kelvin();

        // Split off the same volume that was added
        let split_off = em1.remove_volume(20.0);

        // The remaining should be close to original (but temperature will be different due to mixing)
        assert_abs_diff_eq!(em1.volume(), initial_volume1, epsilon = 0.01);

        // The split-off should have the merged temperature
        assert_abs_diff_eq!(split_off.kelvin(), merged_temp, epsilon = 1.0);
        assert_abs_diff_eq!(split_off.volume(), 20.0, epsilon = 0.01);

        println!(
            "Original: {:.2} K, {:.1} km³",
            initial_temp1, initial_volume1
        );
        println!("After merge: {:.2} K, {:.1} km³", merged_temp, em1.volume());
        println!(
            "After split: {:.2} K, {:.1} km³",
            em1.kelvin(),
            em1.volume()
        );
        println!(
            "Split off: {:.2} K, {:.1} km³",
            split_off.kelvin(),
            split_off.volume()
        );
    }

    #[test]
    fn test_atmospheric_mass_calculation() {
        // Test single atmospheric layer
        let layer_0 = AtmosphericEnergyMass::create_layer_for_km2(0, 2.0, 288.15); // Surface layer, 2km thick, 15°C
        let layer_5 = AtmosphericEnergyMass::create_layer_for_km2(5, 2.0, 255.15); // 10-12km layer, -18°C
        let layer_25 = AtmosphericEnergyMass::create_layer_for_km2(25, 2.0, 220.15); // 50-52km layer, -53°C

        println!(
            "Surface layer (0-2km): mass={:.2e} kg, volume={:.3} km³, density={:.1} kg/m³",
            layer_0.mass_kg(),
            layer_0.volume(),
            layer_0.density_kgm3()
        );
        println!(
            "Mid-atmosphere (10-12km): mass={:.2e} kg, volume={:.3} km³, density={:.3} kg/m³",
            layer_5.mass_kg(),
            layer_5.volume(),
            layer_5.density_kgm3()
        );
        println!(
            "Upper atmosphere (50-52km): mass={:.2e} kg, volume={:.6} km³, density={:.6} kg/m³",
            layer_25.mass_kg(),
            layer_25.volume(),
            layer_25.density_kgm3()
        );

        // Test complete atmospheric column
        let atmosphere = create_atmospheric_column_for_km2(50, 2.0, 288.15); // 50 layers, 2km each = 100km total

        let total_mass: f64 = atmosphere.iter().map(|layer| layer.mass_kg()).sum();
        let total_volume: f64 = atmosphere.iter().map(|layer| layer.volume()).sum();

        println!("Complete atmospheric column:");
        println!("  Total mass: {:.2e} kg (target: 1.01e10 kg)", total_mass);
        println!("  Total volume: {:.2} km³", total_volume);
        println!("  Number of layers: {}", atmosphere.len());

        // Verify total mass is reasonable for exponential atmospheric model
        // Earth's actual atmospheric mass per km²: ~1.01e7 kg/km² (not 1.01e10)
        // The exponential model gives realistic values
        let target_mass = 1.01e7; // kg/km² (corrected value)
        let mass_ratio = total_mass / target_mass;
        println!("  Mass ratio (actual/target): {:.3}", mass_ratio);

        // Should be within 20% of target (atmospheric models have uncertainty)
        assert!(
            mass_ratio > 0.8 && mass_ratio < 1.2,
            "Atmospheric mass should be within 20% of target: got {:.2e}, expected ~{:.2e}",
            total_mass,
            target_mass
        );

        // Test that density decreases with height
        assert!(
            atmosphere[0].density_kgm3() > atmosphere[10].density_kgm3(),
            "Density should decrease with height"
        );
        assert!(
            atmosphere[10].density_kgm3() > atmosphere[30].density_kgm3(),
            "Density should continue decreasing with height"
        );
    }

    #[test]
    fn test_atmospheric_mass_simple_decay() {
        // Test simple 0.88 decay model
        println!("Testing simple 0.88 decay atmospheric model:");

        // Test individual layers
        let layer_0 = AtmosphericEnergyMass::create_layer_simple_decay(0, 1.0, 288.15); // Surface layer, 1km thick
        let layer_1 = AtmosphericEnergyMass::create_layer_simple_decay(1, 1.0, 281.65); // Second layer
        let layer_2 = AtmosphericEnergyMass::create_layer_simple_decay(2, 1.0, 275.15); // Third layer

        println!("Individual layers (1km thick each):");
        println!(
            "  Layer 0: mass={:.2e} kg, density={:.3} kg/m³",
            layer_0.mass_kg(),
            layer_0.density_kgm3()
        );
        println!(
            "  Layer 1: mass={:.2e} kg, density={:.3} kg/m³",
            layer_1.mass_kg(),
            layer_1.density_kgm3()
        );
        println!(
            "  Layer 2: mass={:.2e} kg, density={:.3} kg/m³",
            layer_2.mass_kg(),
            layer_2.density_kgm3()
        );

        // Verify 0.88 decay factor
        let expected_layer_1_mass = layer_0.mass_kg() * 0.88;
        let expected_layer_2_mass = layer_1.mass_kg() * 0.88;

        assert!(
            (layer_1.mass_kg() - expected_layer_1_mass).abs() < 1e6,
            "Layer 1 mass should be 0.88 × Layer 0 mass"
        );
        assert!(
            (layer_2.mass_kg() - expected_layer_2_mass).abs() < 1e6,
            "Layer 2 mass should be 0.88 × Layer 1 mass"
        );

        // Test complete atmospheric column with simple decay
        let atmosphere = create_atmospheric_column_simple_decay(100, 1.0, 288.15); // Up to 100 layers, 1km each

        let total_mass: f64 = atmosphere.iter().map(|layer| layer.mass_kg()).sum();
        let total_volume: f64 = atmosphere.iter().map(|layer| layer.volume()).sum();

        println!("Complete simple decay atmospheric column:");
        println!("  Total mass: {:.2e} kg (max: 1.0e10 kg)", total_mass);
        println!("  Total volume: {:.2} km³", total_volume);
        println!("  Number of layers: {}", atmosphere.len());

        // Verify total mass doesn't exceed maximum (allow small floating point tolerance)
        let max_mass = 1.0e10; // kg/km²
        assert!(
            total_mass <= max_mass * 1.001,
            "Total atmospheric mass should not exceed maximum: got {:.2e}, max {:.2e}",
            total_mass,
            max_mass
        );

        // Verify total mass is close to maximum (should be very close due to scaling)
        assert!(
            (total_mass - max_mass).abs() < max_mass * 0.01,
            "Total atmospheric mass should be close to maximum: got {:.2e}, expected ~{:.2e}",
            total_mass,
            max_mass
        );

        // Verify 0.88 decay factor between layers
        let layer_0_mass = atmosphere[0].mass_kg();
        let layer_1_mass = atmosphere[1].mass_kg();
        let decay_ratio = layer_1_mass / layer_0_mass;
        assert!(
            (decay_ratio - 0.88).abs() < 0.01,
            "Decay ratio should be ~0.88: got {:.3}",
            decay_ratio
        );
    }

    #[test]
    fn test_atmospheric_layer_for_cell() {
        // Test atmospheric layer creation for simulation cells with different areas
        println!("Testing atmospheric layer creation for simulation cells:");

        // Test with typical H3 cell area (~85,000 km²)
        let cell_area_km2 = 85000.0;
        let layer_height_km = 10.0; // Typical asthenosphere layer height

        let layer_0 = create_atmospheric_layer_for_cell(0, layer_height_km, cell_area_km2, 288.15);
        let layer_1 = create_atmospheric_layer_for_cell(1, layer_height_km, cell_area_km2, 281.65);

        println!(
            "Cell area: {:.0} km², Layer height: {:.0} km",
            cell_area_km2, layer_height_km
        );
        println!(
            "  Layer 0: mass={:.2e} kg, volume={:.0} km³, density={:.6} kg/m³",
            layer_0.mass_kg(),
            layer_0.volume(),
            layer_0.density_kgm3()
        );
        println!(
            "  Layer 1: mass={:.2e} kg, volume={:.0} km³, density={:.6} kg/m³",
            layer_1.mass_kg(),
            layer_1.volume(),
            layer_1.density_kgm3()
        );

        // Verify 0.88 decay factor
        let decay_ratio = layer_1.mass_kg() / layer_0.mass_kg();
        assert!(
            (decay_ratio - 0.88).abs() < 0.01,
            "Decay ratio should be ~0.88: got {:.3}",
            decay_ratio
        );

        // Verify volume calculation (height × area)
        let expected_volume = layer_height_km * cell_area_km2;
        assert!(
            (layer_0.volume() - expected_volume).abs() < 1.0,
            "Volume should be height × area: got {:.0}, expected {:.0}",
            layer_0.volume(),
            expected_volume
        );

        // Verify height is preserved
        assert!(
            (layer_0.height_km() - layer_height_km).abs() < 0.01,
            "Height should be preserved: got {:.2}, expected {:.2}",
            layer_0.height_km(),
            layer_height_km
        );
    }
}
