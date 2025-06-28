#![allow(dead_code)]
#![allow(unused_variables)]

use std::fs::File;
use std::io::Write;
use std::collections::VecDeque;

// Import the real EnergyMass trait and material system
extern crate atmo_asth_rust;
use atmo_asth_rust::energy_mass::{EnergyMass, StandardEnergyMass};
use atmo_asth_rust::material::MaterialType;

/// Configuration parameters for thermal diffusion experiments
#[derive(Clone, Debug)]
struct ExperimentState {
    // Thermal parameters
    pub melting_point_k: f64,
    pub cooling_point_k: f64,
    pub outgassing_threshold_k: f64,

    // Transition rates (base rates per year)
    pub base_melting_rate: f64,      // Base rate for solid -> liquid transition
    pub base_cooling_rate: f64,      // Base rate for liquid -> solid transition
    pub melting_temp_factor: f64,    // Temperature sensitivity for melting
    pub cooling_temp_factor: f64,    // Temperature sensitivity for cooling

    // Thermal conductivity
    pub solid_conductivity: f64,
    pub liquid_conductivity: f64,
    pub transition_conductivity: f64,

    // Diffusion parameters
    pub conductivity_factor: f64,
    pub distance_length: f64,
    pub pressure_baseline: f64,
    pub max_change_rate: f64,        // Maximum energy change rate per step

    // Boundary conditions
    pub foundry_temperature_k: f64,
    pub surface_temperature_k: f64,
    pub core_heat_input: f64,        // J per year
    pub surface_radiation_factor: f64,

    // Outgassing parameters
    pub outgassing_rate_multiplier: f64,
    pub outgassing_energy_fraction: f64,

    // Time parameters
    pub years_per_step: f64,
    pub total_years: f64,
}

impl ExperimentState {
    /// Create default experiment configuration
    fn default() -> Self {
        Self {
            // Thermal parameters
            melting_point_k: 1523.0,
            cooling_point_k: 1473.0,
            outgassing_threshold_k: 1400.0,

            // Transition rates (faster for testing, realistic geological time scales)
            base_melting_rate: 1.0 / 50.0,     // 50 years to melt at melting point (faster for testing)
            base_cooling_rate: 1.0 / 100.0,    // 100 years to solidify at cooling point (faster for testing)
            melting_temp_factor: 1.0 / 100.0,  // +1% rate per 100K excess
            cooling_temp_factor: 1.0 / 200.0,  // +0.5% rate per 200K deficit

            // Thermal conductivity
            solid_conductivity: 2.5,
            liquid_conductivity: 3.2,
            transition_conductivity: 2.8,

            // Diffusion parameters
            conductivity_factor: 3.0,
            distance_length: 4.0,
            pressure_baseline: 1.0,
            max_change_rate: 0.02,  // 2% max change per step

            // Boundary conditions (realistic for million-year geological simulations)
            foundry_temperature_k: 1800.0,  // Just above melting point for realistic asthenosphere
            surface_temperature_k: 300.0,   // Earth surface temperature
            core_heat_input: 2.52e12,       // J per km² per year
            surface_radiation_factor: 1.0,

            // Outgassing parameters
            outgassing_rate_multiplier: 0.001,
            outgassing_energy_fraction: 0.1,

            // Time parameters
            years_per_step: 100.0,
            total_years: 1_000_000.0,
        }
    }

    /// Create high-speed experiment configuration for testing
    fn fast_test() -> Self {
        let mut config = Self::default();
        config.years_per_step = 1000.0;  // Larger time steps
        config.total_years = 100_000.0;  // Shorter simulation
        config.base_melting_rate *= 10.0;  // Faster transitions
        config.base_cooling_rate *= 10.0;
        config.max_change_rate = 0.05;   // Allow larger changes
        config
    }

    /// Create slow, realistic experiment configuration
    fn realistic() -> Self {
        let mut config = Self::default();
        config.years_per_step = 50.0;    // Smaller time steps
        config.total_years = 10_000_000.0; // Longer simulation
        config.base_melting_rate *= 0.1;  // Slower transitions
        config.base_cooling_rate *= 0.1;
        config.max_change_rate = 0.01;   // Smaller changes
        config
    }
}

/// Pending update for a thermal cell
#[derive(Clone, Debug)]
struct PendingUpdate {
    index: usize,
    energy_j: f64,
    volume_km3: f64,
    material_type: MaterialType,
    density_factor: f64,  // For atmosphere/magma variable density
}

impl PendingUpdate {
    fn new(index: usize, energy_j: f64, volume_km3: f64, material_type: MaterialType) -> Self {
        Self {
            index,
            energy_j,
            volume_km3,
            material_type,
            density_factor: 1.0,  // Default full density
        }
    }

    fn with_density_factor(mut self, density_factor: f64) -> Self {
        self.density_factor = density_factor.clamp(0.0, 1.0);
        self
    }
}

/// Queue for managing pending changes to thermal cells
#[derive(Debug)]
struct PendingChanges {
    updates: VecDeque<PendingUpdate>,
}

impl PendingChanges {
    fn new() -> Self {
        Self {
            updates: VecDeque::new(),
        }
    }

    fn add_update(&mut self, update: PendingUpdate) {
        self.updates.push_back(update);
    }

    fn add_energy_update(&mut self, index: usize, energy_j: f64, nodes: &[ThermalNode]) {
        if index < nodes.len() {
            let node = &nodes[index];
            let update = PendingUpdate::new(
                index,
                energy_j,
                node.volume_km3(),
                node.material_type(),
            );
            self.add_update(update);
        }
    }

    fn add_outgassing_update(&mut self, from_index: usize, to_atmosphere: bool, volume_km3: f64, nodes: &[ThermalNode]) {
        if from_index < nodes.len() {
            let source_node = &nodes[from_index];

            // Remove volume from source
            let remaining_volume = source_node.volume_km3() - volume_km3;
            let source_update = PendingUpdate::new(
                from_index,
                source_node.energy_j(),
                remaining_volume,
                source_node.material_type(),
            );
            self.add_update(source_update);

            // Add to atmosphere (simplified - would need proper atmosphere indexing)
            if to_atmosphere {
                // Create atmosphere layer update
                let atmosphere_update = PendingUpdate::new(
                    0, // Insert at top
                    source_node.energy_j() * (volume_km3 / source_node.volume_km3()),
                    volume_km3,
                    MaterialType::Icy, // Use Icy as atmosphere placeholder
                ).with_density_factor(0.1); // Low density for atmosphere

                self.add_update(atmosphere_update);
            }
        }
    }

    fn drain_updates(&mut self) -> Vec<PendingUpdate> {
        self.updates.drain(..).collect()
    }

    fn is_empty(&self) -> bool {
        self.updates.is_empty()
    }

    fn len(&self) -> usize {
        self.updates.len()
    }
}

/// Experimental EnergyMass implementation for thermal experiments
/// Adds max_density_factor for atmosphere and magma with variable density
/// Adds state field for thermal state tracking
#[derive(Clone, Debug)]
struct ThermalEnergyMassExp {
    inner: StandardEnergyMass,
    max_density_factor: f64,  // 0.0-1.0: fraction of material's max density
    state: u8,                // 1-4=solid, 5=cooling, 6=heating, 10=liquid, 11-15=atmosphere
}

impl ThermalEnergyMassExp {
    fn new(material_type: MaterialType, temperature_k: f64, volume_km3: f64, height_km: f64, state: u8) -> Self {
        let inner = StandardEnergyMass::new_with_material_and_height(
            material_type,
            temperature_k,
            volume_km3,
            height_km,
        );
        Self {
            inner,
            max_density_factor: 1.0,  // Default: use full material density
            state,
        }
    }

    fn new_with_density_factor(
        material_type: MaterialType,
        temperature_k: f64,
        volume_km3: f64,
        height_km: f64,
        max_density_factor: f64,
        state: u8
    ) -> Self {
        let inner = StandardEnergyMass::new_with_material_and_height(
            material_type,
            temperature_k,
            volume_km3,
            height_km,
        );
        Self {
            inner,
            max_density_factor: max_density_factor.clamp(0.0, 1.0),
            state,
        }
    }

    fn new_silicate(temperature_k: f64, volume_km3: f64, height_km: f64, state: u8) -> Self {
        Self::new(MaterialType::Silicate, temperature_k, volume_km3, height_km, state)
    }

    fn new_air(temperature_k: f64, volume_km3: f64, height_km: f64, state: u8) -> Self {
        // Use Icy as a placeholder for Air since Air doesn't exist in MaterialType
        Self::new(MaterialType::Icy, temperature_k, volume_km3, height_km, state)
    }

    fn new_air_with_density(temperature_k: f64, volume_km3: f64, height_km: f64, density_factor: f64, state: u8) -> Self {
        Self::new_with_density_factor(MaterialType::Icy, temperature_k, volume_km3, height_km, density_factor, state)
    }

    fn new_magma_with_density(temperature_k: f64, volume_km3: f64, height_km: f64, density_factor: f64, state: u8) -> Self {
        Self::new_with_density_factor(MaterialType::Silicate, temperature_k, volume_km3, height_km, density_factor, state)
    }

    /// Get the effective density (material density * max_density_factor)
    fn effective_density_kgm3(&self) -> f64 {
        self.inner.density_kgm3() * self.max_density_factor
    }

    /// Get the effective mass using the density factor
    fn effective_mass_kg(&self) -> f64 {
        const KM3_TO_M3: f64 = 1.0e9;
        let volume_m3 = self.inner.volume() * KM3_TO_M3;
        volume_m3 * self.effective_density_kgm3()
    }

    /// Set the density factor (for atmosphere pressure changes, magma expansion, etc.)
    fn set_density_factor(&mut self, factor: f64) {
        self.max_density_factor = factor.clamp(0.0, 1.0);
    }

    /// Get the current thermal state
    fn state(&self) -> u8 {
        self.state
    }

    /// Set the thermal state
    fn set_state(&mut self, new_state: u8) {
        self.state = new_state;
    }

    /// Update thermal state based on temperature with rate-dependent transitions
    /// time_years: time step for calculating transition progress
    /// config: experiment configuration with thermal parameters
    fn update_state_from_temperature(&mut self, time_years: f64, config: &ExperimentState) {
        let temp = self.kelvin();

        match self.state {
            1..=4 => {
                // Solid states - check for melting
                if temp > config.melting_point_k {
                    let temp_excess = temp - config.melting_point_k;
                    let transition_rate = self.calculate_melting_rate(temp_excess, config);

                    if self.should_transition(transition_rate, time_years) {
                        self.state = 6; // Start heating transition
                    }
                }
            },
            5 => {
                // Cooling transition state
                if temp < config.cooling_point_k {
                    // Cool enough to become solid
                    let temp_deficit = config.cooling_point_k - temp;
                    let transition_rate = self.calculate_cooling_rate(temp_deficit, config);

                    if self.should_transition(transition_rate, time_years) {
                        self.state = 4; // Fully solid
                    }
                } else if temp > config.melting_point_k {
                    // Temperature rising again - back to liquid
                    self.state = 10; // Switch back to liquid
                }
                // Stay in cooling state if between cooling_point and melting_point
            },
            6 => {
                // Heating transition state
                let temp_excess = temp - config.melting_point_k;
                if temp_excess > 0.0 {
                    let transition_rate = self.calculate_melting_rate(temp_excess, config);

                    if self.should_transition(transition_rate, time_years) {
                        self.state = 10; // Fully liquid
                    }
                } else if temp < config.cooling_point_k {
                    // Temperature dropping again
                    self.state = 5; // Switch to cooling transition
                }
            },
            10 => {
                // Liquid state - check for solidification
                if temp < config.melting_point_k {
                    // Below melting point - start cooling transition
                    let temp_deficit = config.melting_point_k - temp;
                    let transition_rate = self.calculate_cooling_rate(temp_deficit, config);
                    let transition_probability = transition_rate * time_years;

                    // Debug output for first few transitions
                    if temp_deficit < 10.0 && transition_probability > 0.5 {
                        println!("   🔄 Liquid->Cooling: {:.1}K (deficit {:.1}K), rate {:.4}, prob {:.2}",
                                temp, temp_deficit, transition_rate, transition_probability);
                    }

                    if self.should_transition(transition_rate, time_years) {
                        self.state = 5; // Start cooling transition
                        println!("   ✅ Transitioned to cooling state at {:.1}K", temp);
                    }
                }
            },
            11..=15 => {
                // Atmosphere states - no temperature-based transitions
            },
            _ => {
                // Unknown state - default to solid or liquid based on temperature
                self.state = if temp > config.melting_point_k { 10 } else { 4 };
            }
        }
    }

    /// Calculate melting rate based on temperature excess above melting point
    fn calculate_melting_rate(&self, temp_excess_k: f64, config: &ExperimentState) -> f64 {
        let temp_factor = 1.0 + (temp_excess_k * config.melting_temp_factor);
        config.base_melting_rate * temp_factor
    }

    /// Calculate cooling rate based on temperature deficit below cooling point
    fn calculate_cooling_rate(&self, temp_deficit_k: f64, config: &ExperimentState) -> f64 {
        let temp_factor = 1.0 + (temp_deficit_k * config.cooling_temp_factor);
        config.base_cooling_rate * temp_factor
    }

    /// Determine if transition should occur based on rate and time
    fn should_transition(&self, rate_per_year: f64, time_years: f64) -> bool {
        let transition_probability = rate_per_year * time_years;

        // For simplicity, use deterministic threshold
        // In a full simulation, you might use random probability
        transition_probability >= 1.0
    }

    /// Check if this cell can outgas (liquid state + hot enough)
    fn can_outgas(&self, config: &ExperimentState) -> bool {
        self.state == 10 && self.kelvin() > config.outgassing_threshold_k
    }

    /// Check if this is an atmosphere cell
    fn is_atmosphere(&self) -> bool {
        self.state >= 11 && self.state <= 15
    }

    /// Check if this is a liquid cell
    fn is_liquid(&self) -> bool {
        self.state == 10
    }

    /// Check if this is a solid cell
    fn is_solid(&self) -> bool {
        self.state >= 1 && self.state <= 4
    }
}

// Delegate EnergyMass methods to inner StandardEnergyMass, with density factor overrides
impl EnergyMass for ThermalEnergyMassExp {
    fn kelvin(&self) -> f64 { self.inner.kelvin() }
    fn energy(&self) -> f64 { self.inner.energy() }
    fn volume(&self) -> f64 { self.inner.volume() }
    fn height_km(&self) -> f64 { self.inner.height_km() }
    fn set_kelvin(&mut self, kelvin: f64) { self.inner.set_kelvin(kelvin) }

    // Override mass and density to use the density factor
    fn mass_kg(&self) -> f64 { self.effective_mass_kg() }
    fn density_kgm3(&self) -> f64 { self.effective_density_kgm3() }

    // Delegate other material properties
    fn specific_heat_j_kg_k(&self) -> f64 { self.inner.specific_heat_j_kg_k() }
    fn thermal_conductivity(&self) -> f64 { self.inner.thermal_conductivity() }
    fn material_type(&self) -> MaterialType { self.inner.material_type() }
    fn material_profile(&self) -> &'static atmo_asth_rust::material::MaterialProfile { self.inner.material_profile() }

    // Delegate operations
    fn scale(&mut self, factor: f64) { self.inner.scale(factor) }
    fn remove_heat(&mut self, heat_joules: f64) { self.inner.remove_heat(heat_joules) }
    fn add_energy(&mut self, energy_joules: f64) { self.inner.add_energy(energy_joules) }

    // Delegate thermal operations
    fn radiate_to(&mut self, other: &mut dyn EnergyMass, distance_km: f64, area_km2: f64, time_years: f64) -> f64 {
        self.inner.radiate_to(other, distance_km, area_km2, time_years)
    }
    fn radiate_to_space(&mut self, area_km2: f64, time_years: f64) -> f64 {
        self.inner.radiate_to_space(area_km2, time_years)
    }
    fn radiate_to_space_with_skin_depth(&mut self, area_km2: f64, time_years: f64, energy_throttle: f64) -> f64 {
        self.inner.radiate_to_space_with_skin_depth(area_km2, time_years, energy_throttle)
    }
    fn skin_depth_km(&self, time_years: f64) -> f64 { self.inner.skin_depth_km(time_years) }

    // Missing trait methods
    fn thermal_transmission_r0(&self) -> f64 {
        // Use material profile's thermal transmission value
        let profile = self.material_profile();
        (profile.thermal_transmission_r0_min + profile.thermal_transmission_r0_max) / 2.0
    }

    fn split_by_fraction(&mut self, fraction: f64) -> Box<dyn EnergyMass> {
        // Split by volume fraction
        let volume_to_remove = self.volume() * fraction;
        self.remove_volume(volume_to_remove)
    }

    // Delegate volume operations
    fn remove_volume_internal(&mut self, volume_to_remove: f64) { self.inner.remove_volume_internal(volume_to_remove) }
    fn merge_em(&mut self, other: &dyn EnergyMass) { self.inner.merge_em(other) }
    fn remove_volume(&mut self, volume_to_remove: f64) -> Box<dyn EnergyMass> {
        // When removing volume, preserve the density factor and state
        let removed = self.inner.remove_volume(volume_to_remove);

        // Wrap the removed StandardEnergyMass in our experimental wrapper
        // Note: This is a simplified approach - in practice you'd want to handle this more carefully
        Box::new(ThermalEnergyMassExp {
            inner: StandardEnergyMass::new_with_material_and_height(
                removed.material_type(),
                removed.kelvin(),
                removed.volume(),
                removed.height_km(),
            ),
            max_density_factor: self.max_density_factor,
            state: self.state, // Preserve the state
        })
    }
}

/// Force-directed thermal diffusion experiment with boundary conditions
/// Each layer equalizes with neighbors + heat source (core) + heat sink (space)
fn main() {
    println!("🔬 Force-Directed Thermal Diffusion with Optimized Parameters");
    println!("==============================================================");
    println!("🎯 Deep geological time: 1 million years of thermal evolution");

    // Create experiment with scientifically optimized parameters
    let mut experiment = ForceDirectedThermal::new();
    experiment.create_chunky_temperature_profile();

    println!("📊 Initial Setup:");
    println!("   {} thermal nodes", experiment.nodes.len());
    println!("   🔥 HEAT SOURCE: Core heat input at bottom layer");
    println!("   ❄️  HEAT SINK: Space radiation cooling at top layer");

    experiment.print_initial_profile();

    // Run force-directed thermal equilibration
    println!("\n🌡️  Running scientifically calibrated thermal equilibration...");

    let years_per_step = 1000.0;   // 1,000-year steps (10x larger than original 100)
    let total_years = 1000000000.0; // 1 billion years total
    let steps = (total_years / years_per_step) as usize;

    experiment.run_force_directed_diffusion(steps, years_per_step);

    println!("\n📈 Results exported to examples/data/thermal_experiment.csv");
    println!("   Tracking 5 key depths: 52.5km, 77.5km, 102.5km, 127.5km, 152.5km");
    println!("   Columns: temp_XXkm (temperatures) + state_XXkm (thermal states)");
    println!("   1 billion years with 1k-year steps for deep geological time!");
}

#[derive(Debug, Clone)]
struct ThermalNode {
    index: usize,              // Cell's index in the array
    depth_km: f64,
    height_km: f64,
    energy_mass: ThermalEnergyMassExp,  // Contains state internally
}

impl ThermalNode {
    fn new(index: usize, depth_km: f64, height_km: f64, energy_mass: ThermalEnergyMassExp) -> Self {
        Self {
            index,
            depth_km,
            height_km,
            energy_mass,
        }
    }

    // Convenience getters that delegate to EnergyMass
    fn temperature_k(&self) -> f64 {
        let temp = self.energy_mass.kelvin();

        // Debug: Check for invalid temperatures
        if !temp.is_finite() || temp < 0.0 {
            eprintln!("⚠️  Invalid temperature: {}", temp);
            return 273.15; // Return room temperature as fallback
        }

        temp
    }
    fn energy_j(&self) -> f64 { self.energy_mass.energy() }
    fn volume_km3(&self) -> f64 { self.energy_mass.volume() }
    fn density_kg_m3(&self) -> f64 { self.energy_mass.density_kgm3() }
    fn thermal_capacity(&self) -> f64 {
        let mass = self.energy_mass.mass_kg();
        let specific_heat = self.energy_mass.specific_heat_j_kg_k();
        let capacity = mass * specific_heat;

        // Debug: Check for invalid values
        if capacity <= 0.0 || !capacity.is_finite() {
            eprintln!("⚠️  Invalid thermal capacity: mass={}, specific_heat={}, capacity={}",
                     mass, specific_heat, capacity);
            return 1.0; // Return a small positive value to avoid division by zero
        }

        capacity
    }
    fn thermal_conductivity(&self) -> f64 { self.energy_mass.thermal_conductivity() }
    fn material_type(&self) -> MaterialType { self.energy_mass.material_type() }

    // Convenience setters
    fn set_temperature(&mut self, temp_k: f64) {
        self.energy_mass.set_kelvin(temp_k);
    }

    fn set_energy(&mut self, energy_j: f64) {
        // Calculate what temperature this energy represents
        let mass_kg = self.energy_mass.mass_kg();
        let specific_heat = self.energy_mass.specific_heat_j_kg_k();
        if mass_kg > 0.0 && specific_heat > 0.0 {
            let temp_k = energy_j / (mass_kg * specific_heat);
            self.energy_mass.set_kelvin(temp_k);
        }
    }

    // Access to density factor for atmosphere/magma
    fn set_density_factor(&mut self, factor: f64) {
        self.energy_mass.set_density_factor(factor);
    }

    fn effective_density(&self) -> f64 {
        self.energy_mass.effective_density_kgm3()
    }

    // Thermal state methods (delegate to EnergyMass)
    fn thermal_state(&self) -> u8 {
        self.energy_mass.state()
    }

    fn set_thermal_state(&mut self, state: u8) {
        self.energy_mass.set_state(state);
    }

    fn update_thermal_state(&mut self, time_years: f64, config: &ExperimentState) {
        self.energy_mass.update_state_from_temperature(time_years, config);
    }

    fn can_outgas(&self, config: &ExperimentState) -> bool {
        self.energy_mass.can_outgas(config)
    }

    fn is_atmosphere(&self) -> bool {
        self.energy_mass.is_atmosphere()
    }

    fn is_liquid(&self) -> bool {
        self.energy_mass.is_liquid()
    }

    fn is_solid(&self) -> bool {
        self.energy_mass.is_solid()
    }

    // Methods for creating pending updates
    fn create_energy_update(&self, new_energy_j: f64) -> PendingUpdate {
        PendingUpdate::new(
            self.index,
            new_energy_j,
            self.volume_km3(),
            self.material_type(),
        )
    }

    fn create_volume_update(&self, new_volume_km3: f64) -> PendingUpdate {
        PendingUpdate::new(
            self.index,
            self.energy_j(),
            new_volume_km3,
            self.material_type(),
        )
    }

    fn create_outgassing_update(&self, outgassed_volume: f64) -> PendingUpdate {
        let remaining_volume = self.volume_km3() - outgassed_volume;
        let remaining_energy = self.energy_j() * (remaining_volume / self.volume_km3());

        PendingUpdate::new(
            self.index,
            remaining_energy,
            remaining_volume,
            self.material_type(),
        )
    }

    fn create_atmosphere_from_outgassing(&self, outgassed_volume: f64, atmosphere_index: usize) -> PendingUpdate {
        let outgassed_energy = self.energy_j() * (outgassed_volume / self.volume_km3());

        PendingUpdate::new(
            atmosphere_index,
            outgassed_energy,
            outgassed_volume,
            MaterialType::Icy, // Use Icy as atmosphere placeholder
        ).with_density_factor(0.1) // Low density for atmosphere
    }

    // Calculate thermal interactions with neighbors and create updates
    fn calculate_thermal_updates(&self, nodes: &[ThermalNode], years: f64) -> Vec<PendingUpdate> {
        let mut updates = Vec::new();

        // Calculate energy changes from thermal diffusion
        let mut total_energy_change = 0.0;

        // Check neighbors (simplified - just immediate neighbors)
        for neighbor_offset in [-1i32, 1i32] {
            let neighbor_index = self.index as i32 + neighbor_offset;
            if neighbor_index >= 0 && (neighbor_index as usize) < nodes.len() {
                let neighbor = &nodes[neighbor_index as usize];
                let energy_transfer = self.calculate_energy_transfer_to(neighbor, years);
                total_energy_change -= energy_transfer; // Energy flowing out
            }
        }

        // Create energy update if there's a significant change
        if total_energy_change.abs() > 1.0 {
            let new_energy = (self.energy_j() + total_energy_change).max(0.0);
            updates.push(self.create_energy_update(new_energy));
        }

        // Check for outgassing if hot enough (liquid state)
        if self.temperature_k() > 1400.0 && self.thermal_state() == 10 {
            let outgassing_rate = self.calculate_outgassing_rate(years);
            if outgassing_rate > 0.001 {
                updates.push(self.create_outgassing_update(outgassing_rate));
                // Would also create atmosphere update here
            }
        }

        updates
    }

    fn calculate_energy_transfer_to(&self, other: &ThermalNode, years: f64) -> f64 {
        let temp_diff = other.temperature_k() - self.temperature_k();
        if temp_diff.abs() < 0.1 { return 0.0; }

        let distance_km = (other.depth_km - self.depth_km).abs();
        let conductivity = (self.thermal_conductivity() + other.thermal_conductivity()) / 2.0;
        let area_km2 = 1.0; // 1 km² cross-sectional area

        // Simplified thermal diffusion
        let base_coefficient = 0.00004; // Energy transfer coefficient
        let energy_difference = temp_diff * self.thermal_capacity();
        let flow_coefficient = base_coefficient * conductivity / distance_km.max(0.1);
        let energy_transfer = energy_difference * flow_coefficient * years;

        // Limit transfer to prevent instability
        let max_transfer = self.thermal_capacity() * 0.25;
        energy_transfer.clamp(-max_transfer, max_transfer)
    }

    fn calculate_outgassing_rate(&self, years: f64) -> f64 {
        let temp_factor = ((self.temperature_k() - 1400.0) / 600.0).max(0.0);
        let base_rate = 0.001; // 0.1% volume per step
        let outgassing_rate = base_rate * temp_factor.exp() * years / 1000.0;

        // Limit to available volume
        (outgassing_rate * self.volume_km3()).min(self.volume_km3() * 0.05)
    }
}

struct ForceDirectedThermal {
    nodes: Vec<ThermalNode>,
    initial_nodes: Vec<ThermalNode>,
    pending_changes: PendingChanges,
    // Tunable parameters
    conductivity_factor: f64,
    distance_length: f64,
    pressure_baseline: f64,
}

impl ForceDirectedThermal {
    fn new() -> Self {
        Self {
            nodes: Vec::new(),
            initial_nodes: Vec::new(),
            pending_changes: PendingChanges::new(),
            conductivity_factor: 0.058,  // Realistic geological timescales
            distance_length: 20.0,       // Optimal diffusion range
            pressure_baseline: 100.0,    // Realistic thermal pressure
        }
    }

    fn new_with_params(conductivity_factor: f64, distance_length: f64, pressure_baseline: f64) -> Self {
        Self {
            nodes: Vec::new(),
            initial_nodes: Vec::new(),
            pending_changes: PendingChanges::new(),
            conductivity_factor,
            distance_length,
            pressure_baseline,
        }
    }
    
    fn create_chunky_temperature_profile(&mut self) {
        // Create DRAMATIC temperature split to clearly see smoothing
        // First half: COLD (500K), Second half: HOT (1500K)

        let num_layers = 60;  // Extended to 60 layers for deeper thermal column
        let total_depth = 300.0; // Extended to 300km for more asthenosphere space
        let layer_thickness = total_depth / num_layers as f64;

        for i in 0..num_layers {
            let depth = (i as f64 + 0.5) * layer_thickness; // Center of each layer

            // REALISTIC GEOTHERMAL GRADIENT: Using codebase values
            let depth = (i as f64 + 0.5) * layer_thickness;
            let geothermal_gradient_k_per_km = 25.0; // More realistic than 0.25 for this experiment
            let surface_temp_k = 288.0; // 15°C surface temperature
            let foundry_temp_k = 1800.0; // FOUNDRY: Realistic temperature just above melting point
            let asthenosphere_surface_temp_k = 1873.0; // Engine-asthenosphere boundary

            let temp = if i < 4 {
                // CORE/FOUNDRY (Fixed temperatures) - 4 layers with realistic heat
                let gradient = (foundry_temp_k - 1523.0) / 4.0; // From foundry temp to melting point
                foundry_temp_k - (i as f64 * gradient) // 1800K down to 1523K (realistic gradient)
            } else if i >= 52 {
                // SURFACE LAYERS (No space sink) - natural thermal gradient
                400.0 - ((i - 52) as f64 * 40.0) // 400K down to 80K natural cooling
            } else {
                // ALL LAYERS START HOT (molten planet genesis)
                // Let space cooling naturally create lithosphere from top down
                let depth_factor = (i as f64 - 4.0) / 56.0; // Normalize from engine to surface
                1800.0 + (depth_factor * 400.0) // 1800K up to 2200K (HOTTER as we go deeper toward foundry)
            };

            // REALISTIC material-based thermal conductivity and states
            let (conductivity, thermal_state) = if i < 8 {
                (3.2, 10) // CORE/ASTHENOSPHERE: Liquid state (8 layers)
            } else if i >= 32 {
                (10.0, 4)  // SURFACE: High conductivity for strong heat sink, solid state (8 layers)
            } else {
                (2.5, 10)  // LITHOSPHERE: Start as liquid (will transition to solid, 24 layers)
            };
            let density = 3000.0; // kg/m³
            let specific_heat = 1200.0; // J/kg/K
            let volume_km3 = layer_thickness * 1.0; // layer_thickness_km * 1km² area = volume in km³
            let volume_m3 = volume_km3 * 1e9; // Convert km³ to m³
            let mass_kg = density * volume_m3;
            let thermal_capacity = mass_kg * specific_heat;
            let energy_j = thermal_capacity * temp;

            // Determine material type based on thermal state
            let material_type = if i < 8 {
                MaterialType::Silicate // Core/asthenosphere
            } else if i >= 32 {
                MaterialType::Silicate // Space (using silicate as placeholder)
            } else {
                MaterialType::Silicate // Lithosphere
            };

            // Create experimental EnergyMass for this layer
            let energy_mass = ThermalEnergyMassExp::new(
                material_type,
                temp,
                volume_km3,
                layer_thickness,
                thermal_state,  // Include state in EnergyMass
            );

            self.nodes.push(ThermalNode::new(
                i,  // Add index
                depth,
                layer_thickness,
                energy_mass,
            ));
        }
        
        self.initial_nodes = self.nodes.clone();

        println!("   Realistic foundry heat: 1800K foundry with gradient to 1523K melting point");
        println!("   Foundry layers (0-20km): 1800K-1523K (realistic asthenosphere heat)");
        println!("   Surface radiation: Stefan-Boltzmann T^4 radiation to space (top layers only)");
    }

    /// Process all pending changes and apply them to the nodes
    fn apply_pending_changes(&mut self) {
        let updates = self.pending_changes.drain_updates();

        if updates.is_empty() {
            return;
        }

        println!("   📝 Processing {} pending updates", updates.len());

        // Create a clone of nodes to work with
        let mut updated_nodes = self.nodes.clone();

        // Apply each update
        for update in updates {
            if update.index < updated_nodes.len() {
                let node = &mut updated_nodes[update.index];

                // Create new EnergyMass with updated values
                let new_energy_mass = ThermalEnergyMassExp::new_with_density_factor(
                    update.material_type,
                    update.energy_j / (node.thermal_capacity()),  // Calculate temperature
                    update.volume_km3,
                    node.height_km,
                    update.density_factor,
                    node.thermal_state(), // Preserve current state
                );

                // Update the node
                node.energy_mass = new_energy_mass;

                println!("   ✅ Updated node {}: {:.1}K, {:.3} km³",
                         update.index, node.temperature_k(), node.volume_km3());
            }
        }

        // Replace the nodes with updated ones
        self.nodes = updated_nodes;
    }

    /// Collect thermal updates from all nodes
    fn collect_thermal_updates(&mut self, years: f64) {
        // Each node calculates its own updates
        for node in &self.nodes {
            let updates = node.calculate_thermal_updates(&self.nodes, years);
            for update in updates {
                self.pending_changes.add_update(update);
            }
        }

        println!("   🔄 Collected {} thermal updates", self.pending_changes.len());
    }




    
    fn print_initial_profile(&self) {
        println!("📊 Initial Setup:");
        println!("   {} thermal nodes (doubled depth to 200km)", self.nodes.len());
        println!("   🔥 HEAT SOURCE: Core heat engine (8 layers, 0-40km)");
        println!("   ❄️  HEAT SINK: Space cooling sink (8 layers, 160-200km)");
        println!("   ⚡ EXPONENTIAL DIFFUSION: 2 neighbors each side with falloff");

        println!("\n🌡️  Initial Temperature Profile (REALISTIC GEOLOGICAL HEAT):");
        println!("   Foundry layers (0-20km): 1800K → 1523K (realistic asthenosphere)");
        println!("   Lithosphere (20-160km): Variable geothermal gradient");
        println!("   Surface layers (160-300km): 400K → 80K (natural cooling + T^4 radiation)");
    }
    
    fn run_force_directed_diffusion(&mut self, steps: usize, years_per_step: f64) {
        let mut file = File::create("examples/data/thermal_experiment.csv").expect("Could not create file");
        
        // Write header - only every 5th layer, excluding boundaries
        write!(file, "years").unwrap();

        // Export every 5th layer from variable zone (8-31), so layers: 10, 15, 20, 25, 30
        let export_layers = [10, 15, 20, 25, 30];
        for &layer_idx in &export_layers {
            let depth_km = (layer_idx as f64 + 0.5) * 5.0; // 5km per layer
            write!(file, ",temp_{}km", depth_km as i32).unwrap();
        }
        for &layer_idx in &export_layers {
            let depth_km = (layer_idx as f64 + 0.5) * 5.0;
            write!(file, ",state_{}km", depth_km as i32).unwrap();
        }
        writeln!(file).unwrap();
        
        // Export initial state
        self.export_state(&mut file, 0, 0.0);
        
        // Run force-directed thermal diffusion
        for step in 0..steps {
            self.force_directed_step(years_per_step);
            
            // Export every few steps
            if step % 5 == 0 || step == steps - 1 {
                let years = (step + 1) as f64 * years_per_step;
                self.export_state(&mut file, step + 1, years);
            }
            
            if step % 100 == 0 {
                let temp_range = self.calculate_temperature_range();
                let surface_temp = self.nodes[0].temperature_k();
                let core_temp = self.nodes[self.nodes.len()-1].temperature_k();
                let boundary_temps = (self.nodes[9].temperature_k(), self.nodes[10].temperature_k());
                println!("   Step {}: Range={:.1}K, Surface={:.1}K, Core={:.1}K, Boundary=({:.1}K,{:.1}K)",
                         step, temp_range, surface_temp, core_temp, boundary_temps.0, boundary_temps.1);
            }
        }
        
        self.print_final_analysis();
    }
    
    fn force_directed_step(&mut self, years: f64) {
        let mut energy_changes = vec![0.0; self.nodes.len()];

        // Update thermal states based on temperature with time-dependent transitions
        let config = ExperimentState::default();
        self.update_thermal_states(years, &config);

        // BOUNDARY CONDITIONS: Heat source and heat sink
        self.apply_boundary_conditions(&mut energy_changes, years);

        // EXPONENTIAL THERMAL DIFFUSION: Each layer exchanges with 2 neighbors on each side
        for i in 0..self.nodes.len() {
            let current_node = &self.nodes[i];

            // Exchange with 2 layers on each side (i-2, i-1, i+1, i+2) with exponential falloff
            for offset in [-2, -1, 1, 2] {
                let j = i as i32 + offset;
                if j < 0 || j >= self.nodes.len() as i32 { continue; }
                let j = j as usize;

                let other_node = &self.nodes[j];
                let distance_km = (current_node.depth_km - other_node.depth_km).abs();

                // Calculate thermal force using scientific physics with exponential falloff
                let energy_transfer = self.calculate_thermal_force(current_node, other_node, distance_km, years);
                energy_changes[i] += energy_transfer;
            }
        }
        
        // Apply energy changes (with conservation check)
        let total_energy_before: f64 = self.nodes.iter().map(|n| n.energy_j()).sum();

        for (i, &energy_change) in energy_changes.iter().enumerate() {
            let new_energy = self.nodes[i].energy_j() + energy_change;
            self.nodes[i].set_energy(new_energy);
        }

        // AFTER energy flow, reset boundary temperatures to fixed values (infinite reservoirs)
        self.reset_boundary_temperatures();

        let total_energy_after: f64 = self.nodes.iter().map(|n| n.energy_j()).sum();
        let energy_conservation_error = (total_energy_after - total_energy_before).abs() / total_energy_before;
        
        if energy_conservation_error > 0.01 {
            println!("   Warning: Energy conservation error: {:.3}%", energy_conservation_error * 100.0);
        }
    }

    fn apply_boundary_conditions(&self, _energy_changes: &mut Vec<f64>, _years: f64) {
        // No boundary conditions needed here - energy flows naturally
        // Boundary temperatures will be reset after energy changes are applied
    }

    fn reset_boundary_temperatures(&mut self) {
        // FOUNDRY LAYERS (0-3): Reset to realistic hot temperatures (infinite heat reservoir)
        for i in 0..4 {
            let gradient = (1800.0 - 1523.0) / 4.0; // From 1800K to melting point
            let target_temp = 1800.0 - (i as f64 * gradient); // 1800K down to 1523K (realistic foundry)
            self.nodes[i].set_temperature(target_temp);
        }

        // GENTLE SPACE COOLING: Create lithosphere from top down (genesis process)
        // Only apply to the top few layers (thin skin effect)
        for i in 55..60 {
            let temp_k = self.nodes[i].temperature_k();

            // Gentle Stefan-Boltzmann radiation for lithosphere formation
            let stefan_boltzmann_coefficient = 5.67e-8; // W/m²/K⁴
            let surface_area = 25.0e6; // 25 km² surface area per node
            let years_to_seconds = 365.25 * 24.0 * 3600.0; // Convert years to seconds
            let power_watts = stefan_boltzmann_coefficient * surface_area * temp_k.powi(3);
            let energy_loss_per_year = power_watts * years_to_seconds;

            // Apply enhanced radiation loss for thermal equilibrium
            let radiation_loss = energy_loss_per_year * 0.0005; // Enhanced cooling (5x stronger)
            let thermal_capacity = self.nodes[i].thermal_capacity();
            let new_energy = (self.nodes[i].energy_j() - radiation_loss).max(thermal_capacity * 200.0); // Minimum 200K
            self.nodes[i].set_energy(new_energy);
        }
    }


    
    fn calculate_thermal_force(&self, from_node: &ThermalNode, to_node: &ThermalNode, distance_km: f64, years: f64) -> f64 {
        // PROPER DIFFUSION EQUILIBRIUM MODEL:
        // Only the energy DIFFERENCE flows, weighted by distance and conductivity

        let temp_diff = to_node.temperature_k() - from_node.temperature_k();
        if temp_diff.abs() < 0.1 { return 0.0; }

        // Get material-based conductivities
        let from_conductivity = self.get_material_conductivity(from_node);
        let to_conductivity = self.get_material_conductivity(to_node);

        // Distance weights: 1.0, 0.25, 0.125 for neighbors 1, 2, 3 away
        let distance_weight = if distance_km <= 5.1 {
            1.0  // Adjacent neighbor
        } else if distance_km <= 10.1 {
            0.25 // Second neighbor
        } else if distance_km <= 15.1 {
            0.125 // Third neighbor
        } else {
            0.0  // Too far
        };

        // Conductivity factor: both sender and recipient must be conductive
        let sender_factor = from_conductivity / 3.0; // Normalize by average (3.0)
        let recipient_factor = to_conductivity / 3.0; // Normalize by average (3.0)
        let conductivity_factor = sender_factor * recipient_factor;

        // Base diffusion rate: 2% per century (DOUBLED for faster heat transport)
        let base_coefficient = 0.02 / years; // Doubled diffusion rate

        // PRESSURE ACCELERATION: Help foundry heat reach farther
        // (temp_diff / 100)^1.5 with minimum 0.33 for stronger acceleration
        let temp_diff_abs = temp_diff.abs();
        let pressure_factor = ((temp_diff_abs / 100.0).powf(1.5)).max(0.33);

        // DIFFUSION PHYSICS: Only the energy difference flows
        // This represents the portion of energy difference that flows through this specific path
        let energy_difference = temp_diff * from_node.thermal_capacity();
        let flow_coefficient = base_coefficient * distance_weight * conductivity_factor * pressure_factor;
        let energy_transfer = energy_difference * flow_coefficient * years;

        // Limit transfer to prevent instability (max 25% of thermal capacity per step)
        let max_transfer = from_node.thermal_capacity() * 0.25;
        let limited_transfer = energy_transfer.abs().min(max_transfer);

        if temp_diff > 0.0 { limited_transfer } else { -limited_transfer }
    }

    fn update_thermal_states(&mut self, time_years: f64, config: &ExperimentState) {
        for node in &mut self.nodes {
            // Skip fixed boundary layers (foundry core layers)
            if node.index < 4 {
                continue; // Keep foundry layers at state 10 (liquid)
            }

            // Update thermal state based on temperature with time-dependent transitions
            node.update_thermal_state(time_years, config);
        }


    }

    fn get_material_conductivity(&self, node: &ThermalNode) -> f64 {
        match node.thermal_state() {
            1..=4 => 2.5,       // Solid: lower conductivity (lithosphere)
            5 => 2.8,           // Cooling transition: intermediate conductivity
            6 => 2.8,           // Heating transition: intermediate conductivity
            10 => 3.2,          // Liquid: higher conductivity (asthenosphere/core)
            11..=15 => {        // Atmosphere: use material conductivity
                // For atmosphere layers, use the material's thermal conductivity
                node.thermal_conductivity()
            },
            _ => 2.5,           // Default to solid
        }
    }
    
    fn export_state(&self, file: &mut File, step: usize, years: f64) {
        write!(file, "{:.0}", years).unwrap();

        // Export only every 5th layer from variable zone (8-31)
        let export_layers = [10, 15, 20, 25, 30];

        // Export temperatures for selected layers
        for &layer_idx in &export_layers {
            write!(file, ",{:.1}", self.nodes[layer_idx].temperature_k()).unwrap();
        }

        // Export thermal states for selected layers
        for &layer_idx in &export_layers {
            write!(file, ",{}", self.nodes[layer_idx].thermal_state()).unwrap();
        }

        writeln!(file).unwrap();
    }
    
    fn calculate_temperature_range(&self) -> f64 {
        let min_temp = self.nodes.iter().map(|n| n.temperature_k()).fold(f64::INFINITY, f64::min);
        let max_temp = self.nodes.iter().map(|n| n.temperature_k()).fold(f64::NEG_INFINITY, f64::max);
        max_temp - min_temp
    }
    
    fn calculate_max_gradient(&self) -> f64 {
        let mut max_gradient: f64 = 0.0;
        for i in 1..self.nodes.len() {
            let temp_diff = (self.nodes[i].temperature_k() - self.nodes[i-1].temperature_k()).abs();
            let depth_diff = (self.nodes[i].depth_km - self.nodes[i-1].depth_km).abs();
            if depth_diff > 0.0 {
                let gradient = temp_diff / depth_diff;
                max_gradient = max_gradient.max(gradient);
            }
        }
        max_gradient
    }
    
    fn print_final_analysis(&self) {
        println!("\n📊 Final Analysis:");
        println!("   Initial temperature range: {:.1}K",
                 self.initial_nodes.iter().map(|n| n.temperature_k()).fold(f64::NEG_INFINITY, f64::max) -
                 self.initial_nodes.iter().map(|n| n.temperature_k()).fold(f64::INFINITY, f64::min));
        println!("   Final temperature range: {:.1}K",
                 self.nodes.iter().map(|n| n.temperature_k()).fold(f64::NEG_INFINITY, f64::max) -
                 self.nodes.iter().map(|n| n.temperature_k()).fold(f64::INFINITY, f64::min));

        // Melting point constants for reference
        const BASALT_SOLIDUS_K: f64 = 1473.0;     // 1200°C - lithosphere material
        const PERIDOTITE_SOLIDUS_K: f64 = 1573.0; // 1300°C - asthenosphere material
        const TRANSITION_THRESHOLD_K: f64 = 1523.0; // 1250°C - transition point

        println!("\n🌡️  Final Temperature Profile:");
        println!("   🔥 Melting Points: Basalt=1473K, Transition=1523K, Peridotite=1573K");
        println!("   🌡️  Thermal States: 1-4=Solid, 5=Cooling, 6=Heating, 10=Liquid, 11-15=Atmosphere");
        for (i, node) in self.nodes.iter().enumerate() {
            let initial_temp = self.initial_nodes[i].temperature_k();
            let change = node.temperature_k() - initial_temp;

            // Determine layer type based on temperature (ALL layers, not just < 32)
            let layer_type = if i < 4 {
                "FOUNDRY ENGINE".to_string()
            } else {
                // Check if above melting point (1523K) for asthenosphere vs lithosphere
                if node.temperature_k() >= TRANSITION_THRESHOLD_K {
                    let asth_layer = i - 4;
                    format!("AL {} (Asth)", asth_layer)
                } else {
                    let lith_layer = i - 4;
                    format!("EL {} (Lith)", lith_layer)
                }
            };

            // Show melting status with color coding: red for melting, blue for cooling
            let melting_status = if node.temperature_k() >= PERIDOTITE_SOLIDUS_K {
                "\x1b[31m🔥HOT\x1b[0m"  // Red for melting/above melting
            } else if node.temperature_k() >= TRANSITION_THRESHOLD_K {
                "\x1b[31m🌋MELT\x1b[0m"  // Red for melting
            } else if node.temperature_k() >= BASALT_SOLIDUS_K {
                "\x1b[34m🔶WARM\x1b[0m"  // Blue for cooling
            } else if node.temperature_k() >= 273.0 && node.temperature_k() <= 373.0 {
                "\x1b[32m🌍HABITABLE\x1b[0m"  // Green for human habitable range (0-100°C)
            } else {
                "\x1b[34m❄️COLD\x1b[0m"  // Blue for cooling
            };

            // Show thermal state description
            let state_desc = match node.thermal_state() {
                1..=4 => "Solid",
                5 => "Cool",
                6 => "Heat",
                10 => "Liquid",
                11..=15 => "Atmo",
                _ => "Unknown"
            };

            println!("   Layer {}: {:.1}K (Δ{:+.1}K) at {:.1}km depth - {} [State:{}={}] {}",
                     i, node.temperature_k(), change, node.depth_km, layer_type, node.thermal_state(), state_desc, melting_status);
        }
        
        println!("\n🔥 Thermal equilibration complete!");

        println!("\n🌡️  Melting point {}K --- Lith formation temp {}K", PERIDOTITE_SOLIDUS_K, TRANSITION_THRESHOLD_K);

        // Count asthenosphere vs lithosphere layers for final summary
        let mut asth_count = 0;
        let mut lith_count = 0;
        let mut total_layers = 0;

        for (i, node) in self.nodes.iter().enumerate() {
            if i >= 4 && i < 52 { // Exclude foundry (0-3) and surface (52-59) layers
                total_layers += 1;
                if node.temperature_k() >= TRANSITION_THRESHOLD_K {
                    asth_count += 1;
                } else {
                    lith_count += 1;
                }
            }
        }

        let asth_percentage = (asth_count as f64 / total_layers as f64) * 100.0;
        let asth_thickness_km = asth_count * 5; // 5km per layer
        let lith_thickness_km = lith_count * 5; // 5km per layer

        println!("\n🎯 Asthenosphere: {:.1}% ({}km) | Lithosphere: {:.1}% ({}km)",
                 asth_percentage, asth_thickness_km, 100.0 - asth_percentage, lith_thickness_km);
    }
}
