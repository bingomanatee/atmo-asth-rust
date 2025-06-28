#![allow(dead_code)]
#![allow(unused_variables)]

use std::fs::File;
use std::io::Write;
use std::collections::VecDeque;

// Import the real EnergyMass trait and material system
extern crate atmo_asth_rust;
use atmo_asth_rust::energy_mass::{EnergyMass, StandardEnergyMass};
use atmo_asth_rust::material::MaterialType;

/// Material phase state
#[derive(Clone, Debug, PartialEq)]
enum MaterialPhase {
    Solid,
    Liquid,
    Gas,
}

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
    /// Create 4x scaled experiment configuration for scientific accuracy
    fn new_4x_scaled() -> Self {
        Self {
            // Thermal parameters (scaled by 4x)
            melting_point_k: 1523.0 * 4.0,     // 6092K
            cooling_point_k: 1473.0 * 4.0,     // 5892K
            outgassing_threshold_k: 1400.0 * 4.0, // 5600K

            // Transition rates (scaled by 4x for faster dynamics)
            base_melting_rate: (1.0 / 50.0) * 4.0,     // 4x faster melting
            base_cooling_rate: (1.0 / 100.0) * 4.0,    // 4x faster cooling
            melting_temp_factor: (1.0 / 100.0) * 4.0,  // 4x temperature sensitivity
            cooling_temp_factor: (1.0 / 200.0) * 4.0,  // 4x temperature sensitivity

            // Thermal conductivity (scaled by 4x)
            solid_conductivity: 2.5 * 4.0,      // 10.0
            liquid_conductivity: 3.2 * 4.0,     // 12.8
            transition_conductivity: 2.8 * 4.0, // 11.2

            // Diffusion parameters (scaled by 4x)
            conductivity_factor: 3.0 * 4.0,     // 12.0
            distance_length: 4.0 * 4.0,         // 16.0
            pressure_baseline: 1.0 * 4.0,       // 4.0
            max_change_rate: 0.02 * 4.0,        // 8% max change per step

            // Boundary conditions (4x core heat for early Earth conditions)
            foundry_temperature_k: 1800.0 * 4.0,    // 7200K (4x foundry heat)
            surface_temperature_k: 300.0,           // Keep surface temp realistic
            core_heat_input: 2.52e12 * 4.0,         // 10.08e12 J per km² per year (4x core heat = ~270 mW/m²)
            surface_radiation_factor: 1.0 * 4.0,    // 4x radiation cooling

            // Outgassing parameters (scaled by 4x)
            outgassing_rate_multiplier: 0.001 * 4.0,    // 4x outgassing rate
            outgassing_energy_fraction: 0.1 * 4.0,      // 4x energy fraction (capped at reasonable levels)

            // Time parameters (same geological time scale)
            years_per_step: 100.0,
            total_years: 1_000_000_000.0, // 1 billion years
        }
    }
}

/// Thermal node with enhanced state tracking
#[derive(Clone, Debug)]
struct ThermalNode {
    energy_mass: StandardEnergyMass,
    thermal_state: i32,          // -100 to +100 scale
    phase: MaterialPhase,
    depth_km: f64,
    height_km: f64,
    
    // Thermal history for analysis
    initial_temperature: f64,
    max_temperature: f64,
    min_temperature: f64,
    
    // Outgassing tracking
    total_outgassed_mass: f64,
    outgassing_rate: f64,
}

impl ThermalNode {
    fn new(material_type: MaterialType, temperature_k: f64, volume_km3: f64, depth_km: f64, height_km: f64) -> Self {
        let energy_mass = StandardEnergyMass::new_with_material(material_type, temperature_k, volume_km3);
        
        Self {
            energy_mass,
            thermal_state: 100, // Start as solid
            phase: MaterialPhase::Solid,
            depth_km,
            height_km,
            initial_temperature: temperature_k,
            max_temperature: temperature_k,
            min_temperature: temperature_k,
            total_outgassed_mass: 0.0,
            outgassing_rate: 0.0,
        }
    }

    fn temperature_k(&self) -> f64 {
        self.energy_mass.kelvin()
    }

    fn volume_km3(&self) -> f64 {
        self.energy_mass.volume()
    }

    fn mass_kg(&self) -> f64 {
        self.energy_mass.mass_kg()
    }

    fn add_energy(&mut self, energy_j: f64) {
        self.energy_mass.add_energy(energy_j);
        let temp = self.temperature_k();
        if temp > self.max_temperature {
            self.max_temperature = temp;
        }
    }

    fn remove_energy(&mut self, energy_j: f64) {
        if energy_j > 0.0 && energy_j <= self.energy_mass.energy() {
            self.energy_mass.remove_heat(energy_j);
            let temp = self.temperature_k();
            if temp < self.min_temperature {
                self.min_temperature = temp;
            }
        }
    }

    /// Update thermal state based on temperature and melting points
    fn update_thermal_state(&mut self, config: &ExperimentState) {
        let temp = self.temperature_k();
        let melting_point = config.melting_point_k;
        let cooling_point = config.cooling_point_k;
        
        // Calculate thermal state on -100 to +100 scale
        // -100 = 100% gas, 0 = 100% liquid, +100 = 100% solid
        if temp > melting_point {
            // Above melting point: trending toward liquid/gas
            let excess_temp = temp - melting_point;
            let gas_fraction = (excess_temp / 1000.0).min(1.0); // 1000K range for full gas transition
            self.thermal_state = -(gas_fraction * 100.0) as i32;
            self.phase = if gas_fraction > 0.5 { MaterialPhase::Gas } else { MaterialPhase::Liquid };
        } else if temp > cooling_point {
            // Between cooling and melting: liquid state
            self.thermal_state = 0;
            self.phase = MaterialPhase::Liquid;
        } else {
            // Below cooling point: trending toward solid
            let temp_deficit = cooling_point - temp;
            let solid_fraction = (temp_deficit / 500.0).min(1.0); // 500K range for full solid transition
            self.thermal_state = (solid_fraction * 100.0) as i32;
            self.phase = MaterialPhase::Solid;
        }
    }

    /// Calculate outgassing based on temperature
    fn calculate_outgassing(&mut self, config: &ExperimentState, years: f64) -> f64 {
        let temp = self.temperature_k();
        if temp > config.outgassing_threshold_k {
            let temp_excess = temp - config.outgassing_threshold_k;
            let base_rate = config.outgassing_rate_multiplier * temp_excess * self.mass_kg();
            self.outgassing_rate = base_rate * years;
            self.total_outgassed_mass += self.outgassing_rate;
            
            // Remove energy corresponding to outgassed material
            let energy_loss = self.outgassing_rate * config.outgassing_energy_fraction;
            self.remove_energy(energy_loss);
            
            self.outgassing_rate
        } else {
            self.outgassing_rate = 0.0;
            0.0
        }
    }

    /// Get thermal conductivity based on current phase
    fn thermal_conductivity(&self, config: &ExperimentState) -> f64 {
        match self.phase {
            MaterialPhase::Solid => config.solid_conductivity,
            MaterialPhase::Liquid => config.liquid_conductivity,
            MaterialPhase::Gas => config.transition_conductivity,
        }
    }

    /// Format thermal state for display
    fn format_thermal_state(&self) -> String {
        let state_value = self.thermal_state;
        if state_value <= -75 {
            format!("{} Gas", (-state_value))
        } else if state_value <= -25 {
            format!("{} Magma", (-state_value))
        } else if state_value <= 25 {
            "Magma".to_string()
        } else if state_value <= 65 {
            format!("{} Magma", state_value)
        } else {
            format!("{} Solid", state_value)
        }
    }
}

/// 4x Scaled Force-Directed Thermal Diffusion Experiment
struct ScaledThermalExperiment {
    nodes: Vec<ThermalNode>,
    config: ExperimentState,
    step_count: usize,

    // Tracking for analysis
    temperature_history: VecDeque<Vec<f64>>,
    energy_transfers: Vec<f64>,
    total_outgassing: f64,

    // Boundary layer indices
    foundry_start: usize,
    foundry_end: usize,
    surface_start: usize,
    surface_end: usize,
}

impl ScaledThermalExperiment {
    fn new() -> Self {
        let config = ExperimentState::new_4x_scaled();
        let num_nodes = 60;
        let total_depth_km = 300.0; // 300km total depth
        let layer_height_km = total_depth_km / num_nodes as f64; // 5km per layer

        let mut nodes = Vec::new();

        // Create thermal nodes with realistic temperature profile
        for i in 0..num_nodes {
            let depth_km = (i as f64 + 0.5) * layer_height_km;
            let volume_km3 = 1.0 * layer_height_km; // 1 km² surface area

            // Temperature profile: foundry -> geothermal gradient -> surface cooling
            let temperature_k = if i < 4 {
                // Foundry layers (0-20km): 7200K -> 6092K (4x scaled)
                config.foundry_temperature_k - (i as f64 * (config.foundry_temperature_k - config.melting_point_k) / 4.0)
            } else if i >= 52 {
                // Surface layers (260-300km): cooling gradient
                let surface_layer = i - 52;
                config.surface_temperature_k * 4.0 - (surface_layer as f64 * 40.0) // Scaled surface cooling
            } else {
                // Middle layers: geothermal gradient (4x scaled)
                let middle_temp_range = config.melting_point_k - (config.surface_temperature_k * 4.0);
                let middle_layers = 52 - 4;
                let layer_in_middle = i - 4;
                config.melting_point_k - (layer_in_middle as f64 * middle_temp_range / middle_layers as f64)
            };

            let node = ThermalNode::new(
                MaterialType::Silicate,
                temperature_k,
                volume_km3,
                depth_km,
                layer_height_km,
            );
            nodes.push(node);
        }

        Self {
            nodes,
            config,
            step_count: 0,
            temperature_history: VecDeque::new(),
            energy_transfers: Vec::new(),
            total_outgassing: 0.0,
            foundry_start: 0,
            foundry_end: 8,     // 8 foundry layers (0-40km)
            surface_start: 52,  // Surface cooling starts at layer 52 (260km)
            surface_end: 60,    // Surface cooling ends at layer 60 (300km)
        }
    }

    fn print_initial_state(&self) {
        println!("🔬 4x Scaled Force-Directed Thermal Diffusion with Enhanced Parameters");
        println!("====================================================================");
        println!("🎯 Deep geological time: 1 billion years of thermal evolution");
        println!("   4x Enhanced foundry heat: {}K foundry with gradient to {}K melting point",
                 self.config.foundry_temperature_k, self.config.melting_point_k);
        println!("   Foundry layers (0-40km): {}K-{}K (4x enhanced asthenosphere heat)",
                 self.config.foundry_temperature_k, self.config.melting_point_k);
        println!("   Surface radiation: 4x Stefan-Boltzmann T^4 radiation to space (top layers only)");
        println!("📊 Initial Setup:");
        println!("   {} thermal nodes", self.nodes.len());
        println!("   🔥 HEAT SOURCE: 4x Core heat input at bottom layer");
        println!("   ❄️  HEAT SINK: 4x Space radiation cooling at top layer");
        println!("📊 Initial Setup:");
        println!("   {} thermal nodes (doubled depth to {}km)", self.nodes.len(), 300);
        println!("   🔥 HEAT SOURCE: 4x Core heat engine ({} layers, 0-40km)", self.foundry_end);
        println!("   ❄️  HEAT SINK: 4x Space cooling sink ({} layers, 260-300km)", self.surface_end - self.surface_start);
        println!("   ⚡ EXPONENTIAL DIFFUSION: 2 neighbors each side with 4x falloff");

        println!("\n🌡️  Initial Temperature Profile (4X ENHANCED GEOLOGICAL HEAT):");
        println!("   Foundry layers (0-40km): {}K → {}K (4x enhanced asthenosphere)",
                 self.config.foundry_temperature_k, self.config.melting_point_k);
        println!("   Lithosphere (40-260km): Variable geothermal gradient");
        println!("   Surface layers (260-300km): {}K → {}K (natural cooling + 4x T^4 radiation)",
                 self.config.surface_temperature_k * 4.0, self.config.surface_temperature_k);
    }
