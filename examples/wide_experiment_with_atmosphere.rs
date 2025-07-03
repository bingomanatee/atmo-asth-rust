#![allow(dead_code)]
#![allow(unused_variables)]

use std::collections::VecDeque;
use std::fs::File;
use std::io::Write;
/// Grey-gas IR opacity per unit mass (m²/kg) for key species.
/// These are order-of-magnitude starting values—adjust for your mix!
pub const KAPPA_H2O: f64  = 5e-4;
pub const KAPPA_CO2: f64  = 2e-4;
pub const KAPPA_CH4: f64  = 5e-5;
pub const KAPPA_N2O: f64  = 1e-4;
pub const KAPPA_O3:  f64  = 1e-3;
pub const KAPPA_DRY: f64  = 1e-5; // N2/O2 baseline


// Import the real EnergyMass trait and material system
extern crate atmo_asth_rust;

use atmo_asth_rust::energy_mass_composite::{
    EnergyMassComposite, MaterialCompositeType, MaterialPhase,
    get_profile_fast,
};
use atmo_asth_rust::example::{ExperimentState, ExperimentSpecs, ThermalLayerNodeWide, ThermalLayerNodeWideTempParams};
use atmo_asth_rust::material_composite::{get_melting_point_k, get_boiling_point_k};
use atmo_asth_rust::atmospheric_energy_mass_composite::AtmosphericEnergyMass;
// Re-export MaterialPhase for easier access

use atmo_asth_rust::math_utils::lerp;


mod test_utils_1km3;


/// Atmospheric compound types that can be generated from melting events
#[derive(Debug, Clone)]
pub enum AtmosphericCompound {
    CO2 { mass_kg: f64 },
    H2O { mass_kg: f64 },
    SO2 { mass_kg: f64 },
    N2 { mass_kg: f64 },
    H2S { mass_kg: f64 },
    CH4 { mass_kg: f64 },
    H2 { mass_kg: f64 },
    CO { mass_kg: f64 },
    Ar { mass_kg: f64 },
    Other { mass_kg: f64, compound_type: String },
}

impl AtmosphericCompound {
    pub fn mass_kg(&self) -> f64 {
        match self {
            AtmosphericCompound::CO2 { mass_kg } => *mass_kg,
            AtmosphericCompound::H2O { mass_kg } => *mass_kg,
            AtmosphericCompound::SO2 { mass_kg } => *mass_kg,
            AtmosphericCompound::N2 { mass_kg } => *mass_kg,
            AtmosphericCompound::H2S { mass_kg } => *mass_kg,
            AtmosphericCompound::CH4 { mass_kg } => *mass_kg,
            AtmosphericCompound::H2 { mass_kg } => *mass_kg,
            AtmosphericCompound::CO { mass_kg } => *mass_kg,
            AtmosphericCompound::Ar { mass_kg } => *mass_kg,
            AtmosphericCompound::Other { mass_kg, .. } => *mass_kg,
        }
    }

    pub fn compound_name(&self) -> &str {
        match self {
            AtmosphericCompound::CO2 { .. } => "CO2",
            AtmosphericCompound::H2O { .. } => "H2O",
            AtmosphericCompound::SO2 { .. } => "SO2",
            AtmosphericCompound::N2 { .. } => "N2",
            AtmosphericCompound::H2S { .. } => "H2S",
            AtmosphericCompound::CH4 { .. } => "CH4",
            AtmosphericCompound::H2 { .. } => "H2",
            AtmosphericCompound::CO { .. } => "CO",
            AtmosphericCompound::Ar { .. } => "Ar",
            AtmosphericCompound::Other { compound_type, .. } => compound_type,
        }
    }
}

/// Delayed atmospheric injection from melting events
#[derive(Debug, Clone)]
pub struct DelayedAtmosphericInjection {
    pub compounds: Vec<AtmosphericCompound>,
    pub injection_time_years: f64,
    pub source_depth_km: f64,
    pub source_node_id: usize,
}

/// Enhanced atmospheric stack with exponential density layers
#[derive(Debug)]
pub struct AtmosphericStack {
    pub layers: Vec<AtmosphericEnergyMass>,
    pub layer_height_km: f64,
    pub surface_area_km2: f64,
    pub pending_injections: Vec<DelayedAtmosphericInjection>,
    pub total_compounds: std::collections::HashMap<String, f64>, // Track total mass by compound type
}

impl AtmosphericStack {
    pub fn new(num_layers: usize, layer_height_km: f64, surface_area_km2: f64, surface_temp_k: f64) -> Self {
        let mut layers = Vec::new();

        // Create atmospheric layers with exponential density decay
        for layer_index in 0..num_layers {
            let layer_center_height = layer_index as f64 * layer_height_km + layer_height_km / 2.0;

            // Calculate temperature at this height (tropospheric lapse rate)
            let lapse_rate_k_per_km = 6.5; // Standard atmospheric lapse rate
            let layer_temp = (surface_temp_k - layer_center_height * lapse_rate_k_per_km).max(200.0); // Minimum 200K

            let layer = AtmosphericEnergyMass::create_layer_for_cell(
                layer_index,
                layer_height_km,
                surface_area_km2,
                layer_temp,
            );
            layers.push(layer);
        }

        Self {
            layers,
            layer_height_km,
            surface_area_km2,
            pending_injections: Vec::new(),
            total_compounds: std::collections::HashMap::new(),
        }
    }

    /// Add compounds to the lowest atmospheric layer (surface injection)
    pub fn inject_compounds(&mut self, compounds: Vec<AtmosphericCompound>) {
        if let Some(surface_layer) = self.layers.get_mut(0) {
            let total_mass_kg = compounds.iter().map(|c| c.mass_kg()).sum::<f64>();

            // Add energy to the surface layer (simulating mass addition through energy)
            // Use a reasonable temperature for injected gases (e.g., 300K)
            let injection_temp_k = 300.0;
            let specific_heat = surface_layer.specific_heat_j_kg_k();
            let energy_to_add = total_mass_kg * specific_heat * injection_temp_k;
            surface_layer.add_energy(energy_to_add);

            // Track compound types
            for compound in compounds {
                let compound_name = compound.compound_name().to_string();
                let mass = compound.mass_kg();
                *self.total_compounds.entry(compound_name).or_insert(0.0) += mass;
            }
        }
    }

    /// Process pending injections based on current time
    pub fn process_pending_injections(&mut self, current_time_years: f64) {
        let mut remaining_injections = Vec::new();
        let mut injections_to_process = Vec::new();

        for injection in self.pending_injections.drain(..) {
            if current_time_years >= injection.injection_time_years {
                // Time to inject these compounds
                injections_to_process.push(injection);
            } else {
                // Keep for later
                remaining_injections.push(injection);
            }
        }

        // Process injections after collecting them
        for injection in injections_to_process {
            let compound_count = injection.compounds.len();
            self.inject_compounds(injection.compounds);

            if self.total_compounds.len() % 10 == 0 { // Log every 10th injection
                println!("💨 Atmospheric injection at {:.0} years from depth {:.1} km: {} compounds",
                         current_time_years, injection.source_depth_km, compound_count);
            }
        }

        self.pending_injections = remaining_injections;
    }

    /// Schedule delayed atmospheric injection from melting event
    pub fn schedule_injection(&mut self, compounds: Vec<AtmosphericCompound>, delay_years: f64,
                             current_time_years: f64, source_depth_km: f64, source_node_id: usize) {
        let injection = DelayedAtmosphericInjection {
            compounds,
            injection_time_years: current_time_years + delay_years,
            source_depth_km,
            source_node_id,
        };
        self.pending_injections.push(injection);
    }

    /// Get total atmospheric mass
    pub fn total_mass_kg(&self) -> f64 {
        self.layers.iter().map(|layer| layer.mass_kg()).sum()
    }

    /// Get atmospheric composition summary
    pub fn get_composition_summary(&self) -> Vec<(String, f64, f64)> {
        let total_mass = self.total_mass_kg();
        let mut composition = Vec::new();

        for (compound, mass) in &self.total_compounds {
            let percentage = if total_mass > 0.0 { (mass / total_mass) * 100.0 } else { 0.0 };
            composition.push((compound.clone(), *mass, percentage));
        }

        // Sort by mass (descending)
        composition.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        composition
    }
}

/// Geological-scale melting statistics for atmospheric generation
#[derive(Debug, Clone)]
pub struct MeltingStatistics {
    pub depth_km: f64,
    pub melt_events_count: u32,
    pub total_mass_melted_kg: f64,
    pub average_temperature_k: f64,
    pub material_type: MaterialCompositeType,
    pub last_event_time_years: f64,
}

/// Geological atmospheric generation system
pub struct GeologicalAtmosphericGenerator {
    pub atmospheric_stack: AtmosphericStack,
    pub melting_stats_by_depth: std::collections::HashMap<i32, MeltingStatistics>, // Depth bins (10km resolution)
    pub volatile_content_ppm: f64, // Reduced: Parts per million of volatiles in rock
    pub outgassing_efficiency: f64, // Reduced: Fraction of volatiles that reach atmosphere
    pub geological_timescale_years: f64, // Generate atmosphere every N years
    pub last_generation_time: f64,
    pub cumulative_atmosphere_generated_kg: f64,
}

/// Melting event listener that generates atmospheric compounds
pub struct MeltingEventListener {
    pub atmospheric_stack: AtmosphericStack,
    pub volatile_content_ppm: f64, // Parts per million of volatiles in rock
    pub outgassing_efficiency: f64, // Fraction of volatiles that reach atmosphere
    pub geological_generator: GeologicalAtmosphericGenerator,
}

impl GeologicalAtmosphericGenerator {
    pub fn new(num_atmospheric_layers: usize, layer_height_km: f64, surface_area_km2: f64, surface_temp_k: f64) -> Self {
        Self {
            atmospheric_stack: AtmosphericStack::new(num_atmospheric_layers, layer_height_km, surface_area_km2, surface_temp_k),
            melting_stats_by_depth: std::collections::HashMap::new(),
            volatile_content_ppm: 50.0, // Reduced: 50 ppm volatiles (10x less)
            outgassing_efficiency: 0.05, // Reduced: 5% efficiency (6x less)
            geological_timescale_years: 1_000_000.0, // Generate atmosphere every 1 million years
            last_generation_time: 0.0,
            cumulative_atmosphere_generated_kg: 0.0,
        }
    }

    /// Record a melting event for geological-scale atmospheric generation
    pub fn record_melting_event(&mut self, event: &PhaseChangeEvent) {
        if !matches!(event.event_type, PhaseChangeType::Melting) {
            return;
        }

        let depth_bin = (event.depth_km / 10.0) as i32; // 10 km depth bins

        let stats = self.melting_stats_by_depth.entry(depth_bin).or_insert(MeltingStatistics {
            depth_km: depth_bin as f64 * 10.0 + 5.0, // Center of depth bin
            melt_events_count: 0,
            total_mass_melted_kg: 0.0,
            average_temperature_k: 0.0,
            material_type: event.material_type,
            last_event_time_years: event.timestamp_years,
        });

        // Update statistics
        let old_count = stats.melt_events_count as f64;
        stats.melt_events_count += 1;
        stats.total_mass_melted_kg += event.mass_affected_kg;

        // Running average of temperature
        stats.average_temperature_k = (stats.average_temperature_k * old_count + event.temperature_k) / (old_count + 1.0);
        stats.last_event_time_years = event.timestamp_years;
    }

    /// Generate atmosphere based on accumulated melting statistics
    pub fn generate_geological_atmosphere(&mut self, current_time_years: f64) -> f64 {
        if current_time_years - self.last_generation_time < self.geological_timescale_years {
            return 0.0; // Not time yet
        }

        let time_span_years = current_time_years - self.last_generation_time;
        let mut total_atmosphere_generated = 0.0;

        println!("\n🌍 Geological Atmospheric Generation at {:.0} million years:", current_time_years / 1_000_000.0);
        println!("   📊 Analyzing melting statistics over {:.1} million year period", time_span_years / 1_000_000.0);

        // Sort depth bins for organized output
        let mut depth_bins: Vec<_> = self.melting_stats_by_depth.keys().cloned().collect();
        depth_bins.sort();

        for depth_bin in depth_bins {
            if let Some(stats) = self.melting_stats_by_depth.get(&depth_bin) {
                if stats.melt_events_count == 0 {
                    continue;
                }

                // Calculate melting frequency (events per million years)
                let melting_frequency = (stats.melt_events_count as f64) / (time_span_years / 1_000_000.0);

                // Calculate total volatile mass available
                let volatile_mass_kg = stats.total_mass_melted_kg * (self.volatile_content_ppm / 1_000_000.0);
                let outgassed_mass_kg = volatile_mass_kg * self.outgassing_efficiency;

                // Depth-dependent outgassing efficiency (deeper = less efficient)
                let depth_efficiency = (1.0 - (stats.depth_km / 300.0).min(0.8)).max(0.1);
                let effective_outgassed_mass = outgassed_mass_kg * depth_efficiency;

                // Generate compounds for this depth bin
                let compounds = self.generate_compounds_from_statistics(stats, effective_outgassed_mass);
                let compound_mass: f64 = compounds.iter().map(|c| c.mass_kg()).sum();

                // Inject compounds into atmosphere
                self.atmospheric_stack.inject_compounds(compounds);
                total_atmosphere_generated += compound_mass;

                // Log significant contributions
                if melting_frequency > 0.1 && compound_mass > 1e10 {
                    println!("   🌋 Depth {:.0}-{:.0} km: {:.1} events/Myr → {:.2e} kg atmosphere",
                             stats.depth_km - 5.0, stats.depth_km + 5.0, melting_frequency, compound_mass);
                }
            }
        }

        // Clear statistics for next period
        self.melting_stats_by_depth.clear();
        self.last_generation_time = current_time_years;
        self.cumulative_atmosphere_generated_kg += total_atmosphere_generated;

        println!("   💨 Total atmosphere generated this period: {:.2e} kg", total_atmosphere_generated);
        println!("   🌬️  Cumulative atmosphere generated: {:.2e} kg", self.cumulative_atmosphere_generated_kg);

        total_atmosphere_generated
    }

    /// Generate atmospheric compounds from melting statistics
    fn generate_compounds_from_statistics(&self, stats: &MeltingStatistics, effective_outgassed_mass: f64) -> Vec<AtmosphericCompound> {
        let mut compounds = Vec::new();

        // Use the same compound ratios but with reduced mass
        match stats.material_type {
            MaterialCompositeType::Silicate => {
                compounds.push(AtmosphericCompound::CO2 { mass_kg: effective_outgassed_mass * 0.45 });
                compounds.push(AtmosphericCompound::H2O { mass_kg: effective_outgassed_mass * 0.30 });
                compounds.push(AtmosphericCompound::SO2 { mass_kg: effective_outgassed_mass * 0.12 });
                compounds.push(AtmosphericCompound::N2 { mass_kg: effective_outgassed_mass * 0.08 });
                compounds.push(AtmosphericCompound::H2S { mass_kg: effective_outgassed_mass * 0.03 });
                compounds.push(AtmosphericCompound::Other {
                    mass_kg: effective_outgassed_mass * 0.02,
                    compound_type: "Noble_gases".to_string()
                });
            },
            MaterialCompositeType::Basaltic => {
                compounds.push(AtmosphericCompound::H2O { mass_kg: effective_outgassed_mass * 0.40 });
                compounds.push(AtmosphericCompound::CO2 { mass_kg: effective_outgassed_mass * 0.35 });
                compounds.push(AtmosphericCompound::SO2 { mass_kg: effective_outgassed_mass * 0.15 });
                compounds.push(AtmosphericCompound::N2 { mass_kg: effective_outgassed_mass * 0.06 });
                compounds.push(AtmosphericCompound::H2S { mass_kg: effective_outgassed_mass * 0.04 });
            },
            MaterialCompositeType::Granitic => {
                compounds.push(AtmosphericCompound::H2O { mass_kg: effective_outgassed_mass * 0.50 });
                compounds.push(AtmosphericCompound::CO2 { mass_kg: effective_outgassed_mass * 0.25 });
                compounds.push(AtmosphericCompound::SO2 { mass_kg: effective_outgassed_mass * 0.10 });
                compounds.push(AtmosphericCompound::N2 { mass_kg: effective_outgassed_mass * 0.10 });
                compounds.push(AtmosphericCompound::CH4 { mass_kg: effective_outgassed_mass * 0.05 });
            },
            MaterialCompositeType::Metallic => {
                compounds.push(AtmosphericCompound::H2 { mass_kg: effective_outgassed_mass * 0.60 });
                compounds.push(AtmosphericCompound::CO { mass_kg: effective_outgassed_mass * 0.30 });
                compounds.push(AtmosphericCompound::Other {
                    mass_kg: effective_outgassed_mass * 0.10,
                    compound_type: "Metal_vapors".to_string()
                });
            },
            MaterialCompositeType::Icy => {
                compounds.push(AtmosphericCompound::H2O { mass_kg: effective_outgassed_mass * 0.95 });
                compounds.push(AtmosphericCompound::CO2 { mass_kg: effective_outgassed_mass * 0.03 });
                compounds.push(AtmosphericCompound::Other {
                    mass_kg: effective_outgassed_mass * 0.02,
                    compound_type: "Trace_gases".to_string()
                });
            },
            MaterialCompositeType::Air => {
                compounds.push(AtmosphericCompound::N2 { mass_kg: effective_outgassed_mass * 0.78 });
                compounds.push(AtmosphericCompound::Other {
                    mass_kg: effective_outgassed_mass * 0.21,
                    compound_type: "O2".to_string()
                });
                compounds.push(AtmosphericCompound::Ar { mass_kg: effective_outgassed_mass * 0.01 });
            },
        }

        compounds
    }

    /// Get melting statistics summary
    pub fn get_melting_statistics_summary(&self) -> Vec<(f64, u32, f64)> {
        let mut summary = Vec::new();
        for (depth_bin, stats) in &self.melting_stats_by_depth {
            summary.push((stats.depth_km, stats.melt_events_count, stats.total_mass_melted_kg));
        }
        summary.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        summary
    }
}

impl MeltingEventListener {
    pub fn new(num_atmospheric_layers: usize, layer_height_km: f64, surface_area_km2: f64, surface_temp_k: f64) -> Self {
        Self {
            atmospheric_stack: AtmosphericStack::new(num_atmospheric_layers, layer_height_km, surface_area_km2, surface_temp_k),
            volatile_content_ppm: 10.0, // Much reduced: 10 ppm volatiles (50x less than original)
            outgassing_efficiency: 0.01, // Much reduced: 1% efficiency (30x less than original)
            geological_generator: GeologicalAtmosphericGenerator::new(num_atmospheric_layers, layer_height_km, surface_area_km2, surface_temp_k),
        }
    }

    /// Generate atmospheric compounds from a melting event
    pub fn generate_compounds_from_melting(&self, mass_melted_kg: f64, material_type: MaterialCompositeType,
                                          temperature_k: f64, depth_km: f64) -> Vec<AtmosphericCompound> {
        let mut compounds = Vec::new();

        // Calculate total volatile mass available
        let volatile_mass_kg = mass_melted_kg * (self.volatile_content_ppm / 1_000_000.0);
        let outgassed_mass_kg = volatile_mass_kg * self.outgassing_efficiency;

        // Depth-dependent outgassing efficiency (deeper = less efficient)
        let depth_efficiency = (1.0 - (depth_km / 300.0).min(0.8)).max(0.1); // 10% minimum efficiency
        let effective_outgassed_mass = outgassed_mass_kg * depth_efficiency;

        // Temperature-dependent compound ratios
        let temp_factor = (temperature_k - 1400.0) / 600.0; // Normalized temperature above threshold

        // Generate compound distribution based on material type and conditions
        match material_type {
            MaterialCompositeType::Silicate => {
                // Typical mantle outgassing composition
                compounds.push(AtmosphericCompound::CO2 { mass_kg: effective_outgassed_mass * 0.45 });
                compounds.push(AtmosphericCompound::H2O { mass_kg: effective_outgassed_mass * 0.30 });
                compounds.push(AtmosphericCompound::SO2 { mass_kg: effective_outgassed_mass * 0.12 });
                compounds.push(AtmosphericCompound::N2 { mass_kg: effective_outgassed_mass * 0.08 });
                compounds.push(AtmosphericCompound::H2S { mass_kg: effective_outgassed_mass * 0.03 });
                compounds.push(AtmosphericCompound::Other {
                    mass_kg: effective_outgassed_mass * 0.02,
                    compound_type: "Noble_gases".to_string()
                });
            },
            MaterialCompositeType::Basaltic => {
                // Basaltic outgassing (more water, less CO2)
                compounds.push(AtmosphericCompound::H2O { mass_kg: effective_outgassed_mass * 0.40 });
                compounds.push(AtmosphericCompound::CO2 { mass_kg: effective_outgassed_mass * 0.35 });
                compounds.push(AtmosphericCompound::SO2 { mass_kg: effective_outgassed_mass * 0.15 });
                compounds.push(AtmosphericCompound::N2 { mass_kg: effective_outgassed_mass * 0.06 });
                compounds.push(AtmosphericCompound::H2S { mass_kg: effective_outgassed_mass * 0.04 });
            },
            MaterialCompositeType::Granitic => {
                // Granitic outgassing (high water content)
                compounds.push(AtmosphericCompound::H2O { mass_kg: effective_outgassed_mass * 0.50 });
                compounds.push(AtmosphericCompound::CO2 { mass_kg: effective_outgassed_mass * 0.25 });
                compounds.push(AtmosphericCompound::SO2 { mass_kg: effective_outgassed_mass * 0.10 });
                compounds.push(AtmosphericCompound::N2 { mass_kg: effective_outgassed_mass * 0.10 });
                compounds.push(AtmosphericCompound::CH4 { mass_kg: effective_outgassed_mass * 0.05 });
            },
            MaterialCompositeType::Metallic => {
                // Metallic materials - minimal outgassing
                compounds.push(AtmosphericCompound::H2 { mass_kg: effective_outgassed_mass * 0.60 });
                compounds.push(AtmosphericCompound::CO { mass_kg: effective_outgassed_mass * 0.30 });
                compounds.push(AtmosphericCompound::Other {
                    mass_kg: effective_outgassed_mass * 0.10,
                    compound_type: "Metal_vapors".to_string()
                });
            },
            MaterialCompositeType::Icy => {
                // Ice melting - pure water vapor
                compounds.push(AtmosphericCompound::H2O { mass_kg: effective_outgassed_mass * 0.95 });
                compounds.push(AtmosphericCompound::CO2 { mass_kg: effective_outgassed_mass * 0.03 });
                compounds.push(AtmosphericCompound::Other {
                    mass_kg: effective_outgassed_mass * 0.02,
                    compound_type: "Trace_gases".to_string()
                });
            },
            MaterialCompositeType::Air => {
                // Air doesn't typically melt, but if it does, minimal outgassing
                compounds.push(AtmosphericCompound::N2 { mass_kg: effective_outgassed_mass * 0.78 });
                compounds.push(AtmosphericCompound::Other {
                    mass_kg: effective_outgassed_mass * 0.21,
                    compound_type: "O2".to_string()
                });
                compounds.push(AtmosphericCompound::Ar { mass_kg: effective_outgassed_mass * 0.01 });
            },
        }

        compounds
    }

    /// Calculate delay time for atmospheric injection based on depth and material properties (geological timescales)
    pub fn calculate_injection_delay(&self, depth_km: f64, temperature_k: f64) -> f64 {
        // Base delay increases with depth (gases need much longer to migrate upward)
        let depth_delay_years = depth_km * 5000.0; // 5,000 years per km depth (100x longer)

        // Temperature affects gas mobility (higher temp = faster migration)
        let temp_factor = (temperature_k / 1600.0).min(2.0); // Cap at 2x speed
        let temp_adjusted_delay = depth_delay_years / temp_factor;

        // Minimum delay of 10,000 years, maximum of 1,000,000 years (geological timescales)
        temp_adjusted_delay.max(10_000.0).min(1_000_000.0)
    }

    /// Handle a melting event with both immediate (reduced) and geological-scale generation
    pub fn handle_melting_event(&mut self, event: &PhaseChangeEvent) {
        if matches!(event.event_type, PhaseChangeType::Melting) {
            // Record for geological-scale atmospheric generation
            self.geological_generator.record_melting_event(event);

            // Generate immediate (but much reduced) compounds from this melting event
            let compounds = self.generate_compounds_from_melting(
                event.mass_affected_kg,
                event.material_type,
                event.temperature_k,
                event.depth_km,
            );

            // Calculate injection delay (much longer for geological realism)
            let delay_years = self.calculate_injection_delay(event.depth_km, event.temperature_k);

            // Schedule the immediate (reduced) injection
            self.atmospheric_stack.schedule_injection(
                compounds,
                delay_years,
                event.timestamp_years,
                event.depth_km,
                event.node_id,
            );

            // Only log very shallow melting events to reduce noise
            if event.depth_km < 50.0 {
                println!("🌋 Recorded melting at {:.1} km depth for geological atmospheric generation",
                         event.depth_km);
            }
        }
    }

    /// Update atmospheric stack and process pending injections
    pub fn update(&mut self, current_time_years: f64) {
        // Process immediate (reduced) atmospheric injections
        self.atmospheric_stack.process_pending_injections(current_time_years);

        // Generate geological-scale atmosphere based on accumulated melting statistics
        let geological_atmosphere = self.geological_generator.generate_geological_atmosphere(current_time_years);

        if geological_atmosphere > 0.0 {
            println!("   🌍 Geological atmospheric generation: {:.2e} kg added to atmosphere", geological_atmosphere);
        }
    }

    /// Get atmospheric composition for analysis
    pub fn get_atmospheric_analysis(&self) -> (f64, Vec<(String, f64, f64)>, f64, Vec<(f64, u32, f64)>) {
        let total_mass = self.atmospheric_stack.total_mass_kg();
        let composition = self.atmospheric_stack.get_composition_summary();
        let geological_mass = self.geological_generator.cumulative_atmosphere_generated_kg;
        let melting_stats = self.geological_generator.get_melting_statistics_summary();
        (total_mass, composition, geological_mass, melting_stats)
    }

    /// Calculate atmospheric optical depth for solar radiation attenuation
    pub fn calculate_atmospheric_optical_depth(&self) -> f64 {
        let total_mass = self.atmospheric_stack.total_mass_kg();
        let surface_area_m2 = self.atmospheric_stack.surface_area_km2 * 1_000_000.0; // Convert km² to m²

        // Calculate atmospheric column density (kg/m²)
        let column_density = total_mass / surface_area_m2;

        // Optical depth calculation based on atmospheric mass
        // Using typical atmospheric absorption coefficients
        let base_optical_depth = 0.1; // Baseline for thin atmosphere
        let mass_scaling_factor = 1e-4; // Scaling factor for mass-to-optical-depth conversion

        base_optical_depth + (column_density * mass_scaling_factor)
    }

    /// Calculate solar radiation attenuation factor based on atmospheric thickness
    pub fn calculate_solar_attenuation_factor(&self) -> f64 {
        let optical_depth = self.calculate_atmospheric_optical_depth();

        // Beer-Lambert law: I = I₀ * e^(-τ)
        // Where τ is optical depth
        (-optical_depth).exp()
    }

    /// Get representative atmospheric temperature for thermal equilibration
    pub fn get_atmospheric_representative_temperature(&self) -> f64 {
        if self.atmospheric_stack.layers.is_empty() {
            return 280.0; // Default surface temperature
        }

        // Mass-weighted average temperature of all atmospheric layers
        let mut total_mass = 0.0;
        let mut weighted_temp_sum = 0.0;

        for layer in &self.atmospheric_stack.layers {
            let layer_mass = layer.mass_kg();
            let layer_temp = layer.temperature();

            total_mass += layer_mass;
            weighted_temp_sum += layer_mass * layer_temp;
        }

        if total_mass > 0.0 {
            weighted_temp_sum / total_mass
        } else {
            280.0 // Default if no atmospheric mass
        }
    }

    /// Equilibrate atmosphere as single thermal mass with surface
    pub fn equilibrate_atmosphere_with_surface(&mut self, surface_temperature_k: f64, years: f64) {
        if self.atmospheric_stack.layers.is_empty() {
            return;
        }

        // Calculate total atmospheric thermal mass
        let mut total_atmospheric_mass = 0.0;
        let mut total_atmospheric_energy = 0.0;

        for layer in &self.atmospheric_stack.layers {
            let layer_mass = layer.mass_kg();
            let layer_energy = layer.energy();

            total_atmospheric_mass += layer_mass;
            total_atmospheric_energy += layer_energy;
        }

        if total_atmospheric_mass <= 0.0 {
            return;
        }

        // Calculate atmospheric specific heat capacity (average for air)
        let atmospheric_specific_heat = 1005.0; // J/(kg·K) for dry air at constant pressure

        // Calculate current atmospheric temperature
        let current_atmo_temp = total_atmospheric_energy / (total_atmospheric_mass * atmospheric_specific_heat);

        // Heat exchange rate between surface and atmosphere
        let temp_difference = surface_temperature_k - current_atmo_temp;
        let heat_exchange_coefficient = 0.1; // Coupling strength (adjustable)
        let time_factor = years / 1000.0; // Scale by time step

        // Calculate energy transfer
        let energy_transfer = temp_difference * total_atmospheric_mass * atmospheric_specific_heat * heat_exchange_coefficient * time_factor;

        // Apply energy transfer to atmospheric layers proportionally
        if total_atmospheric_energy > 0.0 {
            for layer in &mut self.atmospheric_stack.layers {
                let layer_fraction = layer.energy() / total_atmospheric_energy;
                let layer_energy_change = energy_transfer * layer_fraction;

                // Add energy to bring atmosphere closer to surface temperature
                layer.add_energy(layer_energy_change);
            }
        }

        // Log significant temperature changes
        let new_atmo_temp = self.get_atmospheric_representative_temperature();
        if (new_atmo_temp - current_atmo_temp).abs() > 10.0 {
            println!("🌡️  Atmosphere-surface equilibration: {:.0}K → {:.0}K (surface: {:.0}K)",
                     current_atmo_temp, new_atmo_temp, surface_temperature_k);
        }
    }
}

// Export layer indices - key layer positions for analysis
const EXPORT_LAYER_INDICES: [usize; 9] = [10, 15, 20, 25, 30, 35, 40, 45, 50];

// Area constant for the 100 km² experiment
const AREA_KM2: f64 = 100.0;

// Locked configuration constants for wide experiment - used in both config and tests
const WIDE_CONDUCTIVITY_FACTOR: f64 = 15.0;
const WIDE_PRESSURE_BASELINE: f64 = 5.0;
const WIDE_MAX_CHANGE_RATE: f64 = 0.06;
const WIDE_FOUNDRY_TEMPERATURE_K: f64 = 5500.0; // Realistic core temperature
const WIDE_SURFACE_TEMPERATURE_K: f64 = 280.0;

// Atmospheric generation constants
const OUTGASSING_THRESHOLD_K: f64 = 1400.0; // Temperature threshold for significant outgassing
const VOLATILE_CONCENTRATION_KG_M3: f64 = 50.0; // kg volatiles per m³ of rock
const DEGASSING_EFFICIENCY: f64 = 0.3; // Fraction of volatiles that reach atmosphere
const LITHOSPHERE_ABSORPTION_RATE: f64 = 0.003; // Absorption rate per km of lithosphere

/// Phase change events for observability
#[derive(Debug, Clone)]
pub struct PhaseChangeEvent {
    pub node_id: usize,
    pub depth_km: f64,
    pub temperature_k: f64,
    pub material_type: MaterialCompositeType,
    pub timestamp_years: f64,
    pub event_type: PhaseChangeType,
    pub mass_affected_kg: f64,
}

#[derive(Debug, Clone)]
pub enum PhaseChangeType {
    Melting,
    Solidifying,
    Vaporizing,
    Condensing,
}

/// Simple event collector for phase changes
pub struct PhaseChangeEvents {
    pub events: Vec<PhaseChangeEvent>,
    pub verbose_logging: bool,
}

impl PhaseChangeEvents {
    pub fn new(verbose: bool) -> Self {
        Self {
            events: Vec::new(),
            verbose_logging: verbose,
        }
    }

    pub fn emit(&mut self, event: PhaseChangeEvent) {
        if self.verbose_logging {
            match event.event_type {
                PhaseChangeType::Melting => {
                    println!("🔥 MELTING at {:.0} years: Node {} ({:.1} km deep) - {:.2e} kg of {:?} melted at {:.0}K",
                             event.timestamp_years, event.node_id, event.depth_km, event.mass_affected_kg, event.material_type, event.temperature_k);
                },
                PhaseChangeType::Solidifying => {
                    println!("🧊 SOLIDIFYING at {:.0} years: Node {} ({:.1} km deep) - {:.2e} kg of {:?} solidified at {:.0}K",
                             event.timestamp_years, event.node_id, event.depth_km, event.mass_affected_kg, event.material_type, event.temperature_k);
                },
                PhaseChangeType::Vaporizing => {
                    println!("💨 VAPORIZING at {:.0} years: Node {} ({:.1} km deep) - {:.2e} kg of {:?} vaporized at {:.0}K",
                             event.timestamp_years, event.node_id, event.depth_km, event.mass_affected_kg, event.material_type, event.temperature_k);
                },
                PhaseChangeType::Condensing => {
                    println!("💧 CONDENSING at {:.0} years: Node {} ({:.1} km deep) - {:.2e} kg of {:?} condensed at {:.0}K",
                             event.timestamp_years, event.node_id, event.depth_km, event.mass_affected_kg, event.material_type, event.temperature_k);
                },
            }
        }

        self.events.push(event);
    }

    pub fn get_events_by_type(&self, event_type: PhaseChangeType) -> Vec<&PhaseChangeEvent> {
        self.events.iter()
            .filter(|e| matches!(e.event_type, ref t if std::mem::discriminant(t) == std::mem::discriminant(&event_type)))
            .collect()
    }

    pub fn get_events_in_timeframe(&self, start_years: f64, end_years: f64) -> Vec<&PhaseChangeEvent> {
        self.events.iter()
            .filter(|e| e.timestamp_years >= start_years && e.timestamp_years <= end_years)
            .collect()
    }

    pub fn clear(&mut self) {
        self.events.clear();
    }

    pub fn count_by_type(&self) -> (usize, usize, usize, usize) {
        let mut melting = 0;
        let mut solidifying = 0;
        let mut vaporizing = 0;
        let mut condensing = 0;

        for event in &self.events {
            match event.event_type {
                PhaseChangeType::Melting => melting += 1,
                PhaseChangeType::Solidifying => solidifying += 1,
                PhaseChangeType::Vaporizing => vaporizing += 1,
                PhaseChangeType::Condensing => condensing += 1,
            }
        }

        (melting, solidifying, vaporizing, condensing)
    }
}

// Test expectation constants - calibrated to the wide experiment configuration
const EXPECTED_TEMPERATURE_RANGE_K: f64 = WIDE_FOUNDRY_TEMPERATURE_K - WIDE_SURFACE_TEMPERATURE_K; // 4520.0K
const EXPECTED_NUM_NODES: usize = 40;
const TEMPERATURE_TOLERANCE_PERCENT: f64 = 5.0; // Wider tolerance for wide experiment

/// Atmospheric composition tracking for different gas species
#[derive(Clone, Debug)]
pub struct AtmosphericComposition {
    pub co2_kg: f64,    // Carbon dioxide
    pub h2o_kg: f64,    // Water vapor
    pub so2_kg: f64,    // Sulfur dioxide
    pub n2_kg: f64,     // Nitrogen
    pub other_kg: f64,  // Other trace gases
}

impl AtmosphericComposition {
    pub fn new() -> Self {
        Self {
            co2_kg: 0.0,
            h2o_kg: 0.0,
            so2_kg: 0.0,
            n2_kg: 0.0,
            other_kg: 0.0,
        }
    }

    pub fn total_mass_kg(&self) -> f64 {
        self.co2_kg + self.h2o_kg + self.so2_kg + self.n2_kg + self.other_kg
    }

    /// Add outgassed volatiles based on material type and mass
    pub fn add_outgassing(&mut self, material_type: MaterialCompositeType, mass_kg: f64) {
        match material_type {
            MaterialCompositeType::Basaltic => {
                // Basaltic outgassing composition (volcanic)
                self.co2_kg += mass_kg * 0.4;   // 40% CO2
                self.h2o_kg += mass_kg * 0.35;  // 35% H2O
                self.so2_kg += mass_kg * 0.15;  // 15% SO2
                self.n2_kg += mass_kg * 0.05;   // 5% N2
                self.other_kg += mass_kg * 0.05; // 5% other
            },
            MaterialCompositeType::Granitic => {
                // Granitic outgassing (more water-rich)
                self.co2_kg += mass_kg * 0.25;   // 25% CO2
                self.h2o_kg += mass_kg * 0.55;   // 55% H2O
                self.so2_kg += mass_kg * 0.08;   // 8% SO2
                self.n2_kg += mass_kg * 0.07;    // 7% N2
                self.other_kg += mass_kg * 0.05; // 5% other
            },
            MaterialCompositeType::Silicate => {
                // Mantle silicate outgassing
                self.co2_kg += mass_kg * 0.5;    // 50% CO2
                self.h2o_kg += mass_kg * 0.25;   // 25% H2O
                self.so2_kg += mass_kg * 0.1;    // 10% SO2
                self.n2_kg += mass_kg * 0.1;     // 10% N2
                self.other_kg += mass_kg * 0.05; // 5% other
            },
            _ => {
                // Default composition for other materials
                self.co2_kg += mass_kg * 0.3;
                self.h2o_kg += mass_kg * 0.4;
                self.so2_kg += mass_kg * 0.1;
                self.n2_kg += mass_kg * 0.15;
                self.other_kg += mass_kg * 0.05;
            }
        }
    }
}

/// Create locked configuration specs for the 100km² wide experiment
/// These values are locked to prevent config drift in tests
fn wide_experiment_specs() -> ExperimentSpecs {
    ExperimentSpecs {
        // Use Silicate as the primary material type (mantle material)
        material_type: MaterialCompositeType::Silicate,

        // Diffusion parameters (4x scaled constants) - locked for wide experiment
        conductivity_factor: WIDE_CONDUCTIVITY_FACTOR,
        pressure_baseline: WIDE_PRESSURE_BASELINE,
        max_change_rate: WIDE_MAX_CHANGE_RATE,

        // Boundary conditions (enhanced for early Earth conditions) - locked for wide experiment
        foundry_temperature_k: WIDE_FOUNDRY_TEMPERATURE_K,
        surface_temperature_k: WIDE_SURFACE_TEMPERATURE_K,
    }
}

///  This simulates a single 100 km2 with only vertical flows up and down
///  with energy coming in from the highest/lowest cell in the array
/// to the surface of the earth at cell 0 which radiates heat into space.
/// the settings for the heat flow come in as ExperimentState
/// Enhanced with atmospheric generation from lithosphere evaporation
struct WideExperimentWithAtmosphere {
    nodes: Vec<ThermalLayerNodeWide>,
    atmospheric_layers: Vec<AtmosphericEnergyMass>,
    atmospheric_composition: AtmosphericComposition,
    config: ExperimentState,
    layer_height_km: f64,
    total_years: u64,
    steps: u64,

    // Tracking for analysis
    temperature_history: VecDeque<Vec<f64>>,
    energy_transfers: Vec<f64>,
    total_outgassing: f64,
    atmospheric_mass_history: VecDeque<f64>,

    // Boundary layer indices
    foundry_start: usize,
    foundry_end: usize,
    surface_start: usize,
    surface_end: usize,

    // Core heat variation for upwells
    core_heat_multiplier: f64,
    heat_cycle_years: f64,
    current_year: f64,

    // Phase change observability
    phase_events: PhaseChangeEvents,
    previous_phases: Vec<MaterialPhase>, // Track previous phase state for each node

    // Enhanced atmospheric system with melting event listeners
    melting_event_listener: MeltingEventListener,

    // Step counter for periodic logging
    step_count: u64,
}

impl WideExperimentWithAtmosphere {
    fn new(steps: u64, total_years: u64) -> Self {
        let config = ExperimentState::new_with_specs(wide_experiment_specs());
        let num_nodes = EXPECTED_NUM_NODES;
        let layer_height_km = 5.0; // 5 km per layer for 200 km total depth
        let mut nodes = Vec::new();

        // Create thermal nodes with realistic temperature profile
        for i in 0..num_nodes {
            let depth_km = (i as f64 + 0.5) * layer_height_km;
            let volume_km3 = AREA_KM2 * layer_height_km; // 100 km² surface area

            // Calculate temperature using linear gradient from surface to foundry
            let temp_kelvin = lerp(
                config.surface_temperature_k,
                config.foundry_temperature_k,
                i as f64 / num_nodes as f64
            );

            // Use simplified constructor that sets temperature directly
            let node = ThermalLayerNodeWide::new_with_temperature(ThermalLayerNodeWideTempParams {
                material_type: MaterialCompositeType::Silicate,
                temperature_k: temp_kelvin,
                volume_km3,
                depth_km,
                height_km: layer_height_km,
                area_km2: AREA_KM2,
            });
            nodes.push(node);
        }

        // Create initial atmospheric layers (thin initial atmosphere)
        let mut atmospheric_layers = Vec::new();
        for i in 0..10 { // 10 atmospheric layers, 2 km each = 20 km atmosphere
            let layer = AtmosphericEnergyMass::create_layer_for_cell(
                i,
                2.0, // 2 km per atmospheric layer
                AREA_KM2,
                config.surface_temperature_k - (i as f64 * 6.5), // Temperature lapse rate
            );
            atmospheric_layers.push(layer);
        }

        // Initialize previous phases for each node
        let mut previous_phases = Vec::new();
        for node in &nodes {
            let temp_k = node.temp_kelvin();
            let material_type = node.material_composite_type();
            let melting_point = get_melting_point_k(&material_type);
            let boiling_point = get_boiling_point_k(&material_type);

            let phase = if temp_k < melting_point {
                MaterialPhase::Solid
            } else if temp_k < boiling_point {
                MaterialPhase::Liquid
            } else {
                MaterialPhase::Gas
            };
            previous_phases.push(phase);
        }

        Self {
            nodes,
            atmospheric_layers,
            atmospheric_composition: AtmosphericComposition::new(),
            config,
            layer_height_km,
            total_years,
            steps,
            temperature_history: VecDeque::new(),
            energy_transfers: Vec::new(),
            total_outgassing: 0.0,
            atmospheric_mass_history: VecDeque::new(),
            foundry_start: num_nodes - 5,
            foundry_end: num_nodes,
            surface_start: 0,
            surface_end: 5,
            core_heat_multiplier: 1.0,
            heat_cycle_years: 50000.0, // 50,000 year heat cycles
            current_year: 0.0,
            phase_events: PhaseChangeEvents::new(true), // Enable verbose logging
            previous_phases,
            melting_event_listener: MeltingEventListener::new(
                10, // 10 atmospheric layers
                5.0, // 5 km per layer (50 km total atmosphere)
                100.0, // 100 km² surface area
                280.0, // 280K surface temperature
            ),
            step_count: 0,
        }
    }

    /// Calculate variable core heat based on cyclic upwells
    fn calculate_core_heat_multiplier(&self) -> f64 {
        let cycle_progress = (self.current_year % self.heat_cycle_years) / self.heat_cycle_years;
        // Sinusoidal variation: 0.5 to 2.0 multiplier
        1.25 + 0.75 * (cycle_progress * 2.0 * std::f64::consts::PI).sin()
    }

    /// Process atmospheric generation from lithosphere evaporation
    fn process_atmospheric_generation(&mut self, years_per_step: f64) {
        let mut total_outgassed_mass = 0.0;

        for node in &mut self.nodes {
            let temp_k = node.temp_kelvin();

            // Check if temperature exceeds outgassing threshold
            if temp_k > OUTGASSING_THRESHOLD_K {
                // Calculate outgassing rate based on temperature excess and material properties
                let temp_excess = temp_k - OUTGASSING_THRESHOLD_K;
                let material_type = node.material_composite_type();

                // Get material properties for outgassing calculation
                let profile = get_profile_fast(&material_type, &MaterialPhase::Solid);
                let density = profile.density_kg_m3;
                let volume_m3 = node.volume() * 1e9; // Convert km³ to m³

                // Calculate mass flux based on temperature and volatile content
                let temp_factor = (temp_excess / 1000.0).min(1.0); // Normalize to 0-1 for 0-1000K excess
                let mass_flux_kg_per_year = volume_m3 * VOLATILE_CONCENTRATION_KG_M3 * temp_factor * 0.001; // 0.1% per year at max temp

                let outgassed_mass = mass_flux_kg_per_year * years_per_step;

                // Apply lithosphere absorption (gases trapped in overlying rock)
                let depth_km = node.depth_km;
                let lithosphere_thickness = depth_km.min(50.0); // Assume max 50km lithosphere
                let absorption_fraction = (LITHOSPHERE_ABSORPTION_RATE * lithosphere_thickness).min(0.9);
                let escaped_mass = outgassed_mass * (1.0 - absorption_fraction) * DEGASSING_EFFICIENCY;

                if escaped_mass > 0.0 {
                    // Add to atmospheric composition
                    self.atmospheric_composition.add_outgassing(material_type, escaped_mass);
                    total_outgassed_mass += escaped_mass;

                    // Remove energy from the node (endothermic outgassing)
                    let energy_loss = escaped_mass * profile.latent_heat_vapor * 0.1; // 10% of vaporization energy
                    node.remove_energy(energy_loss);

                    // Update node tracking
                    node.total_outgassed_mass += escaped_mass;
                    node.outgassing_rate = escaped_mass / years_per_step;
                }
            }
        }

        self.total_outgassing += total_outgassed_mass;

        // Update atmospheric layers with new mass
        if total_outgassed_mass > 0.0 {
            self.update_atmospheric_layers(total_outgassed_mass);
        }
    }

    /// Update atmospheric layers with newly generated gases
    fn update_atmospheric_layers(&mut self, new_mass_kg: f64) {
        if self.atmospheric_layers.is_empty() {
            return;
        }

        // Distribute new atmospheric mass to lower layers (surface outgassing)
        let surface_layer_fraction = 0.7; // 70% goes to surface layer
        let remaining_fraction = 0.3;

        // Add mass to surface layer
        if let Some(surface_layer) = self.atmospheric_layers.get_mut(0) {
            let surface_mass = new_mass_kg * surface_layer_fraction;
            let surface_energy = surface_mass * 1005.0 * surface_layer.kelvin(); // Air specific heat
            surface_layer.add_energy(surface_energy);
        }

        // Distribute remaining mass to other layers
        let remaining_mass = new_mass_kg * remaining_fraction;
        let mass_per_layer = remaining_mass / (self.atmospheric_layers.len() - 1) as f64;

        for i in 1..self.atmospheric_layers.len() {
            if let Some(layer) = self.atmospheric_layers.get_mut(i) {
                let layer_energy = mass_per_layer * 1005.0 * layer.kelvin();
                layer.add_energy(layer_energy);
            }
        }
    }

    /// Apply variable core heating to simulate upwells
    fn apply_variable_core_heating(&mut self, years_per_step: f64) {
        let heat_multiplier = self.calculate_core_heat_multiplier();

        // Apply enhanced heating to foundry layers
        for i in self.foundry_start..self.foundry_end {
            if let Some(node) = self.nodes.get_mut(i) {
                // Base core heat flux (simplified model)
                let base_heat_flux_w_m2 = 0.1; // 0.1 W/m² base geothermal flux
                let enhanced_flux = base_heat_flux_w_m2 * heat_multiplier;

                let area_m2 = AREA_KM2 * 1e6; // Convert km² to m²
                let heat_input_j = enhanced_flux * area_m2 * years_per_step * 365.25 * 24.0 * 3600.0;

                node.add_energy(heat_input_j);
            }
        }

        self.core_heat_multiplier = heat_multiplier;
    }

    fn years_per_step(&self) -> f64 {
        self.total_years as f64 / self.steps as f64
    }

    /// Detect and emit phase change events by comparing current and previous phases
    fn detect_phase_changes(&mut self) {
        for (node_id, node) in self.nodes.iter().enumerate() {
            let temp_k = node.temp_kelvin();
            let material_type = node.material_composite_type();
            let melting_point = get_melting_point_k(&material_type);
            let boiling_point = get_boiling_point_k(&material_type);

            let current_phase = if temp_k < melting_point {
                MaterialPhase::Solid
            } else if temp_k < boiling_point {
                MaterialPhase::Liquid
            } else {
                MaterialPhase::Gas
            };

            let previous_phase = &self.previous_phases[node_id];

            // Check for phase transitions
            if !matches!(current_phase, ref p if std::mem::discriminant(p) == std::mem::discriminant(previous_phase)) {
                let mass_affected = node.mass_kg(); // Simplified - assume all mass changes phase

                let event_type = match (previous_phase, &current_phase) {
                    (MaterialPhase::Solid, MaterialPhase::Liquid) => Some(PhaseChangeType::Melting),
                    (MaterialPhase::Liquid, MaterialPhase::Solid) => Some(PhaseChangeType::Solidifying),
                    (MaterialPhase::Liquid, MaterialPhase::Gas) => Some(PhaseChangeType::Vaporizing),
                    (MaterialPhase::Gas, MaterialPhase::Liquid) => Some(PhaseChangeType::Condensing),
                    (MaterialPhase::Solid, MaterialPhase::Gas) => Some(PhaseChangeType::Vaporizing), // Sublimation
                    (MaterialPhase::Gas, MaterialPhase::Solid) => Some(PhaseChangeType::Condensing), // Deposition
                    // Handle cases where there's no actual phase change (shouldn't happen due to the if condition)
                    _ => None,
                };

                if let Some(event_type) = event_type {
                    let event = PhaseChangeEvent {
                        node_id,
                        depth_km: node.depth_km,
                        temperature_k: temp_k,
                        material_type,
                        timestamp_years: self.current_year,
                        event_type,
                        mass_affected_kg: mass_affected,
                    };

                    self.phase_events.emit(event.clone());

                    // Handle melting events for atmospheric generation
                    self.melting_event_listener.handle_melting_event(&event);
                }

                // Update previous phase
                self.previous_phases[node_id] = current_phase;
            }
        }
    }

    /// Force-directed thermal diffusion step with atmospheric generation
    fn force_directed_step(&mut self) {
        let years_per_step = self.years_per_step();
        self.current_year += years_per_step;

        // Apply variable core heating first
        self.apply_variable_core_heating(years_per_step);

        // Apply thermal diffusion between nodes
        let mut energy_changes = vec![0.0; self.nodes.len()];

        for i in 0..self.nodes.len() {
            let (left_part, right_part) = self.nodes.split_at_mut(i);
            let (current_part, right_part) = right_part.split_at_mut(1);

            let left_neighbor = if i > 0 { left_part.last_mut() } else { None };
            let right_neighbor = right_part.first_mut();
            let current_node = &mut current_part[0];

            let energy_change = current_node.apply_fourier_thermal_transfer(
                left_neighbor,
                right_neighbor,
                years_per_step
            );

            energy_changes[i] = energy_change;
        }

        // Process atmospheric generation from hot nodes
        self.process_atmospheric_generation(years_per_step);

        // Apply solar radiation with atmospheric attenuation
        self.apply_solar_radiation_with_atmospheric_attenuation(years_per_step);

        // Apply surface cooling (Stefan-Boltzmann radiation)
        self.apply_surface_cooling(years_per_step);

        // Equilibrate atmosphere with surface temperature
        self.equilibrate_atmosphere_with_surface_temperature(years_per_step);

        // Detect and emit phase change events
        self.detect_phase_changes();

        // Update atmospheric stack and process pending injections
        self.melting_event_listener.update(self.current_year);

        // Increment step counter
        self.step_count += 1;

        // Record state for analysis
        self.record_state();
    }

    /// Apply solar radiation with atmospheric attenuation
    fn apply_solar_radiation_with_atmospheric_attenuation(&mut self, years_per_step: f64) {
        const SOLAR_CONSTANT: f64 = 1361.0; // W/m² - Solar irradiance at Earth's distance
        const ALBEDO: f64 = 0.3; // Surface albedo (reflectivity)

        // Calculate base solar energy input
        let area_m2 = AREA_KM2 * 1e6;
        let absorbed_solar_fraction = 1.0 - ALBEDO;
        let base_solar_energy_j = SOLAR_CONSTANT * area_m2 * absorbed_solar_fraction * years_per_step * 365.25 * 24.0 * 3600.0;

        // Calculate atmospheric attenuation
        let attenuation_factor = self.melting_event_listener.calculate_solar_attenuation_factor();
        let attenuated_solar_energy = base_solar_energy_j * attenuation_factor;

        // Apply attenuated solar radiation to surface nodes
        let solar_energy_per_node = attenuated_solar_energy / (self.surface_end - self.surface_start) as f64;
        for i in self.surface_start..self.surface_end {
            if let Some(node) = self.nodes.get_mut(i) {
                node.add_energy(solar_energy_per_node);
            }
        }

        // Log atmospheric attenuation effects periodically
        if self.step_count % 20 == 0 { // Log every 20 steps
            let optical_depth = self.melting_event_listener.calculate_atmospheric_optical_depth();
            let atmo_mass = self.melting_event_listener.atmospheric_stack.total_mass_kg();
            let attenuation_percent = (1.0 - attenuation_factor) * 100.0;

            if attenuation_percent > 1.0 { // Only log significant attenuation
                println!("☀️  Solar attenuation: {:.1}% (optical depth: {:.3}, atmo mass: {:.2e} kg)",
                         attenuation_percent, optical_depth, atmo_mass);
            }
        }
    }

    /// Apply Stefan-Boltzmann cooling to surface layers
    fn apply_surface_cooling(&mut self, years_per_step: f64) {
        const STEFAN_BOLTZMANN: f64 = 5.670374419e-8; // W⋅m⁻²⋅K⁻⁴
        const EMISSIVITY: f64 = 0.95; // Surface emissivity

        for i in self.surface_start..self.surface_end {
            if let Some(node) = self.nodes.get_mut(i) {
                let temp_k = node.temp_kelvin();
                let radiated_power_w_m2 = EMISSIVITY * STEFAN_BOLTZMANN * temp_k.powi(4);

                let area_m2 = AREA_KM2 * 1e6;
                let energy_loss_j = radiated_power_w_m2 * area_m2 * years_per_step * 365.25 * 24.0 * 3600.0;

                // Apply atmospheric insulation effect (greenhouse effect)
                let atmospheric_mass = self.melting_event_listener.atmospheric_stack.total_mass_kg();
                let insulation_factor = 1.0 / (1.0 + atmospheric_mass / 1e12); // Reduce cooling with more atmosphere

                node.remove_energy(energy_loss_j * insulation_factor);
            }
        }
    }

    /// Equilibrate atmosphere with surface temperature
    fn equilibrate_atmosphere_with_surface_temperature(&mut self, years_per_step: f64) {
        // Calculate average surface temperature
        let mut surface_temp_sum = 0.0;
        let mut surface_count = 0;

        for i in self.surface_start..self.surface_end {
            if let Some(node) = self.nodes.get(i) {
                surface_temp_sum += node.temp_kelvin();
                surface_count += 1;
            }
        }

        if surface_count > 0 {
            let average_surface_temp = surface_temp_sum / surface_count as f64;

            // Equilibrate atmosphere with surface
            self.melting_event_listener.equilibrate_atmosphere_with_surface(
                average_surface_temp,
                years_per_step
            );
        }
    }

    fn record_state(&mut self) {
        let temps: Vec<f64> = self.nodes.iter().map(|node| node.temp_kelvin()).collect();
        self.temperature_history.push_back(temps);

        let atmospheric_mass = self.atmospheric_composition.total_mass_kg();
        self.atmospheric_mass_history.push_back(atmospheric_mass);

        // Keep only recent history to manage memory
        if self.temperature_history.len() > 1000 {
            self.temperature_history.pop_front();
        }
        if self.atmospheric_mass_history.len() > 1000 {
            self.atmospheric_mass_history.pop_front();
        }
    }

    fn init_csv(&self) -> File {
        let filename = format!("wide_experiment_atmosphere_{}_steps.csv", self.steps);
        let mut file = File::create(&filename).expect("Could not create CSV file");

        // Write header with atmospheric data
        write!(file, "years,core_heat_multiplier,total_atmospheric_mass_kg,co2_kg,h2o_kg,so2_kg,n2_kg,other_kg").unwrap();
        for i in EXPORT_LAYER_INDICES.iter() {
            write!(file, ",temp_layer_{}", i).unwrap();
        }
        for i in 0..self.atmospheric_layers.len() {
            write!(file, ",atm_temp_layer_{}", i).unwrap();
        }
        writeln!(file).unwrap();

        file
    }

    fn export_state(&self, file: &mut File, years: f64) {
        // Export core data and atmospheric composition
        write!(file, "{:.0},{:.3},{:.2e},{:.2e},{:.2e},{:.2e},{:.2e},{:.2e}",
               years,
               self.core_heat_multiplier,
               self.atmospheric_composition.total_mass_kg(),
               self.atmospheric_composition.co2_kg,
               self.atmospheric_composition.h2o_kg,
               self.atmospheric_composition.so2_kg,
               self.atmospheric_composition.n2_kg,
               self.atmospheric_composition.other_kg).unwrap();

        // Export selected thermal layer temperatures
        for &i in EXPORT_LAYER_INDICES.iter() {
            if i < self.nodes.len() {
                write!(file, ",{:.1}", self.nodes[i].temp_kelvin()).unwrap();
            } else {
                write!(file, ",0.0").unwrap();
            }
        }

        // Export atmospheric layer temperatures
        for layer in &self.atmospheric_layers {
            write!(file, ",{:.1}", layer.kelvin()).unwrap();
        }

        writeln!(file).unwrap();
    }

    fn print_initial_state(&self) {
        println!("🌍 Wide Experiment with Atmospheric Generation - Initial State");
        println!("==============================================================");
        println!("📊 Configuration:");
        println!("   • Area: {:.0} km²", AREA_KM2);
        println!("   • Nodes: {}", self.nodes.len());
        println!("   • Layer height: {:.1} km", self.layer_height_km);
        println!("   • Total depth: {:.1} km", self.nodes.len() as f64 * self.layer_height_km);
        println!("   • Atmospheric layers: {}", self.atmospheric_layers.len());
        println!("   • Heat cycle period: {:.0} years", self.heat_cycle_years);
        println!();
        println!("🌡️  Temperature Profile:");
        println!("   • Surface: {:.0}K", self.nodes[0].temp_kelvin());
        println!("   • Mid-depth: {:.0}K", self.nodes[self.nodes.len()/2].temp_kelvin());
        println!("   • Foundry: {:.0}K", self.nodes[self.nodes.len()-1].temp_kelvin());
        println!();
        println!("🌬️  Atmospheric Parameters:");
        println!("   • Outgassing threshold: {:.0}K", OUTGASSING_THRESHOLD_K);
        println!("   • Volatile concentration: {:.1} kg/m³", VOLATILE_CONCENTRATION_KG_M3);
        println!("   • Degassing efficiency: {:.1}%", DEGASSING_EFFICIENCY * 100.0);
        println!("   • Lithosphere absorption: {:.3}/km", LITHOSPHERE_ABSORPTION_RATE);
    }

    fn run(&mut self, steps: u64, total_years: u64) {
        self.total_years = total_years;
        self.steps = steps;
        let mut file = self.init_csv();

        // Export initial state
        self.export_state(&mut file, 0.0);

        let mut last_progress: u32 = 0;
        // Run force-directed thermal diffusion with atmospheric generation
        for step in 0..steps {
            self.force_directed_step();

            let years = (step + 1) as f64 * self.years_per_step();
            self.export_state(&mut file, years);

            // Progress reporting every 10% of total steps
            let progress = (step as f64 / steps as f64) * 100.0;
            let pp = (progress / 10.0).floor() as u32;
            if last_progress != pp {
                println!(
                    "    {}    Progress: {:.1}% - {:.0} years - Atmosphere: {:.2e} kg - Heat: {:.2}x",
                    step, progress, years,
                    self.atmospheric_composition.total_mass_kg(),
                    self.core_heat_multiplier
                );
                last_progress = pp;
            }
        }

        self.print_final_analysis();
    }

    fn print_final_analysis(&self) {
        println!("\n🔬 Final Analysis - Wide Experiment with Atmospheric Generation");
        println!("================================================================");

        // Temperature analysis
        if let Some(final_temps) = self.temperature_history.back() {
            let surface_temp = final_temps[0];
            let mid_temp = final_temps[final_temps.len() / 2];
            let foundry_temp = final_temps[final_temps.len() - 1];

            println!("🌡️  Final Temperature Profile:");
            println!("   • Surface: {:.0}K", surface_temp);
            println!("   • Mid-depth: {:.0}K", mid_temp);
            println!("   • Foundry: {:.0}K", foundry_temp);
            println!("   • Temperature gradient: {:.1}K/km",
                     (foundry_temp - surface_temp) / (self.nodes.len() as f64 * self.layer_height_km));
        }

        // Atmospheric analysis
        println!("\n🌬️  Atmospheric Generation Results:");
        println!("   • Total atmospheric mass: {:.2e} kg", self.atmospheric_composition.total_mass_kg());
        println!("   • CO₂: {:.2e} kg ({:.1}%)",
                 self.atmospheric_composition.co2_kg,
                 100.0 * self.atmospheric_composition.co2_kg / self.atmospheric_composition.total_mass_kg());
        println!("   • H₂O: {:.2e} kg ({:.1}%)",
                 self.atmospheric_composition.h2o_kg,
                 100.0 * self.atmospheric_composition.h2o_kg / self.atmospheric_composition.total_mass_kg());
        println!("   • SO₂: {:.2e} kg ({:.1}%)",
                 self.atmospheric_composition.so2_kg,
                 100.0 * self.atmospheric_composition.so2_kg / self.atmospheric_composition.total_mass_kg());
        println!("   • N₂: {:.2e} kg ({:.1}%)",
                 self.atmospheric_composition.n2_kg,
                 100.0 * self.atmospheric_composition.n2_kg / self.atmospheric_composition.total_mass_kg());
        println!("   • Other: {:.2e} kg ({:.1}%)",
                 self.atmospheric_composition.other_kg,
                 100.0 * self.atmospheric_composition.other_kg / self.atmospheric_composition.total_mass_kg());

        // Outgassing analysis
        println!("\n🌋 Outgassing Analysis:");
        println!("   • Total outgassed mass: {:.2e} kg", self.total_outgassing);
        let hot_nodes = self.nodes.iter().filter(|n| n.temp_kelvin() > OUTGASSING_THRESHOLD_K).count();
        println!("   • Active outgassing nodes: {} / {}", hot_nodes, self.nodes.len());

        if let Some(max_outgassing_node) = self.nodes.iter().max_by(|a, b|
            a.total_outgassed_mass.partial_cmp(&b.total_outgassed_mass).unwrap()) {
            println!("   • Max outgassing from single node: {:.2e} kg at depth {:.1} km",
                     max_outgassing_node.total_outgassed_mass, max_outgassing_node.depth_km);
        }

        // Heat cycle analysis
        println!("\n🔥 Heat Cycle Analysis:");
        println!("   • Final heat multiplier: {:.2}x", self.core_heat_multiplier);
        println!("   • Heat cycle period: {:.0} years", self.heat_cycle_years);
        println!("   • Current cycle position: {:.1}%",
                 100.0 * (self.current_year % self.heat_cycle_years) / self.heat_cycle_years);

        // Phase change analysis
        let (melting_count, solidifying_count, vaporizing_count, condensing_count) = self.phase_events.count_by_type();
        println!("\n🔄 Phase Change Analysis:");
        println!("   • Total phase change events: {}", self.phase_events.events.len());
        println!("   • Melting events: {}", melting_count);
        println!("   • Solidifying events: {}", solidifying_count);
        println!("   • Vaporizing events: {}", vaporizing_count);
        println!("   • Condensing events: {}", condensing_count);

        if !self.phase_events.events.is_empty() {
            // Find most active depth range for phase changes
            let mut depth_counts: std::collections::HashMap<i32, usize> = std::collections::HashMap::new();
            for event in &self.phase_events.events {
                let depth_bin = (event.depth_km / 10.0) as i32; // 10 km bins
                *depth_counts.entry(depth_bin).or_insert(0) += 1;
            }

            if let Some((most_active_depth_bin, count)) = depth_counts.iter().max_by_key(|&(_, count)| count) {
                println!("   • Most active depth range: {}-{} km ({} events)",
                         most_active_depth_bin * 10, (most_active_depth_bin + 1) * 10, count);
            }
        }

        // Enhanced atmospheric analysis from melting events
        let (enhanced_atmo_mass, enhanced_composition, geological_mass, melting_stats) = self.melting_event_listener.get_atmospheric_analysis();
        println!("\n🌬️  Enhanced Atmospheric Analysis (from melting events):");
        println!("   • Total immediate atmospheric mass: {:.2e} kg", enhanced_atmo_mass);
        println!("   • Total geological atmospheric mass: {:.2e} kg", geological_mass);
        println!("   • Combined atmospheric mass: {:.2e} kg", enhanced_atmo_mass + geological_mass);
        println!("   • Pending atmospheric injections: {}", self.melting_event_listener.atmospheric_stack.pending_injections.len());

        if !enhanced_composition.is_empty() {
            println!("   • Enhanced atmospheric composition:");
            for (compound, mass_kg, percentage) in enhanced_composition.iter().take(6) {
                println!("     - {}: {:.2e} kg ({:.1}%)", compound, mass_kg, percentage);
            }
        }

        // Geological melting statistics
        if !melting_stats.is_empty() {
            println!("\n🌋 Geological Melting Statistics (current period):");
            for (depth_km, event_count, total_mass_kg) in melting_stats.iter().take(8) {
                if *event_count > 0 {
                    println!("     - {:.0} km depth: {} events, {:.2e} kg melted", depth_km, event_count, total_mass_kg);
                }
            }
        }

        // Compare with original atmospheric generation
        let original_atmo_mass: f64 = self.atmospheric_layers.iter().map(|layer| layer.mass_kg()).sum();
        println!("   • Original atmospheric mass: {:.2e} kg", original_atmo_mass);
        println!("   • Total enhancement factor: {:.2}x",
                 if original_atmo_mass > 0.0 { (enhanced_atmo_mass + geological_mass) / original_atmo_mass } else { 0.0 });

        println!("\n✅ Simulation completed successfully!");
        println!("📊 Data exported to: wide_experiment_atmosphere_{}_steps.csv", self.steps);
    }
}

fn main() {
    println!("🔬 Wide Experiment with Atmospheric Generation from Lithosphere Evaporation");
    println!("===========================================================================");
    println!("🎯 Simulating atmospheric generation through variable core heating cycles");

    let total_years = 1_000_000; // 1 million years
    let years_per_step = 5_000;  // 5,000 years per step
    let steps = total_years / years_per_step;

    // Create experiment with atmospheric generation
    let mut experiment = WideExperimentWithAtmosphere::new(steps, total_years);
    experiment.print_initial_state();

    // Run thermal equilibration with atmospheric generation
    println!("\n🌡️  Running thermal simulation with atmospheric generation...");
    experiment.run(steps, total_years);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atmospheric_composition_creation() {
        let mut composition = AtmosphericComposition::new();
        assert_eq!(composition.total_mass_kg(), 0.0);

        // Test basaltic outgassing
        composition.add_outgassing(MaterialCompositeType::Basaltic, 1000.0);
        assert!(composition.co2_kg > 0.0);
        assert!(composition.h2o_kg > 0.0);
        assert!(composition.so2_kg > 0.0);
        assert_eq!(composition.total_mass_kg(), 1000.0);

        println!("Basaltic outgassing composition:");
        println!("  CO₂: {:.1} kg ({:.1}%)", composition.co2_kg, 100.0 * composition.co2_kg / 1000.0);
        println!("  H₂O: {:.1} kg ({:.1}%)", composition.h2o_kg, 100.0 * composition.h2o_kg / 1000.0);
        println!("  SO₂: {:.1} kg ({:.1}%)", composition.so2_kg, 100.0 * composition.so2_kg / 1000.0);
    }

    #[test]
    fn test_core_heat_variation() {
        let mut experiment = WideExperimentWithAtmosphere::new(100, 100000);

        // Test heat multiplier at different cycle positions
        experiment.current_year = 0.0;
        let heat_0 = experiment.calculate_core_heat_multiplier();

        experiment.current_year = experiment.heat_cycle_years / 4.0; // Quarter cycle
        let heat_quarter = experiment.calculate_core_heat_multiplier();

        experiment.current_year = experiment.heat_cycle_years / 2.0; // Half cycle
        let heat_half = experiment.calculate_core_heat_multiplier();

        println!("Heat multiplier variation:");
        println!("  Start of cycle: {:.3}x", heat_0);
        println!("  Quarter cycle: {:.3}x", heat_quarter);
        println!("  Half cycle: {:.3}x", heat_half);

        // Verify variation range
        assert!(heat_0 >= 0.5 && heat_0 <= 2.0);
        assert!(heat_quarter >= 0.5 && heat_quarter <= 2.0);
        assert!(heat_half >= 0.5 && heat_half <= 2.0);

        // Verify different values at different cycle positions
        assert!((heat_0 - heat_quarter).abs() > 0.1);
    }

    #[test]
    fn test_atmospheric_generation_process() {
        let mut experiment = WideExperimentWithAtmosphere::new(10, 10000);

        // Set some nodes to high temperature to trigger outgassing
        for i in 30..35 {
            if let Some(node) = experiment.nodes.get_mut(i) {
                node.set_kelvin(2000.0); // Well above outgassing threshold
            }
        }

        let initial_atmospheric_mass = experiment.atmospheric_composition.total_mass_kg();

        // Process atmospheric generation for one step
        experiment.process_atmospheric_generation(1000.0); // 1000 years

        let final_atmospheric_mass = experiment.atmospheric_composition.total_mass_kg();

        println!("Atmospheric generation test:");
        println!("  Initial atmospheric mass: {:.2e} kg", initial_atmospheric_mass);
        println!("  Final atmospheric mass: {:.2e} kg", final_atmospheric_mass);
        println!("  Generated mass: {:.2e} kg", final_atmospheric_mass - initial_atmospheric_mass);

        // Should have generated some atmosphere from hot nodes
        assert!(final_atmospheric_mass > initial_atmospheric_mass);
        assert!(experiment.atmospheric_composition.co2_kg > 0.0);
        assert!(experiment.atmospheric_composition.h2o_kg > 0.0);
    }

    #[test]
    fn test_lithosphere_absorption() {
        let mut experiment = WideExperimentWithAtmosphere::new(10, 10000);

        // Test absorption at different depths
        let shallow_depth = 5.0; // km
        let deep_depth = 40.0; // km

        let shallow_absorption = (LITHOSPHERE_ABSORPTION_RATE * shallow_depth).min(0.9);
        let deep_absorption = (LITHOSPHERE_ABSORPTION_RATE * deep_depth).min(0.9);

        println!("Lithosphere absorption test:");
        println!("  Shallow ({:.0} km): {:.1}% absorbed", shallow_depth, shallow_absorption * 100.0);
        println!("  Deep ({:.0} km): {:.1}% absorbed", deep_depth, deep_absorption * 100.0);

        // Deeper layers should have higher absorption
        assert!(deep_absorption > shallow_absorption);
        assert!(shallow_absorption >= 0.0 && shallow_absorption <= 0.9);
        assert!(deep_absorption >= 0.0 && deep_absorption <= 0.9);
    }

    #[test]
    fn test_experiment_initialization() {
        let experiment = WideExperimentWithAtmosphere::new(100, 100000);

        // Verify basic setup
        assert_eq!(experiment.nodes.len(), EXPECTED_NUM_NODES);
        assert!(experiment.atmospheric_layers.len() > 0);
        assert_eq!(experiment.atmospheric_composition.total_mass_kg(), 0.0);

        // Verify temperature gradient
        let surface_temp = experiment.nodes[0].temp_kelvin();
        let foundry_temp = experiment.nodes[experiment.nodes.len() - 1].temp_kelvin();
        assert!(foundry_temp > surface_temp);
        assert!(surface_temp > 200.0); // Reasonable surface temperature
        assert!(foundry_temp < 10000.0); // Reasonable foundry temperature

        println!("Experiment initialization test:");
        println!("  Nodes: {}", experiment.nodes.len());
        println!("  Atmospheric layers: {}", experiment.atmospheric_layers.len());
        println!("  Surface temp: {:.0}K", surface_temp);
        println!("  Foundry temp: {:.0}K", foundry_temp);
    }

    #[test]
    fn test_phase_change_detection() {
        let mut experiment = WideExperimentWithAtmosphere::new(10, 10000);

        // Get the melting and boiling points for silicate
        let melting_point = get_melting_point_k(&MaterialCompositeType::Silicate);
        let boiling_point = get_boiling_point_k(&MaterialCompositeType::Silicate);

        println!("Material phase transition points:");
        println!("  Melting point: {:.0}K", melting_point);
        println!("  Boiling point: {:.0}K", boiling_point);

        // Set some nodes to different temperatures to trigger phase changes
        // Start with solid temperatures, then heat them up
        if let Some(node) = experiment.nodes.get_mut(20) {
            node.set_kelvin(melting_point - 100.0); // Start below melting point
        }
        if let Some(node) = experiment.nodes.get_mut(25) {
            node.set_kelvin(boiling_point - 100.0); // Start below boiling point
        }

        let initial_event_count = experiment.phase_events.events.len();

        // First detection - should be no changes yet
        experiment.detect_phase_changes();

        // Now heat up the nodes to trigger phase changes
        if let Some(node) = experiment.nodes.get_mut(20) {
            node.set_kelvin(melting_point + 100.0); // Above melting point
        }
        if let Some(node) = experiment.nodes.get_mut(25) {
            node.set_kelvin(boiling_point + 100.0); // Above boiling point
        }

        // Detect phase changes after heating
        experiment.detect_phase_changes();

        let final_event_count = experiment.phase_events.events.len();

        println!("Phase change detection test:");
        println!("  Initial events: {}", initial_event_count);
        println!("  Final events: {}", final_event_count);

        // Should have detected some phase changes
        assert!(final_event_count > initial_event_count);

        // Check event types
        let (melting, solidifying, vaporizing, condensing) = experiment.phase_events.count_by_type();
        println!("  Melting: {}, Solidifying: {}, Vaporizing: {}, Condensing: {}",
                 melting, solidifying, vaporizing, condensing);

        // Should have some melting/vaporizing events from hot nodes
        assert!(melting > 0 || vaporizing > 0);
    }

    #[test]
    fn test_phase_change_events_structure() {
        let mut events = PhaseChangeEvents::new(false); // No verbose logging for test

        let test_event = PhaseChangeEvent {
            node_id: 5,
            depth_km: 25.0,
            temperature_k: 1800.0,
            material_type: MaterialCompositeType::Silicate,
            timestamp_years: 1000.0,
            event_type: PhaseChangeType::Melting,
            mass_affected_kg: 1e12,
        };

        events.emit(test_event);

        assert_eq!(events.events.len(), 1);

        let melting_events = events.get_events_by_type(PhaseChangeType::Melting);
        assert_eq!(melting_events.len(), 1);

        let timeframe_events = events.get_events_in_timeframe(500.0, 1500.0);
        assert_eq!(timeframe_events.len(), 1);

        let (melting, solidifying, vaporizing, condensing) = events.count_by_type();
        assert_eq!(melting, 1);
        assert_eq!(solidifying, 0);
        assert_eq!(vaporizing, 0);
        assert_eq!(condensing, 0);

        println!("Phase change events structure test passed!");
    }
}
