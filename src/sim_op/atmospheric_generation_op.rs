/// Atmospheric generation operation
/// Generates atmosphere specifically from MELTING LITHOSPHERE that produces gas.
/// When lithosphere layers melt (transition to liquid/gas), they outgas volatile materials
/// that are added to atmospheric layers with exponentially decreasing density (88% reduction per layer)

use crate::sim_op::SimOp;
use crate::sim::simulation::Simulation;
use crate::energy_mass_composite::{EnergyMassComposite, MaterialPhase, MaterialCompositeType};
use crate::material_composite::{get_emission_compound_ratios, get_profile_fast};
use std::collections::HashMap;
use std::any::Any;
use h3o::CellIndex;

use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

/// Properties of atmospheric compounds loaded from JSON
#[derive(Debug, Clone, Deserialize)]
struct CompoundProperties {
    pub name: String,
    pub chemical_formula: String,
    pub molar_mass_g_mol: f64,
    pub density_stp_kg_m3: f64,
    pub specific_heat_capacity_j_kg_k: f64,
    pub thermal_conductivity_w_m_k: f64,
    pub viscosity_pa_s: f64,
    pub greenhouse_potential: f64,
    pub absorption_bands_um: Vec<f64>,
    pub atmospheric_lifetime_years: f64,
}

/// Load compound data from JSON file
fn load_compounds_data() -> &'static HashMap<String, CompoundProperties> {
    static COMPOUNDS_DATA: OnceLock<HashMap<String, CompoundProperties>> = OnceLock::new();
    COMPOUNDS_DATA.get_or_init(|| {
        let json_str = include_str!("../compounds.json");
        serde_json::from_str(json_str).expect("Failed to parse compounds.json")
    })
}

/// Phase change event for melting detection
#[derive(Debug, Clone)]
pub struct PhaseChangeEvent {
    pub cell_id: CellIndex,
    pub layer_index: usize,
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

/// Atmospheric compound generated from melting events
#[derive(Debug, Clone)]
pub struct AtmosphericCompound {
    pub compound_type: String,  // CO2, H2O, SO2, etc.
    pub mass_kg: f64,
}

/// Global atmospheric properties blended from all compound sources
#[derive(Debug, Clone)]
pub struct GlobalAtmosphericProperties {
    pub total_mass_kg: f64,
    pub average_molar_mass_g_mol: f64,
    pub average_density_stp_kg_m3: f64,
    pub average_specific_heat_j_kg_k: f64,
    pub average_thermal_conductivity_w_m_k: f64,
    pub total_greenhouse_potential: f64,
    pub compound_fractions: HashMap<String, f64>, // compound_name -> mass_fraction
}

impl Default for GlobalAtmosphericProperties {
    fn default() -> Self {
        Self {
            total_mass_kg: 0.0,
            average_molar_mass_g_mol: 28.97, // Default to air
            average_density_stp_kg_m3: 1.225, // Default to air at STP
            average_specific_heat_j_kg_k: 1004.0, // Default to air
            average_thermal_conductivity_w_m_k: 0.024, // Default to air
            total_greenhouse_potential: 0.0,
            compound_fractions: HashMap::new(),
        }
    }
}

/// Parameters for atmospheric generation with crystallization
#[derive(Debug, Clone)]
pub struct CrystallizationParams {
    /// Multiplier for outgassing rate from melting (fraction, 0.0-1.0)
    pub outgassing_rate: f64,
    /// Exponential decay factor for atmospheric volumes (fraction, 0.0-1.0)
    pub volume_decay: f64,
    /// Exponential decay factor for atmospheric density (fraction, 0.0-1.0)
    pub density_decay: f64,
    /// How much deeper sources contribute less (fraction, 0.0-1.0)
    pub depth_attenuation: f64,
    /// Percentage of rising gas that crystallizes per layer (fraction, 0.0-1.0)
    pub crystallization_rate: f64,
    /// Enable debug output
    pub debug: bool,
}

/// Cached crystallization factor for cell/depth combinations
#[derive(Debug, Clone)]
struct CrystallizationCache {
    pub factor: f64,
    pub last_updated_step: usize,
}

/// Cached emission compounds for material types
#[derive(Debug, Clone)]
struct EmissionCache {
    pub compounds: Vec<(String, f64)>,
}

/// Batch of melting events by material type and depth for efficient processing
#[derive(Debug)]
struct MaterialDepthBatch {
    pub material_type: MaterialCompositeType,
    pub depth_km: f64,
    pub total_raw_mass: f64,
    pub max_base_crystallization_rate: f64,
    pub representative_cell_id: CellIndex, // Representative cell for crystallization calculation
}

pub struct AtmosphericGenerationOp {
    pub apply_during_simulation: bool,
    pub outgassing_rate_multiplier: f64,  // Multiplier for outgassing rate from melting
    pub catastrophic_event_outgassing_rate: f64, // Very high rate for catastrophic events (>2500K, rare)
    pub major_event_outgassing_rate: f64, // Higher rate for major volcanic events (>1800K)
    pub background_outgassing_rate: f64,  // Lower rate for background volcanic activity
    pub catastrophic_event_probability: f64, // Probability multiplier for catastrophic events (billion-year scale)
    pub volume_decay_factor: f64,         // Exponential decay factor for atmospheric volumes
    pub density_decay_factor: f64,        // Exponential decay factor for atmospheric density (0.12 = 88% reduction per layer)
    pub depth_attenuation_factor: f64,    // How much deeper sources contribute less (0.8 = 20% reduction per layer depth)
    pub crystallization_rate: f64,        // Base crystallization rate (0.5 = 50% per layer)
    pub catastrophic_event_crystallization: f64, // Very low crystallization for catastrophic events (massive escape)
    pub major_event_crystallization: f64, // Lower crystallization for major events (more gas escapes)
    pub background_crystallization: f64,  // Higher crystallization for background activity (more gas trapped)
    pub debug_output: bool,

    // Event-driven melting tracking
    previous_layer_phases: HashMap<(CellIndex, usize), MaterialPhase>, // Track previous phase state for each layer
    melting_events_this_step: Vec<PhaseChangeEvent>,

    // Global atmospheric tracking - blended from all material sources
    global_atmosphere_compounds: HashMap<String, f64>, // Total compound masses in global atmosphere (all sources blended)
    compounds_added_this_step: HashMap<String, f64>,   // Compounds added this step from all materials
    global_atmospheric_properties: GlobalAtmosphericProperties, // Blended properties of the global atmosphere

    // Performance caches
    crystallization_cache: HashMap<(CellIndex, u32), CrystallizationCache>, // Cache crystallization factors by cell/depth
    emission_cache: HashMap<MaterialCompositeType, EmissionCache>, // Cache emission compounds by material type
    atmospheric_layers_cache: HashMap<CellIndex, Vec<usize>>, // Cache atmospheric layer indices per cell
    
    // Batch processing optimization
    skip_counter: usize,  // Skip processing every N steps for performance
    skip_interval: usize, // Process every N steps instead of every step

    // Tracking state
    total_outgassed_mass: f64,
    total_redistributed_volume: f64,
    total_crystallized_mass: f64,
    step_count: usize,
}

/// Configuration structure for loading atmospheric generation parameters from JSON
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AtmosphericGenerationConfig {
    pub outgassing_rates: OutgassingRatesConfig,
    pub crystallization_rates: CrystallizationRatesConfig,
    pub event_classification: EventClassificationConfig,
    pub foundry_temperature_config: FoundryTemperatureConfig,
    pub atmospheric_layer_config: AtmosphericLayerConfig,
    pub debug_and_reporting: DebugReportingConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OutgassingRatesConfig {
    pub base_outgassing_rate_multiplier: f64,
    pub catastrophic_event_outgassing_rate: f64,
    pub major_event_outgassing_rate: f64,
    pub background_outgassing_rate: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CrystallizationRatesConfig {
    pub base_crystallization_rate: f64,
    pub catastrophic_event_crystallization: f64,
    pub major_event_crystallization: f64,
    pub background_crystallization: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EventClassificationConfig {
    pub catastrophic_threshold_k: f64,
    pub major_event_threshold_k: f64,
    pub catastrophic_event_probability: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FoundryTemperatureConfig {
    pub base_core_temp_k: f64,
    pub surface_temp_k: f64,
    pub geothermal_gradient_k_per_km: f64,
    pub oscillation_enabled: bool,
    pub oscillation_period_years: f64,
    pub min_multiplier: f64,
    pub max_multiplier: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AtmosphericLayerConfig {
    pub volume_decay_factor: f64,
    pub density_decay_factor: f64,
    pub depth_attenuation_factor: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DebugReportingConfig {
    pub debug_output_enabled: bool,
    pub crystallization_debug_threshold_kg: f64,
    pub major_event_debug_threshold_kg: f64,
    pub melting_event_debug_threshold: usize,
}

impl AtmosphericGenerationOp {


    /// Optimized melting event detection with early termination
    fn detect_melting_events_optimized(&mut self, sim: &Simulation) {
        self.melting_events_this_step.clear();

        // Early termination if no cells have changed temperature significantly
        let current_time_years = sim.step as f64 * sim.years_per_step as f64;
        let mut events_found = 0;
        let max_events_to_process = 1000; // Limit events processed per step

        for (cell_id, cell) in &sim.cells {
            if events_found >= max_events_to_process {
                break;
            }

            for (layer_index, (current_layer, _next_layer)) in cell.layers_t.iter().enumerate() {
                // Skip atmospheric layers - they don't contribute to melting
                if matches!(current_layer.energy_mass.material_composite_type(), MaterialCompositeType::Air) {
                    continue;
                }

                let layer_key = (*cell_id, layer_index);
                let current_phase = current_layer.phase();
                let previous_phase = self.previous_layer_phases.get(&layer_key).copied().unwrap_or(MaterialPhase::Solid);

                // Only detect solid->liquid/gas transitions
                if previous_phase == MaterialPhase::Solid &&
                    (current_phase == MaterialPhase::Liquid || current_phase == MaterialPhase::Gas) {

                    let event_type = if current_phase == MaterialPhase::Liquid {
                        PhaseChangeType::Melting
                    } else {
                        PhaseChangeType::Vaporizing
                    };

                    let event = PhaseChangeEvent {
                        cell_id: *cell_id,
                        layer_index,
                        depth_km: current_layer.start_depth_km + current_layer.height_km / 2.0,
                        temperature_k: current_layer.temperature_k(),
                        material_type: current_layer.energy_mass.material_composite_type(),
                        timestamp_years: current_time_years,
                        event_type,
                        mass_affected_kg: current_layer.mass_kg(),
                    };

                    self.melting_events_this_step.push(event);
                    events_found += 1;
                }

                // Update previous phase
                self.previous_layer_phases.insert(layer_key, current_phase);
            }
        }
    }

    /// Optimized atmosphere generation with caching
    fn generate_atmosphere_from_melting_events_optimized(&mut self, sim: &mut Simulation) -> f64 {
        if self.melting_events_this_step.is_empty() {
            return 0.0;
        }

        // Pre-populate emission cache
        for event in &self.melting_events_this_step {
            if !self.emission_cache.contains_key(&event.material_type) {
                let emission_ratios = get_emission_compound_ratios(&event.material_type);
                let compounds_vec: Vec<(String, f64)> = emission_ratios.into_iter().collect();
                self.emission_cache.insert(event.material_type, EmissionCache {
                    compounds: compounds_vec,
                });
            }
        }

        self.compounds_added_this_step.clear();
        let mut total_outgassed = 0.0;

        // Batch events by material type for efficient crystallization processing
        total_outgassed += self.process_events_by_material_batches(sim);

        if total_outgassed > 0.0 {
            self.update_global_atmospheric_properties();
        }

        total_outgassed
    }

    /// Process melting events by batching masses by material type and depth for efficient crystallization
    fn process_events_by_material_batches(&mut self, sim: &Simulation) -> f64 {
        use std::collections::HashMap;
        
        // Group events by (material_type, depth) pairs
        let mut material_depth_batches: HashMap<(MaterialCompositeType, i32), MaterialDepthBatch> = HashMap::new();
        
        // Aggregate events by material type and depth (round to nearest km for batching)
        for event in &self.melting_events_this_step {
            // Determine event type and rates
            let is_catastrophic_candidate = event.temperature_k > 2500.0;
            let is_major_event = event.temperature_k > 1800.0;
            let is_catastrophic_event = is_catastrophic_candidate &&
                (rand::random::<f64>() < self.catastrophic_event_probability);

            let (outgassing_rate, base_crystallization_rate) = if is_catastrophic_event {
                (self.catastrophic_event_outgassing_rate, self.catastrophic_event_crystallization)
            } else if is_major_event {
                (self.major_event_outgassing_rate, self.major_event_crystallization)
            } else {
                (self.background_outgassing_rate, self.background_crystallization)
            };

            let raw_outgassed_mass = event.mass_affected_kg * outgassing_rate;
            let batch_key = (event.material_type, event.depth_km.round() as i32);
            
            let batch = material_depth_batches.entry(batch_key).or_insert_with(|| MaterialDepthBatch {
                material_type: event.material_type,
                depth_km: event.depth_km,
                total_raw_mass: 0.0,
                max_base_crystallization_rate: 0.0,
                representative_cell_id: event.cell_id, // Use first cell as representative
            });

            batch.total_raw_mass += raw_outgassed_mass;
            
            // Store the maximum crystallization rate for this batch
            batch.max_base_crystallization_rate = batch.max_base_crystallization_rate.max(base_crystallization_rate);
        }

        // Process each material-depth batch
        let mut total_outgassed = 0.0;
        for batch in material_depth_batches.values() {
            total_outgassed += self.process_material_depth_batch(sim, batch);
        }

        total_outgassed
    }

    /// Process a single material-depth batch with aggregated mass and crystallization
    fn process_material_depth_batch(&mut self, sim: &Simulation, batch: &MaterialDepthBatch) -> f64 {
        if batch.total_raw_mass <= 0.0 {
            return 0.0;
        }

        // Calculate crystallization factor using the actual depth and representative cell
        let crystallization_factor = self.calculate_crystallization_factor_with_base(
            sim, &batch.representative_cell_id, batch.depth_km, batch.max_base_crystallization_rate
        );

        // Apply crystallization and depth attenuation to total mass
        let total_escaped_mass = batch.total_raw_mass * (1.0 - crystallization_factor);
        let depth_attenuation = self.depth_attenuation_factor.powf(batch.depth_km / 10.0);
        let final_atmospheric_mass = total_escaped_mass * depth_attenuation;

        if final_atmospheric_mass > 0.0 {
            // Use cached emission compounds for this material type
            if let Some(cache) = self.emission_cache.get(&batch.material_type) {
                for (compound_name, ratio) in &cache.compounds {
                    let compound_mass = final_atmospheric_mass * ratio;
                    if compound_mass > 0.0 {
                        *self.global_atmosphere_compounds.entry(compound_name.clone()).or_insert(0.0) += compound_mass;
                        *self.compounds_added_this_step.entry(compound_name.clone()).or_insert(0.0) += compound_mass;
                    }
                }
            }
        }

        final_atmospheric_mass
    }


    /// Optimized atmosphere redistribution with cached atmospheric layers
    fn redistribute_global_atmosphere_to_cells_optimized(&mut self, sim: &mut Simulation) -> f64 {
        if self.global_atmosphere_compounds.is_empty() {
            return 0.0;
        }

        let total_global_mass: f64 = self.global_atmosphere_compounds.values().sum();
        if total_global_mass <= 0.0 {
            return 0.0;
        }

        // Cache atmospheric layers for each cell
        for (cell_id, cell) in &sim.cells {
            if !self.atmospheric_layers_cache.contains_key(cell_id) {
                let mut atmospheric_layers = Vec::new();
                for (i, (current, _)) in cell.layers_t.iter().enumerate() {
                    if matches!(current.energy_mass.material_composite_type(), MaterialCompositeType::Air) {
                        atmospheric_layers.push(i);
                    }
                }
                self.atmospheric_layers_cache.insert(*cell_id, atmospheric_layers);
            }
        }

        let mut total_redistributed = 0.0;

        // Calculate total surface area first to avoid borrowing issues
        let total_cells = sim.cells.len() as f64;
        let cell_surface_area_m2 = if let Some((layer, _)) = sim.cells.values().next().and_then(|c| c.layers_t.first()) {
            layer.surface_area_km2 * 1e6 // Convert km² to m²
        } else {
            return 0.0; // No layers
        };
        let total_surface_area_m2 = total_cells * cell_surface_area_m2;

        for cell in sim.cells.values_mut() {
            if let Some(atmospheric_layers) = self.atmospheric_layers_cache.get(&cell.h3_index) {
                if atmospheric_layers.is_empty() {
                    continue;
                }

                // Calculate this cell's share of total atmospheric mass
                let cell_total_mass = total_global_mass * (cell_surface_area_m2 / total_surface_area_m2);

                // Distribute using proper exponential atmospheric model
                let layer_masses = self.calculate_exponential_atmospheric_distribution(
                    cell, atmospheric_layers, cell_total_mass
                );

                // Apply the calculated masses to layers
                for (layer_index, mass_to_add) in layer_masses {
                    if let Some((current, _next)) = cell.layers_t.get_mut(layer_index) {
                        if mass_to_add > 0.0 {
                            let temp_k = current.temperature_k();
                            // Add atmospheric mass using StandardEnergyMassComposite
                            current.energy_mass.add_atmospheric_mass(mass_to_add, temp_k);
                            
                            // Update layer density based on mass and volume
                            self.update_layer_density(current, mass_to_add);
                            
                            total_redistributed += mass_to_add;
                        }
                    }
                }
            }
        }

        total_redistributed
    }

    /// Calculate exponential atmospheric distribution following barometric formula
    fn calculate_exponential_atmospheric_distribution(
        &self,
        cell: &crate::global_thermal::sim_cell::SimCell,
        atmospheric_layers: &[usize],
        total_cell_mass: f64,
    ) -> Vec<(usize, f64)> {
        if atmospheric_layers.is_empty() || total_cell_mass <= 0.0 {
            return Vec::new();
        }

        // Atmospheric parameters (Earth-like defaults)
        let gas_constant = 8.314; // J/(mol·K)
        let molar_mass = 0.028964; // kg/mol for dry air
        let gravity = 9.81; // m/s²
        let temperature = 288.0; // K (average tropospheric temperature)
        
        // Calculate scale height: H = (R * T) / (M * g)
        let scale_height_m = (gas_constant * temperature) / (molar_mass * gravity);
        
        // Sample atmospheric distribution every 4 miles (6.44 km) up to reasonable altitude
        let sample_interval_m = 6440.0; // 4 miles in meters
        let max_altitude_m = scale_height_m * 5.0; // 5 scale heights covers ~99% of atmosphere
        
        // Calculate mass distribution at sample points
        let mut sample_masses = Vec::new();
        let mut total_sample_mass = 0.0;
        
        for i in 0..((max_altitude_m / sample_interval_m) as usize) {
            let altitude_m = i as f64 * sample_interval_m;
            let density_fraction = (-altitude_m / scale_height_m).exp();
            let mass_at_sample = density_fraction * sample_interval_m; // Mass per unit area
            sample_masses.push((altitude_m, mass_at_sample));
            total_sample_mass += mass_at_sample;
        }
        
        // Redistribute sample masses into actual atmospheric layers
        let mut layer_masses = Vec::new();
        
        for &layer_index in atmospheric_layers {
            if let Some((current_layer, _)) = cell.layers_t.get(layer_index) {
                let layer_bottom_m = (-current_layer.start_depth_km) * 1000.0; // Convert to meters, flip sign
                let layer_top_m = layer_bottom_m + (current_layer.height_km * 1000.0);
                
                // Only process layers above ground (positive altitude)
                if layer_top_m > 0.0 {
                    let layer_bottom_altitude = layer_bottom_m.max(0.0);
                    let layer_top_altitude = layer_top_m.max(0.0);
                    
                    // Calculate mass fraction for this layer by interpolating sample points
                    let layer_mass_fraction = self.interpolate_atmospheric_mass_fraction(
                        &sample_masses, 
                        layer_bottom_altitude, 
                        layer_top_altitude,
                        total_sample_mass
                    );
                    
                    let layer_mass = total_cell_mass * layer_mass_fraction;
                    layer_masses.push((layer_index, layer_mass));
                }
            }
        }
        
        layer_masses
    }

    /// Interpolate atmospheric mass fraction for a layer altitude range
    fn interpolate_atmospheric_mass_fraction(
        &self,
        sample_masses: &[(f64, f64)], // (altitude_m, mass_fraction)
        layer_bottom_m: f64,
        layer_top_m: f64,
        total_sample_mass: f64,
    ) -> f64 {
        if sample_masses.is_empty() || total_sample_mass <= 0.0 {
            return 0.0;
        }
        
        let mut layer_mass = 0.0;
        
        // Find samples that overlap with this layer
        for (altitude, mass_at_sample) in sample_masses {
            if *altitude >= layer_bottom_m && *altitude < layer_top_m {
                layer_mass += mass_at_sample;
            }
        }
        
        // Normalize to fraction of total mass
        layer_mass / total_sample_mass
    }

    /// Update layer density based on mass and volume
    fn update_layer_density(&self, _layer: &mut crate::global_thermal::thermal_layer::ThermalLayer, _added_mass: f64) {
        // For StandardEnergyMassComposite atmospheric layers, density is automatically calculated
        // from the energy content via mass_kg() / volume. The add_atmospheric_mass() method
        // updates the energy content, which automatically updates the density calculation.
        // No additional density update is needed.
    }

    /// Get cached crystallization factor with lazy computation
    fn get_cached_crystallization_factor(
        &mut self,
        sim: &Simulation,
        cell_id: &CellIndex,
        depth_km: f64,
        base_rate: f64,
    ) -> f64 {
        let depth_key = (depth_km * 10.0) as u32; // Round to 100m precision
        let cache_key = (*cell_id, depth_key);

        // Check cache first
        if let Some(cached) = self.crystallization_cache.get(&cache_key) {
            if self.step_count - cached.last_updated_step < 50 { // Cache valid for 50 steps
                return cached.factor;
            }
        }

        // Compute and cache
        let factor = self.calculate_crystallization_factor_with_base(sim, cell_id, depth_km, base_rate);
        self.crystallization_cache.insert(cache_key, CrystallizationCache {
            factor,
            last_updated_step: self.step_count,
        });

        factor
    }
    /// Detect phase changes by comparing current and previous layer phases
    fn detect_melting_events(&mut self, sim: &Simulation) {
        self.melting_events_this_step.clear();

        let current_time_years = sim.step as f64 * sim.years_per_step as f64;

        for (cell_id, cell) in &sim.cells {
            for (layer_index, (current_layer, _next_layer)) in cell.layers_t.iter().enumerate() {
                let layer_key = (*cell_id, layer_index);
                let current_phase = current_layer.phase();

                // Get previous phase (default to Solid for new layers)
                let previous_phase = self.previous_layer_phases.get(&layer_key).copied().unwrap_or(MaterialPhase::Solid);

                // Detect melting transition (Solid -> Liquid or Solid -> Gas)
                if previous_phase == MaterialPhase::Solid &&
                   (current_phase == MaterialPhase::Liquid || current_phase == MaterialPhase::Gas) {

                    let event_type = if current_phase == MaterialPhase::Liquid {
                        PhaseChangeType::Melting
                    } else {
                        PhaseChangeType::Vaporizing
                    };

                    let event = PhaseChangeEvent {
                        cell_id: *cell_id,
                        layer_index,
                        depth_km: current_layer.start_depth_km + current_layer.height_km / 2.0,
                        temperature_k: current_layer.temperature_k(),
                        material_type: current_layer.energy_mass.material_composite_type(),
                        timestamp_years: current_time_years,
                        event_type,
                        mass_affected_kg: current_layer.mass_kg(),
                    };

                    if self.debug_output {
                        println!("🔥 MELTING EVENT: Cell {:016x}, Layer {}, {:.1}km depth, {:.2e} kg melted at {:.0}K",
                                 u64::from(*cell_id), layer_index, event.depth_km, event.mass_affected_kg, event.temperature_k);
                    }

                    self.melting_events_this_step.push(event);
                }

                // Update previous phase for next step
                self.previous_layer_phases.insert(layer_key, current_phase);
            }
        }

        // Reduce debug output frequency
        if self.debug_output && self.melting_events_this_step.len() > 1000 {
            println!("🌋 Detected {} melting events this step", self.melting_events_this_step.len());
        }
    }

    /// Generate atmosphere from melting events (global approach)
    fn generate_atmosphere_from_melting_events(&mut self, sim: &mut Simulation) -> f64 {
        if self.melting_events_this_step.is_empty() {
            return 0.0;
        }

        // Clear compounds added this step
        self.compounds_added_this_step.clear();
        let mut total_outgassed = 0.0;

        // Clone events to avoid borrowing issues
        let events = self.melting_events_this_step.clone();

        // Process all melting events and add compounds to global atmosphere
        for event in &events {
            // Determine event type based on temperature and probability
            let is_catastrophic_candidate = event.temperature_k > 2500.0; // Catastrophic events above 2500K
            let is_major_event = event.temperature_k > 1800.0; // Major events above 1800K

            // Apply probability filter for catastrophic events (billion-year scale)
            let is_catastrophic_event = is_catastrophic_candidate &&
                (rand::random::<f64>() < self.catastrophic_event_probability);

            // Use different rates based on event type
            let (outgassing_rate, base_crystallization_rate, event_type) = if is_catastrophic_event {
                (self.catastrophic_event_outgassing_rate, self.catastrophic_event_crystallization, "CATASTROPHIC")
            } else if is_major_event {
                (self.major_event_outgassing_rate, self.major_event_crystallization, "MAJOR")
            } else {
                (self.background_outgassing_rate, self.background_crystallization, "background")
            };

            // Calculate outgassing from this melting event
            let raw_outgassed_mass = event.mass_affected_kg * outgassing_rate;

            // Calculate crystallization based on overlying layer density and depth
            let crystallization_factor = self.calculate_crystallization_factor_with_base(
                sim, &event.cell_id, event.depth_km, base_crystallization_rate
            );
            let crystallized_mass = raw_outgassed_mass * crystallization_factor;
            let escaped_mass = raw_outgassed_mass - crystallized_mass;

            // Debug crystallization calculation for significant events only
            if self.debug_output && raw_outgassed_mass > 1e18 {
                println!("🔬 Crystallization debug: raw={:.2e}kg, factor={:.3}, crystallized={:.2e}kg, escaped={:.2e}kg",
                         raw_outgassed_mass, crystallization_factor, crystallized_mass, escaped_mass);
            }

            // Apply depth attenuation to escaped mass only
            let depth_attenuation = self.depth_attenuation_factor.powf(event.depth_km / 10.0);
            let final_atmospheric_mass = escaped_mass * depth_attenuation;

            if final_atmospheric_mass > 0.0 {
                // Generate specific atmospheric compounds based on material type
                let compounds = self.generate_atmospheric_compounds(&event.material_type, final_atmospheric_mass);

                // Add compounds to global atmosphere
                for compound in &compounds {
                    // Add to global atmosphere
                    *self.global_atmosphere_compounds.entry(compound.compound_type.clone()).or_insert(0.0) += compound.mass_kg;

                    // Track compounds added this step
                    *self.compounds_added_this_step.entry(compound.compound_type.clone()).or_insert(0.0) += compound.mass_kg;
                }

                // Minimal debug output for major events only
                if self.debug_output && final_atmospheric_mass > 1e19 && event_type == "CATASTROPHIC" {
                    println!("🌋 {} volcanic event: {:.2e} kg from {:.1}km depth at {:.0}K",
                             event_type, final_atmospheric_mass, event.depth_km, event.temperature_k);
                }

                total_outgassed += final_atmospheric_mass;
            }
        }

        // Update global atmospheric properties by blending all compounds
        if total_outgassed > 0.0 {
            self.update_global_atmospheric_properties();
        }

        if self.debug_output && total_outgassed > 1e19 {
            println!("🌍 Total atmospheric generation: {:.2e} kg from {} events",
                     total_outgassed, events.len());
        }

        total_outgassed
    }

    /// Generate specific atmospheric compounds based on material emission ratios
    fn generate_atmospheric_compounds(&self, material_type: &MaterialCompositeType, total_mass_kg: f64) -> Vec<AtmosphericCompound> {
        let emission_ratios = get_emission_compound_ratios(material_type);
        let mut compounds = Vec::new();

        for (compound_name, ratio) in emission_ratios {
            let compound_mass = total_mass_kg * ratio;
            if compound_mass > 0.0 {
                compounds.push(AtmosphericCompound {
                    compound_type: compound_name,
                    mass_kg: compound_mass,
                });
            }
        }

        compounds
    }

    /// Calculate crystallization factor based on overlying layer properties
    /// Accounts for pressure, density, material type, and interference factors
    fn calculate_crystallization_factor(&self, sim: &Simulation, cell_id: &CellIndex, melting_depth_km: f64) -> f64 {
        self.calculate_crystallization_factor_with_base(sim, cell_id, melting_depth_km, self.crystallization_rate)
    }

    /// Calculate crystallization factor with custom base rate for different event types
    fn calculate_crystallization_factor_with_base(&self, sim: &Simulation, cell_id: &CellIndex, melting_depth_km: f64, base_rate: f64) -> f64 {
        if let Some(cell) = sim.cells.get(cell_id) {
            let mut total_crystallization_factor = 0.0;
            let mut contributing_layers = 0;

            // Analyze each layer above the melting depth - keep them individually
            for (current, _next) in &cell.layers_t {
                let layer_top = current.start_depth_km;
                let layer_bottom = current.start_depth_km + current.height_km;

                // Only count layers above the melting depth AND skip atmospheric layers
                if layer_bottom <= melting_depth_km && !self.is_atmospheric_layer(current) {
                    // Calculate layer-specific properties
                    let layer_density_kg_m3 = current.current_density_kg_m3();
                    let layer_pressure_gpa = current.energy_mass.pressure_gpa();
                    let layer_material = current.energy_mass.material_composite_type();
                    let layer_phase = current.energy_mass.phase();

                    // Material-specific interference factors - use gas interference factor from material composite
                    let material_profile = get_profile_fast(&layer_material, &layer_phase);
                    let material_interference_factor = material_profile.gas_interference_factor;

                    // Pressure factor (higher pressure = more crystallization)
                    let pressure_factor = (1.0 + layer_pressure_gpa * 0.5).min(3.0); // Cap at 3x

                    // Density factor (denser material = more crystallization)
                    let reference_density = 2500.0; // kg/m³ - typical rock density
                    let density_factor = (layer_density_kg_m3 / reference_density).min(4.0); // Cap at 4x

                    // Distance factor (closer to melting source = more effective)
                    let distance_from_melting = (melting_depth_km - layer_bottom).max(0.1);
                    let distance_factor = (10.0 / distance_from_melting).min(2.0); // Closer layers more effective

                    // Layer thickness factor (thicker layers = more crystallization)
                    let thickness_factor = (current.height_km / 4.0).min(2.0); // Normalize to 4km reference

                    // Combine all factors for this layer
                    let layer_crystallization = material_interference_factor
                        * pressure_factor
                        * density_factor
                        * distance_factor
                        * thickness_factor;

                    total_crystallization_factor += layer_crystallization;
                    contributing_layers += 1;

                    // Disable layer-by-layer debug output for cleaner logs
                }
            }

            if contributing_layers > 0 {
                // Average crystallization factor from all contributing layers
                let average_crystallization = total_crystallization_factor / contributing_layers as f64;

                // Apply base crystallization rate more conservatively
                let base_crystallization = base_rate / 100.0; // Convert percentage to fraction
                // Use additive rather than multiplicative to prevent extreme values
                let final_crystallization = (base_crystallization + average_crystallization * 0.1).min(0.95);

                // Cap crystallization at 99.9% (always allow some gas to escape)
                final_crystallization.min(0.999)
            } else {
                // No overlying layers - minimal crystallization
                (base_rate / 100.0).min(0.1)
            }
        } else {
            // Cell not found - use base crystallization rate
            (base_rate / 100.0).min(0.5)
        }
    }

    /// Get material-specific interference factor for gas crystallization
    /// Different materials have different abilities to trap rising gases

    /// Check if a layer is atmospheric (should be skipped for crystallization)
    fn is_atmospheric_layer(&self, layer: &crate::global_thermal::thermal_layer::ThermalLayer) -> bool {
        // Atmospheric layers are typically:
        // 1. Above surface (negative depth)
        // 2. Air material type
        // 3. Gas phase
        // 4. Very low density

        let is_above_surface = layer.start_depth_km < 0.0;
        let is_air_material = matches!(layer.energy_mass.material_composite_type(), MaterialCompositeType::Air);
        let is_gas_phase = matches!(layer.energy_mass.phase(), MaterialPhase::Gas);
        let is_low_density = layer.current_density_kg_m3() < 10.0; // Very low density threshold

        // A layer is atmospheric if it meets most of these criteria
        let atmospheric_indicators = [is_above_surface, is_air_material, is_gas_phase, is_low_density];
        let indicator_count = atmospheric_indicators.iter().filter(|&&x| x).count();

        // Consider it atmospheric if it meets at least 3 out of 4 criteria
        indicator_count >= 3
    }

    /// Update global atmospheric properties by blending all compounds from all material sources
    fn update_global_atmospheric_properties(&mut self) {
        if self.global_atmosphere_compounds.is_empty() {
            self.global_atmospheric_properties = GlobalAtmosphericProperties::default();
            return;
        }

        // Calculate total mass and compound fractions
        let total_mass: f64 = self.global_atmosphere_compounds.values().sum();
        if total_mass <= 0.0 {
            self.global_atmospheric_properties = GlobalAtmosphericProperties::default();
            return;
        }

        // Load compound properties from JSON
        let compounds_data = load_compounds_data();

        // Calculate mass-weighted averages of all properties
        let mut weighted_molar_mass = 0.0;
        let mut weighted_density = 0.0;
        let mut weighted_specific_heat = 0.0;
        let mut weighted_thermal_conductivity = 0.0;
        let mut total_greenhouse_potential = 0.0;
        let mut compound_fractions = HashMap::new();

        for (compound_name, &compound_mass) in &self.global_atmosphere_compounds {
            let mass_fraction = compound_mass / total_mass;
            compound_fractions.insert(compound_name.clone(), mass_fraction);

            if let Some(properties) = compounds_data.get(compound_name) {
                // Mass-weighted averages
                weighted_molar_mass += mass_fraction * properties.molar_mass_g_mol;
                weighted_density += mass_fraction * properties.density_stp_kg_m3;
                weighted_specific_heat += mass_fraction * properties.specific_heat_capacity_j_kg_k;
                weighted_thermal_conductivity += mass_fraction * properties.thermal_conductivity_w_m_k;

                // Greenhouse potential (additive based on mass fraction)
                total_greenhouse_potential += mass_fraction * properties.greenhouse_potential;
            } else {
                // Default to air properties for unknown compounds
                weighted_molar_mass += mass_fraction * 28.97;
                weighted_density += mass_fraction * 1.225;
                weighted_specific_heat += mass_fraction * 1004.0;
                weighted_thermal_conductivity += mass_fraction * 0.024;
            }
        }

        // Update global atmospheric properties
        self.global_atmospheric_properties = GlobalAtmosphericProperties {
            total_mass_kg: total_mass,
            average_molar_mass_g_mol: weighted_molar_mass,
            average_density_stp_kg_m3: weighted_density,
            average_specific_heat_j_kg_k: weighted_specific_heat,
            average_thermal_conductivity_w_m_k: weighted_thermal_conductivity,
            total_greenhouse_potential,
            compound_fractions: compound_fractions.clone(),
        };

        if self.debug_output && total_mass > 1e20 {
            println!("🌍 Global atmosphere: {:.2e} kg, greenhouse: {:.2}",
                     total_mass, total_greenhouse_potential);

            // Show top 3 compounds only
            let mut sorted_compounds: Vec<_> = compound_fractions.iter().collect();
            sorted_compounds.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
            for (compound, fraction) in sorted_compounds.iter().take(3) {
                if **fraction > 0.05 { // Only show compounds > 5%
                    println!("   {}: {:.1}%", compound, *fraction * 100.0);
                }
            }
        }
    }

    /// Redistribute global atmosphere to all cells with exponential distribution
    fn redistribute_global_atmosphere_to_cells(&mut self, sim: &mut Simulation) -> f64 {
        if self.global_atmosphere_compounds.is_empty() {
            return 0.0;
        }

        // Calculate total global atmospheric mass
        let total_global_mass: f64 = self.global_atmosphere_compounds.values().sum();
        if total_global_mass <= 0.0 {
            return 0.0;
        }

        // Calculate column mass per unit area (total mass / total surface area)
        let total_surface_area_m2 = sim.cells.len() as f64 * 1.0; // Approximate - should use actual H3 cell areas
        let column_mass_kg_m2 = total_global_mass / total_surface_area_m2;

        if self.debug_output && total_global_mass > 1e20 {
            println!("🌍 Redistributing {:.2e} kg to {} cells", total_global_mass, sim.cells.len());
        }

        // Redistribute to each cell with exponential distribution by altitude
        let mut total_redistributed = 0.0;

        for cell in sim.cells.values_mut() {
            // Find atmospheric layers
            let mut atmospheric_layers = Vec::new();
            for (i, (current, _)) in cell.layers_t.iter().enumerate() {
                if matches!(current.energy_mass.material_composite_type(), MaterialCompositeType::Air) {
                    atmospheric_layers.push(i);
                }
            }

            if atmospheric_layers.is_empty() {
                continue;
            }

            // Calculate exponential distribution parameters
            let _surface_pressure_pa = 101325.0; // Standard atmospheric pressure
            let scale_height_km = 8.0; // Earth-like scale height

            // Distribute mass to atmospheric layers with exponential falloff
            for &layer_index in &atmospheric_layers {
                if let Some((current, _next)) = cell.layers_t.get_mut(layer_index) {
                    // Calculate altitude (negative depth means above surface)
                    let altitude_km = (-current.start_depth_km - current.height_km / 2.0).max(0.0);

                    // Exponential distribution: mass(h) = mass_0 * exp(-h/H)
                    let altitude_factor = (-altitude_km / scale_height_km).exp();
                    let layer_mass_fraction = altitude_factor / atmospheric_layers.len() as f64; // Normalize

                    // Calculate mass for this layer (equal share of column mass for this cell)
                    let cell_surface_area_m2 = current.surface_area_km2 * 1e6; // Convert km² to m²
                    let cell_column_mass = column_mass_kg_m2 * cell_surface_area_m2;
                    let layer_mass_to_add = cell_column_mass * layer_mass_fraction;

                    if layer_mass_to_add > 0.0 {
                        // Add mass using atmospheric mass addition (maintains constant volume)
                        let temp_k = current.temperature_k();
                        // Add atmospheric mass using StandardEnergyMassComposite
                        current.energy_mass.add_atmospheric_mass(layer_mass_to_add, temp_k);

                        // Add blended atmospheric compounds to this layer
                        for (compound_name, &global_fraction) in &self.global_atmospheric_properties.compound_fractions {
                            let compound_mass = layer_mass_to_add * global_fraction;
                            if compound_mass > 0.0 {
                                current.add_atmospheric_compound(compound_name.clone(), compound_mass);
                            }
                        }

                        total_redistributed += layer_mass_to_add;

                        // Minimal debug output for redistribution
                        if self.debug_output && layer_index == atmospheric_layers[0] && layer_mass_to_add > 1e15 {
                            println!("   Added {:.2e} kg to layer {} at {:.1}km",
                                     layer_mass_to_add, layer_index, altitude_km);
                        }
                    }
                }
            }
        }

        // Removed verbose redistribution debug output

        total_redistributed
    }

    /// Distribute outgassed mass to atmospheric layers with crystallization
    fn distribute_outgassed_mass_to_atmosphere(&mut self, cell: &mut crate::global_thermal::sim_cell::SimCell, outgassed_mass: f64) {
        if outgassed_mass <= 0.0 {
            return;
        }

        // Find atmospheric layers (Air material type)
        let mut atmo_layer_indices: Vec<usize> = Vec::new();
        for (i, (current, _)) in cell.layers_t.iter().enumerate() {
            if matches!(current.energy_mass.material_composite_type(), MaterialCompositeType::Air) {
                atmo_layer_indices.push(i);
            }
        }

        if atmo_layer_indices.is_empty() {
            if self.debug_output {
                println!("⚠️  No atmospheric layers found to distribute outgassed mass");
            }
            return;
        }

        // Distribute mass with exponential decay and crystallization
        let mut remaining_mass = outgassed_mass;
        let mut total_crystallized = 0.0;

        for (_layer_order, &layer_index) in atmo_layer_indices.iter().enumerate() {
            if remaining_mass <= 0.0 {
                break;
            }

            // Calculate how much mass reaches this layer (with crystallization loss)
            let crystallization_loss = remaining_mass * self.crystallization_rate;
            let mass_to_add = remaining_mass - crystallization_loss;

            if mass_to_add > 0.0 {
                // Add mass to this atmospheric layer using the new atmospheric mass addition method
                if let Some((current, _next)) = cell.layers_t.get_mut(layer_index) {
                    let temp_k = current.temperature_k();
                    // Add atmospheric mass using StandardEnergyMassComposite
                    current.energy_mass.add_atmospheric_mass(mass_to_add, temp_k);

                    // Mass is now tracked in global atmosphere

                    if self.debug_output {
                        println!("   Layer {}: Added {:.2e} kg atmospheric mass (crystallized: {:.2e} kg), new mass: {:.2e} kg",
                                 layer_index, mass_to_add, crystallization_loss, current.mass_kg());
                    }
                }
            }

            total_crystallized += crystallization_loss;
            remaining_mass = crystallization_loss; // What crystallizes becomes available for next layer
        }

        self.total_crystallized_mass += total_crystallized;

        if self.debug_output && total_crystallized > 0.0 {
            println!("   Total crystallization loss: {:.2e} kg ({:.1}% of outgassed material)",
                     total_crystallized, (total_crystallized / outgassed_mass) * 100.0);
        }
    }
    pub fn new() -> Self {
        Self {
            apply_during_simulation: true,
            outgassing_rate_multiplier: 0.01, // Base outgassing rate (will be overridden by event-specific rates)
            catastrophic_event_outgassing_rate: 0.0001, // 0.01% for catastrophic events (>2500K, billion-year scale)
            major_event_outgassing_rate: 0.000001, // 0.0001% for major volcanic events (>1800K)
            background_outgassing_rate: 0.0000001, // 0.00001% for background volcanic activity
            catastrophic_event_probability: 0.001, // 0.1% chance - truly rare catastrophic events
            volume_decay_factor: 0.7,         // Each layer up has 70% of the volume below
            density_decay_factor: 0.12,       // Each layer up has 12% of the density below (88% reduction)
            depth_attenuation_factor: 0.8,    // Deeper sources contribute 80% as much (20% reduction per layer)
            crystallization_rate: 0.5,        // Base crystallization rate
            catastrophic_event_crystallization: 0.05, // 5% crystallization for catastrophic events (95% escapes)
            major_event_crystallization: 0.3,  // 30% crystallization for major events (70% escapes)
            background_crystallization: 0.8,   // 80% crystallization for background (20% escapes)
            debug_output: false,
            previous_layer_phases: HashMap::new(),
            melting_events_this_step: Vec::new(),
            global_atmosphere_compounds: HashMap::new(),
            compounds_added_this_step: HashMap::new(),
            global_atmospheric_properties: GlobalAtmosphericProperties::default(),
            crystallization_cache: HashMap::new(),
            emission_cache: HashMap::new(),
            atmospheric_layers_cache: HashMap::new(),
            skip_counter: 0,
            skip_interval: 3, // Process every 3rd step for performance
            total_outgassed_mass: 0.0,
            total_redistributed_volume: 0.0,
            total_crystallized_mass: 0.0,
            step_count: 0,
        }
    }

    pub fn with_params(outgassing_rate: f64, volume_decay: f64, debug: bool) -> Self {
        Self {
            apply_during_simulation: true,
            outgassing_rate_multiplier: outgassing_rate,
            catastrophic_event_outgassing_rate: outgassing_rate * 0.01, // 1% of base rate for catastrophic events
            major_event_outgassing_rate: outgassing_rate * 0.0001, // 0.01% of base rate for major events
            background_outgassing_rate: outgassing_rate * 0.00001, // 0.001% of base rate for background
            catastrophic_event_probability: 0.001,              // 0.1% chance for catastrophic events
            volume_decay_factor: volume_decay,
            density_decay_factor: 0.12,       // Each layer up has 12% of the density below (88% reduction)
            depth_attenuation_factor: 0.8,    // Deeper sources contribute 80% as much (20% reduction per layer)
            crystallization_rate: 0.5,        // Base crystallization rate
            catastrophic_event_crystallization: 0.05, // 5% crystallization for catastrophic events
            major_event_crystallization: 0.3,  // 30% crystallization for major events
            background_crystallization: 0.8,   // 80% crystallization for background
            debug_output: debug,
            previous_layer_phases: HashMap::new(),
            melting_events_this_step: Vec::new(),
            global_atmosphere_compounds: HashMap::new(),
            compounds_added_this_step: HashMap::new(),
            global_atmospheric_properties: GlobalAtmosphericProperties::default(),
            crystallization_cache: HashMap::new(),
            emission_cache: HashMap::new(),
            atmospheric_layers_cache: HashMap::new(),
            skip_counter: 0,
            skip_interval: 3, // Process every 3rd step for performance
            total_outgassed_mass: 0.0,
            total_redistributed_volume: 0.0,
            total_crystallized_mass: 0.0,
            step_count: 0,
        }
    }

    pub fn with_full_params(outgassing_rate: f64, volume_decay: f64, density_decay: f64, debug: bool) -> Self {
        Self {
            apply_during_simulation: true,
            outgassing_rate_multiplier: outgassing_rate,
            catastrophic_event_outgassing_rate: outgassing_rate * 0.01, // 1% of base rate for catastrophic events
            major_event_outgassing_rate: outgassing_rate * 0.0001, // 0.01% of base rate for major events
            background_outgassing_rate: outgassing_rate * 0.00001, // 0.001% of base rate for background
            catastrophic_event_probability: 0.001,              // 0.1% chance for catastrophic events
            volume_decay_factor: volume_decay,
            density_decay_factor: density_decay,
            depth_attenuation_factor: 0.8,    // Deeper sources contribute 80% as much (20% reduction per layer)
            crystallization_rate: 0.5,        // Base crystallization rate
            catastrophic_event_crystallization: 0.05, // 5% crystallization for catastrophic events
            major_event_crystallization: 0.3,  // 30% crystallization for major events
            background_crystallization: 0.8,   // 80% crystallization for background
            debug_output: debug,
            previous_layer_phases: HashMap::new(),
            melting_events_this_step: Vec::new(),
            global_atmosphere_compounds: HashMap::new(),
            compounds_added_this_step: HashMap::new(),
            global_atmospheric_properties: GlobalAtmosphericProperties::default(),
            crystallization_cache: HashMap::new(),
            emission_cache: HashMap::new(),
            atmospheric_layers_cache: HashMap::new(),
            skip_counter: 0,
            skip_interval: 3, // Process every 3rd step for performance
            total_outgassed_mass: 0.0,
            total_redistributed_volume: 0.0,
            total_crystallized_mass: 0.0,
            step_count: 0,
        }
    }

    pub fn with_crystallization_params(params: CrystallizationParams) -> Self {
        Self {
            apply_during_simulation: true,
            outgassing_rate_multiplier: params.outgassing_rate,
            catastrophic_event_outgassing_rate: params.outgassing_rate * 0.01, // 1% of base rate for catastrophic events
            major_event_outgassing_rate: params.outgassing_rate * 0.0001, // 0.01% of base rate for major events
            background_outgassing_rate: params.outgassing_rate * 0.00001, // 0.001% of base rate for background
            catastrophic_event_probability: 0.001,                     // 0.1% chance for catastrophic events
            volume_decay_factor: params.volume_decay,
            density_decay_factor: params.density_decay,
            depth_attenuation_factor: params.depth_attenuation,
            crystallization_rate: params.crystallization_rate,
            catastrophic_event_crystallization: 0.05, // 5% crystallization for catastrophic events
            major_event_crystallization: 0.3,  // 30% crystallization for major events
            background_crystallization: 0.8,   // 80% crystallization for background
            debug_output: params.debug,
            previous_layer_phases: HashMap::new(),
            melting_events_this_step: Vec::new(),
            global_atmosphere_compounds: HashMap::new(),
            compounds_added_this_step: HashMap::new(),
            global_atmospheric_properties: GlobalAtmosphericProperties::default(),
            crystallization_cache: HashMap::new(),
            emission_cache: HashMap::new(),
            atmospheric_layers_cache: HashMap::new(),
            skip_counter: 0,
            skip_interval: 3, // Process every 3rd step for performance
            total_outgassed_mass: 0.0,
            total_redistributed_volume: 0.0,
            total_crystallized_mass: 0.0,
            step_count: 0,
        }
    }
    
    /// Check for melting lithosphere layers and generate atmospheric gas from outgassing
    fn process_outgassing(&mut self, cell: &mut crate::global_thermal::sim_cell::SimCell) -> f64 {
        let mut total_outgassed = 0.0;

        // Find atmospheric and lithosphere layer ranges
        let mut atmo_end_index = 0;
        let mut lith_start_index = 0;
        let mut lith_end_index = 0;

        for (i, (current, _)) in cell.layers_t.iter().enumerate() {
            if current.is_atmospheric() {
                atmo_end_index = i + 1;
            } else if lith_start_index == 0 {
                lith_start_index = i;
            }
        }

        // Find end of lithosphere (before asthenosphere)
        for i in lith_start_index..cell.layers_t.len() {
            let (current, _) = &cell.layers_t[i];
            if current.is_surface_layer ||
               (i > lith_start_index && current.start_depth_km < cell.layers_t[i-1].0.start_depth_km + 50.0) {
                lith_end_index = i + 1;
            } else {
                break;
            }
        }

        // Check lithosphere layers for melting that generates gas
        for i in lith_start_index..lith_end_index {
            let (current, _) = &mut cell.layers_t[i];

            // Only process outgassing when lithosphere melts to liquid or gas phase
            let current_phase = current.energy_mass.phase();
            let material_type = current.energy_mass.material_composite_type();

            // Atmospheric generation occurs when solid lithosphere melts
            if current_phase == MaterialPhase::Liquid || current_phase == MaterialPhase::Gas {
                let layer_mass = current.energy_mass.mass_kg();
                let layer_temp = current.energy_mass.kelvin();

                // Calculate outgassing rate based on how much above melting point
                let melting_point = crate::material_composite::get_melting_point_k(&material_type);
                let temp_excess = (layer_temp - melting_point).max(0.0);

                // More outgassing at higher temperatures above melting point
                let base_outgas_rate = self.calculate_outgassing_rate(&material_type);
                let temp_multiplier = 1.0 + (temp_excess / 500.0); // Extra outgassing per 500K above melting
                let effective_outgas_rate = base_outgas_rate * temp_multiplier;

                // Apply depth attenuation: deeper layers contribute less
                let depth_from_surface = i - atmo_end_index; // Layers below atmosphere
                let depth_attenuation = self.depth_attenuation_factor.powi(depth_from_surface as i32);

                let raw_outgassed_mass = layer_mass * effective_outgas_rate * self.outgassing_rate_multiplier;
                let depth_attenuated_mass = raw_outgassed_mass * depth_attenuation;

                if depth_attenuated_mass > 0.0 {
                    total_outgassed += depth_attenuated_mass;

                    if self.debug_output {
                        println!("Lithosphere layer {} melting -> outgassing: {:.2e} kg (depth attenuation: {:.3}, phase: {:?}, temp: {:.0}K)",
                                i, depth_attenuated_mass, depth_attenuation, current_phase, layer_temp);
                    }
                }
            }
        }

        // Add outgassed material to atmospheric layers if any was generated
        if total_outgassed > 0.0 && atmo_end_index > 0 {
            self.add_atmospheric_material(cell, total_outgassed, atmo_end_index);
        }

        total_outgassed
    }
    
    /// Calculate outgassing rate based on material type
    fn calculate_outgassing_rate(&self, material_type: &MaterialCompositeType) -> f64 {
        match material_type {
            MaterialCompositeType::Silicate => 0.005,   // 0.5% outgassing rate
            MaterialCompositeType::Basaltic => 0.008,   // 0.8% outgassing rate (more volatile)
            MaterialCompositeType::Granitic => 0.012,   // 1.2% outgassing rate (most volatile)
            MaterialCompositeType::Metallic => 0.001,   // 0.1% outgassing rate (least volatile)
            _ => 0.003, // Default rate
        }
    }
    
    /// Add outgassed material to atmospheric layers with crystallization losses
    fn add_atmospheric_material(&mut self, cell: &mut crate::global_thermal::sim_cell::SimCell,
                                total_mass: f64, atmo_layer_count: usize) {
        if atmo_layer_count == 0 {
            return;
        }

        // Start with total outgassed mass and apply crystallization as it rises through layers
        let mut remaining_mass = total_mass;
        let mut total_crystallized = 0.0;

        // Process from bottom atmospheric layer (closest to surface) upward
        for i in (0..atmo_layer_count).rev() {
            if remaining_mass <= 0.0 {
                break;
            }

            // Calculate crystallization loss for this layer
            let crystallized_mass = remaining_mass * self.crystallization_rate;
            remaining_mass -= crystallized_mass;
            total_crystallized += crystallized_mass;

            // Add remaining mass to this layer
            let mass_to_add = remaining_mass / (i + 1) as f64; // Distribute remaining among this and layers above

            let (current, _) = &mut cell.layers_t[i];

            // Add mass by increasing volume (atmospheric material is low density)
            let air_density = current.energy_mass.density_kgm3();
            if air_density > 0.0 && mass_to_add > 0.0 {
                let additional_volume_m3 = mass_to_add / air_density;
                let additional_volume_km3 = additional_volume_m3 / 1e9;

                // Scale the layer to include new volume
                let current_volume = current.energy_mass.volume();
                if current_volume > 0.0 {
                    let new_volume = current_volume + additional_volume_km3;
                    let scale_factor = new_volume / current_volume;

                    current.energy_mass.scale(scale_factor);

                    if self.debug_output {
                        println!("Layer {} added: {:.2e} kg, crystallized: {:.2e} kg, volume: {:.2} -> {:.2} km³",
                                i, mass_to_add, crystallized_mass, current_volume, new_volume);
                    }
                }
            }
        }

        self.total_crystallized_mass += total_crystallized;

        if self.debug_output && total_crystallized > 0.0 {
            println!("Total crystallization loss: {:.2e} kg ({:.1}% of outgassed material)",
                    total_crystallized, (total_crystallized / total_mass) * 100.0);
        }
    }
    
    /// Redistribute atmospheric material with exponential density and volume decay
    fn redistribute_atmosphere(&mut self, cell: &mut crate::global_thermal::sim_cell::SimCell) -> f64 {
        let mut total_redistributed = 0.0;

        // Find atmospheric layers
        let mut atmo_layers: Vec<usize> = Vec::new();
        for (i, (current, _)) in cell.layers_t.iter().enumerate() {
            if current.is_atmospheric() {
                atmo_layers.push(i);
            } else {
                break; // Atmospheric layers are at the top
            }
        }

        if atmo_layers.len() < 2 {
            return 0.0; // Need at least 2 layers for redistribution
        }

        // Calculate total atmospheric mass
        let mut total_atmo_mass = 0.0;
        for &i in &atmo_layers {
            total_atmo_mass += cell.layers_t[i].0.energy_mass.mass_kg();
        }

        if total_atmo_mass <= 0.0 {
            return 0.0;
        }

        // Calculate target densities with exponential decay (bottom layer = highest density)
        // Each layer up has density_decay_factor (e.g., 0.12 = 12%) of the layer below
        let bottom_layer_idx = atmo_layers[atmo_layers.len()-1];
        let base_density = cell.layers_t[bottom_layer_idx].0.energy_mass.density_kgm3();

        let mut target_densities = Vec::new();
        let mut target_masses = Vec::new();
        let mut _total_target_mass = 0.0;

        for i in 0..atmo_layers.len() {
            // Higher layers (lower index) have exponentially lower density
            let layer_from_bottom = atmo_layers.len() - 1 - i;
            let target_density = base_density * self.density_decay_factor.powi(layer_from_bottom as i32);
            target_densities.push(target_density);

            // Calculate target mass based on layer volume and target density
            let layer_idx = atmo_layers[i];
            let layer_volume_m3 = cell.layers_t[layer_idx].0.energy_mass.volume() * 1e9; // km³ to m³
            let target_mass = target_density * layer_volume_m3;
            target_masses.push(target_mass);
            _total_target_mass += target_mass;
        }

        // Calculate base volume for volume decay before mutable borrows
        let base_volume = cell.layers_t[atmo_layers[atmo_layers.len()-1]].0.energy_mass.volume();

        // Redistribute mass according to exponential density profile
        for (idx, &layer_i) in atmo_layers.iter().enumerate() {
            let target_mass = target_masses[idx];
            let target_density = target_densities[idx];

            let (current, _) = &mut cell.layers_t[layer_i];
            let current_mass = current.energy_mass.mass_kg();
            let current_density = current.energy_mass.density_kgm3();

            if current_mass > 0.0 && target_mass > 0.0 {
                let mass_ratio = target_mass / current_mass;
                current.energy_mass.scale(mass_ratio);
                total_redistributed += (target_mass - current_mass).abs();

                if self.debug_output {
                    println!("Layer {} density: {:.3} -> {:.3} kg/m³ (mass: {:.2e} -> {:.2e} kg)",
                            layer_i, current_density, target_density, current_mass, target_mass);
                }

                // Remove excess volume that cannot be transported upward
                // Use exponential volume decay as well
                let layer_from_bottom = atmo_layers.len() - 1 - idx;
                let target_volume = base_volume * self.volume_decay_factor.powi(layer_from_bottom as i32);

                let new_volume = current.energy_mass.volume();
                if new_volume > target_volume * 1.1 { // Allow 10% tolerance
                    let volume_ratio = target_volume / new_volume;
                    current.energy_mass.scale(volume_ratio);

                    if self.debug_output {
                        println!("Removed excess volume from layer {}: {:.2} -> {:.2} km³",
                                layer_i, new_volume, target_volume);
                    }
                }
            }
        }

        total_redistributed
    }

    /// Globally redistribute atmospheric material equally across all H3 cells
    fn globally_redistribute_atmosphere(&mut self, sim: &mut Simulation) -> f64 {
        let mut total_redistributed = 0.0;

        if sim.cells.is_empty() {
            return 0.0;
        }

        // Step 1: Calculate total atmospheric mass and volumes across all cells
        let mut global_atmospheric_masses: Vec<f64> = Vec::new(); // Mass per atmospheric layer globally
        let mut cell_atmospheric_layers: Vec<Vec<usize>> = Vec::new(); // Atmospheric layer indices per cell
        let mut max_atmo_layers = 0;

        // Find atmospheric layers in each cell and calculate totals
        for cell in sim.cells.values() {
            let mut atmo_layers: Vec<usize> = Vec::new();
            for (i, (current, _)) in cell.layers_t.iter().enumerate() {
                if current.is_atmospheric() {
                    atmo_layers.push(i);
                } else {
                    break; // Atmospheric layers are at the top
                }
            }

            max_atmo_layers = max_atmo_layers.max(atmo_layers.len());
            cell_atmospheric_layers.push(atmo_layers);
        }

        // Initialize global mass totals for each atmospheric layer
        global_atmospheric_masses.resize(max_atmo_layers, 0.0);

        // Sum up total mass in each atmospheric layer across all cells
        for cell in sim.cells.values() {
            for (i, (current, _)) in cell.layers_t.iter().enumerate() {
                if current.is_atmospheric() && i < max_atmo_layers {
                    global_atmospheric_masses[i] += current.energy_mass.mass_kg();
                }
            }
        }

        // Step 2: Calculate target masses per cell with exponential density decay
        let cell_count = sim.cells.len() as f64;
        let mut target_masses_per_layer: Vec<f64> = Vec::new();

        for layer_idx in 0..max_atmo_layers {
            let total_mass_this_layer = global_atmospheric_masses[layer_idx];
            let target_mass_per_cell = total_mass_this_layer / cell_count;
            target_masses_per_layer.push(target_mass_per_cell);
        }

        // Step 3: Apply exponential density decay within each cell's atmospheric layers
        let mut cell_index = 0;
        for cell in sim.cells.values_mut() {
            if cell_index >= cell_atmospheric_layers.len() {
                break;
            }

            let atmo_layers = &cell_atmospheric_layers[cell_index];
            let atmo_layer_count = atmo_layers.len();
            cell_index += 1;

            if atmo_layer_count == 0 {
                continue;
            }

            // Calculate total target mass for this cell
            let mut total_target_mass_this_cell = 0.0;
            for layer_idx in 0..atmo_layer_count.min(max_atmo_layers) {
                total_target_mass_this_cell += target_masses_per_layer[layer_idx];
            }

            if total_target_mass_this_cell <= 0.0 {
                continue;
            }

            // Apply exponential density decay within this cell
            let mut density_weights: Vec<f64> = Vec::new();
            let mut total_density_weight = 0.0;

            for i in 0..atmo_layer_count {
                // Higher layers (lower index) have exponentially lower density
                let layer_from_bottom = atmo_layer_count - 1 - i;
                let density_weight = self.density_decay_factor.powi(layer_from_bottom as i32);
                density_weights.push(density_weight);
                total_density_weight += density_weight;
            }

            // Redistribute mass according to density profile
            for (idx, &layer_i) in atmo_layers.iter().enumerate() {
                if idx >= density_weights.len() {
                    break;
                }

                let density_weight = density_weights[idx];
                let target_mass = total_target_mass_this_cell * (density_weight / total_density_weight);

                // Get base volume before borrowing mutably
                let base_volume = if atmo_layer_count > 0 {
                    cell.layers_t[atmo_layers[atmo_layer_count-1]].0.energy_mass.volume()
                } else {
                    cell.layers_t[layer_i].0.energy_mass.volume()
                };

                let (current, _) = &mut cell.layers_t[layer_i];
                let current_mass = current.energy_mass.mass_kg();

                if current_mass > 0.0 && target_mass > 0.0 {
                    let mass_ratio = target_mass / current_mass;
                    current.energy_mass.scale(mass_ratio);
                    total_redistributed += (target_mass - current_mass).abs();

                    // Apply volume constraints
                    let layer_from_bottom = atmo_layer_count - 1 - idx;
                    let target_volume = base_volume * self.volume_decay_factor.powi(layer_from_bottom as i32);

                    let new_volume = current.energy_mass.volume();
                    if new_volume > target_volume * 1.1 { // Allow 10% tolerance
                        let volume_ratio = target_volume / new_volume;
                        current.energy_mass.scale(volume_ratio);

                        if self.debug_output && self.step_count % 500 == 0 {
                            println!("Cell layer {} volume constrained: {:.2} -> {:.2} km³",
                                    layer_i, new_volume, target_volume);
                        }
                    }
                }
            }
        }

        if self.debug_output && self.step_count % 500 == 0 {
            println!("Global atmospheric redistribution: {:.2e} kg across {} cells",
                    total_redistributed, sim.cells.len());
            for (i, &mass) in global_atmospheric_masses.iter().enumerate() {
                if mass > 0.0 {
                    println!("  Layer {}: {:.2e} kg total, {:.2e} kg per cell",
                            i, mass, mass / cell_count);
                }
            }
        }

        total_redistributed
    }
}

impl SimOp for AtmosphericGenerationOp {
    fn name(&self) -> &str {
        "AtmosphericGeneration"
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn init_sim(&mut self, _sim: &mut Simulation) {
        if self.debug_output {
            println!("AtmosphericGenerationOp initialized (EVENT-DRIVEN):");
            println!("  - Detects melting events (solid -> liquid/gas transitions)");
            println!("  - Generates atmosphere from melting events, not static liquid state");
            println!("  - Outgassing rate: {:.3}% of melted lithosphere mass", self.outgassing_rate_multiplier * 100.0);
            println!("  - Depth attenuation: {:.3} ({}% reduction per layer depth)",
                    self.depth_attenuation_factor, ((1.0 - self.depth_attenuation_factor) * 100.0) as i32);
            println!("  - Crystallization rate: {:.3} ({}% crystallizes per atmospheric layer)",
                    self.crystallization_rate, (self.crystallization_rate * 100.0) as i32);
            println!("  - Atmospheric density decay: {:.3} ({}% reduction per layer)",
                    self.density_decay_factor, ((1.0 - self.density_decay_factor) * 100.0) as i32);
            println!("  - Volume decay: {:.3} per layer", self.volume_decay_factor);
        }
    }

    fn update_sim(&mut self, sim: &mut Simulation) {
        if !self.apply_during_simulation {
            return;
        }

        self.step_count += 1;
        self.skip_counter += 1;

        // Performance optimization: Skip processing every N steps
        if self.skip_counter < self.skip_interval {
            return;
        }
        self.skip_counter = 0;

        // Step 1: Detect melting events by comparing current vs previous layer phases
        self.detect_melting_events_optimized(sim);

        // Step 2: Generate compounds from melting events and add to global atmosphere
        let step_outgassed = self.generate_atmosphere_from_melting_events_optimized(sim);

        // Step 3: Redistribute global atmosphere to all cells with exponential distribution
        let step_redistributed = self.redistribute_global_atmosphere_to_cells_optimized(sim);

        self.total_outgassed_mass += step_outgassed;
        self.total_redistributed_volume += step_redistributed;

        // Only show summary every 10 steps and only for significant events
        if self.debug_output && self.step_count % 10 == 0 && step_outgassed > 1e19 {
            let total_global_mass: f64 = self.global_atmosphere_compounds.values().sum();
            println!("Step {}: {:.2e} kg outgassed, {:.2e} kg global atmosphere",
                     self.step_count, step_outgassed, total_global_mass);
        }
    }

    fn after_sim(&mut self, _sim: &mut Simulation) {
        if self.debug_output {
            println!("AtmosphericGeneration complete:");
            println!("  - Total outgassed: {:.2e} kg", self.total_outgassed_mass);
            println!("  - Total crystallized: {:.2e} kg ({:.1}% loss)",
                    self.total_crystallized_mass,
                    if self.total_outgassed_mass > 0.0 {
                        (self.total_crystallized_mass / self.total_outgassed_mass) * 100.0
                    } else { 0.0 });
            println!("  - Net atmospheric addition: {:.2e} kg",
                    self.total_outgassed_mass - self.total_crystallized_mass);
            println!("  - Total redistributed: {:.2e} kg", self.total_redistributed_volume);
        }
    }

}
