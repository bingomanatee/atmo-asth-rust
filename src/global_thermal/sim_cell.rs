use std::f64::consts::PI;
/// Global H3 cell with unified thermal layers
///
/// Each cell contains a stack of thermal layers that can naturally transition
/// between atmospheric, liquid, and solid phases based on temperature and pressure.

use crate::global_thermal::thermal_layer::ThermalLayer;
use crate::global_thermal::heat_plume::CellPlumeCollection;
use crate::energy_mass_composite::{MaterialCompositeType, MaterialPhase};
use crate::planet::Planet;
use h3o::CellIndex as H3Index;
use std::sync::Arc;

/// Layer configuration for global H3 cell initialization
#[derive(Debug, Clone)]
pub struct LayerConfig {
    pub cell_type: MaterialCompositeType,
    pub cell_count: usize,
    pub height_km: f64,
    pub is_foundry: bool,
}

/// Parameters for creating a new GlobalH3Cell with custom schedule
#[derive(Debug, Clone)]
pub struct GlobalH3CellConfig {
    pub h3_index: H3Index,
    pub planet: Arc<Planet>,
    pub layer_schedule: Vec<LayerConfig>,
}

impl GlobalH3CellConfig {
    /// Create a new config with standard Earth-like schedule
    pub fn new_earth_like(h3_index: H3Index, planet: Arc<Planet>) -> Self {
        Self {
            h3_index,
            planet,
            layer_schedule: SimCell::create_earth_like_schedule(),
        }
    }

    /// Create a new config with thick atmosphere schedule (Venus-like)
    pub fn new_thick_atmosphere(h3_index: H3Index, planet: Arc<Planet>) -> Self {
        Self {
            h3_index,
            planet,
            layer_schedule: SimCell::create_thick_atmosphere_schedule(),
        }
    }

    /// Create a new config with thin atmosphere schedule (Mars-like)
    pub fn new_thin_atmosphere(h3_index: H3Index, planet: Arc<Planet>) -> Self {
        Self {
            h3_index,
            planet,
            layer_schedule: SimCell::create_thin_atmosphere_schedule(),
        }
    }

    /// Create a new config with custom schedule
    pub fn new_custom(
        h3_index: H3Index,
        planet: Arc<Planet>,
        layer_schedule: Vec<LayerConfig>
    ) -> Self {
        Self {
            h3_index,
            planet,
            layer_schedule,
        }
    }
}

/// Global H3 hexagonal cell containing unified thermal layers
#[derive(Debug, Clone)]
pub struct SimCell {
    /// H3 hexagonal index for this cell
    pub h3_index: H3Index,

    /// Shared planet reference with gravity, mass, radius, etc.
    pub planet: Arc<Planet>,

    /// Stack of thermal layers with (current, next) state tuples
    pub layers_t: Vec<(ThermalLayer, ThermalLayer)>,
    
    /// Collection of heat plumes spanning across layers
    pub plumes: CellPlumeCollection,
}

impl SimCell {
    pub fn lateral_area(&self) -> f64 {
        self.surface_area_km2()  * self.height_km()
    }

    pub fn height_km(&self) -> f64 {
        match self.layers_t.first() {
            Some((first, _)) => first.height_km,
            None => 0.0,
        }
    }

    /// Create a new global H3 cell with custom configuration
    pub fn new_with_config(config: GlobalH3CellConfig) -> Self {
        Self::new_with_schedule(
            config.h3_index,
            config.planet,
            &config.layer_schedule
        )
    }

    pub fn cell_radius(&self) -> f64 {
        match self.layers_t.first() {
            Some((layer, _)) => (layer.surface_area_km2 / PI).sqrt(),
            None => 0.0,
        }
    }

    /// Create a new global H3 cell with custom layer configuration from schedule
    pub fn new_with_schedule(
        h3_index: H3Index,
        planet: Arc<Planet>,
        layer_schedule: &[LayerConfig]
    ) -> Self {
        let mut layers = Vec::new();

        // Calculate surface area from H3 cell
        let surface_area_km2 = crate::h3_utils::H3Utils::cell_area(planet.resolution, planet.radius_km);

        // Calculate the total atmospheric depth (only Air layers)
        let atmospheric_depth = layer_schedule.iter()
            .filter(|config| config.cell_type == MaterialCompositeType::Air)
            .map(|config| config.cell_count as f64 * config.height_km)
            .sum::<f64>();

        let mut current_depth = -atmospheric_depth; // Start from top of atmosphere
        let mut layer_index = 0;
        let mut found_surface_layer = false;

        // First pass: collect all non-atmospheric layer indices to identify foundry layers
        let mut all_layers_info = Vec::new();
        let mut temp_current_depth = current_depth;
        let mut temp_layer_index = layer_index;
        
        for config in layer_schedule {
            for _ in 0..config.cell_count {
                let is_surface_layer = !found_surface_layer && config.cell_type != MaterialCompositeType::Air;
                let is_non_atmospheric = config.cell_type != MaterialCompositeType::Air;
                all_layers_info.push((temp_layer_index, is_surface_layer, is_non_atmospheric, config.clone()));
                temp_current_depth += config.height_km;
                temp_layer_index += 1;
            }
        }

        // Identify foundry layers: layers explicitly marked as foundry in config
        let foundry_layer_indices: std::collections::HashSet<usize> = all_layers_info.iter()
            .filter(|(_, _, _, config)| config.is_foundry)
            .map(|(index, _, _, _)| *index)
            .collect();

        // Second pass: create layers according to schedule with (current, next) tuples
        for config in layer_schedule {
            for _ in 0..config.cell_count {
                let is_surface_layer = !found_surface_layer && config.cell_type != MaterialCompositeType::Air;
                if is_surface_layer {
                    found_surface_layer = true;
                }
                let is_foundry = foundry_layer_indices.contains(&layer_index);
                
                let layer = if config.cell_type == MaterialCompositeType::Air {
                    // Atmospheric layers start with zero density/volume - will be filled by outgassing
                    ThermalLayer::new_atmospheric(
                        current_depth,
                        config.height_km,
                        surface_area_km2,
                        layer_index,
                    )
                } else {
                    // Lithosphere/asthenosphere layers start with material density - will be compacted by pressure
                    ThermalLayer::new_solid(
                        current_depth,
                        config.height_km,
                        surface_area_km2,
                        config.cell_type,
                        layer_index,
                        is_surface_layer,
                        is_foundry,
                    )
                };

                // Create (current, next) tuple - next starts as clone of current
                layers.push((layer.clone(), layer));
                current_depth += config.height_km;
                layer_index += 1;
            }
        }

        Self {
            h3_index,
            planet,
            layers_t: layers,
            plumes: CellPlumeCollection::new(h3_index),
        }
    }

    /// Create a new global H3 cell with standard layer configuration
    /// 4×20km atmosphere + 10×4km lithosphere + 10×8km asthenosphere
    pub fn new(h3_index: H3Index, planet: Arc<Planet>) -> Self {
        let standard_schedule = vec![
            LayerConfig {
                cell_type: MaterialCompositeType::Air,
                cell_count: 5,
                height_km: 20.0, // 100 total atmosphere (4×20km layers)
                is_foundry: false,
            },
            LayerConfig {
                cell_type: MaterialCompositeType::Silicate,
                cell_count: 6,
                height_km: 25.0, // 150 total lithosphere
                is_foundry: false,
            },
            LayerConfig {
                cell_type: MaterialCompositeType::Silicate,
                cell_count: 6,
                height_km: 40.0, // 240 total lower lithosphere/asthenosphere
                is_foundry: false,
            },
        ];

        Self::new_with_schedule(h3_index, planet, &standard_schedule)
    }

    /// Get surface area of this cell in km²
    pub fn surface_area_km2(&self) -> f64 {
        crate::h3_utils::H3Utils::cell_area(self.planet.resolution, self.planet.radius_km)
    }

    /// Get total number of layers
    pub fn layer_count(&self) -> usize {
        self.layers_t.len()
    }

    /// Get current layer by index
    pub fn layer(&self, index: usize) -> Option<&ThermalLayer> {
        self.layers_t.get(index).map(|(current, _)| current)
    }

    /// Get mutable current layer by index
    pub fn layer_mut(&mut self, index: usize) -> Option<&mut ThermalLayer> {
        self.layers_t.get_mut(index).map(|(current, _)| current)
    }

    /// Get next layer by index
    pub fn layer_next(&self, index: usize) -> Option<&ThermalLayer> {
        self.layers_t.get(index).map(|(_, next)| next)
    }

    /// Get mutable next layer by index
    pub fn layer_next_mut(&mut self, index: usize) -> Option<&mut ThermalLayer> {
        self.layers_t.get_mut(index).map(|(_, next)| next)
    }

    /// Get layer tuple (current, next) by index
    pub fn layer_tuple(&self, index: usize) -> Option<&(ThermalLayer, ThermalLayer)> {
        self.layers_t.get(index)
    }

    /// Get mutable layer tuple (current, next) by index
    pub fn layer_tuple_mut(&mut self, index: usize) -> Option<&mut (ThermalLayer, ThermalLayer)> {
        self.layers_t.get_mut(index)
    }

    /// Get current layers in depth range (inclusive)
    pub fn layers_in_depth_range(&self, min_depth_km: f64, max_depth_km: f64) -> Vec<&ThermalLayer> {
        self.layers_t.iter()
            .map(|(current, _)| current)
            .filter(|layer| {
                let layer_center = layer.start_depth_km + (layer.height_km / 2.0);
                layer_center >= min_depth_km && layer_center <= max_depth_km
            })
            .collect()
    }

    /// Commit next state to current state for all layers (temporal evolution step)
    pub fn commit_next_state(&mut self) {
        for (current, next) in &mut self.layers_t {
            *current = next.clone();
        }
    }

    /// Reset next state to current state for all layers
    pub fn reset_next_state(&mut self) {
        for (current, next) in &mut self.layers_t {
            *next = current.clone();
        }
    }



    /// Create a custom layer schedule for different planetary configurations
    pub fn create_earth_like_schedule() -> Vec<LayerConfig> {
        vec![
            LayerConfig {
                cell_type: MaterialCompositeType::Air,
                cell_count: 4,
                height_km: 20.0, // 80km atmosphere (4×20km layers)
                is_foundry: false,
            },
            LayerConfig {
                cell_type: MaterialCompositeType::Silicate,
                cell_count: 5,
                height_km: 2.0, // 10km lithosphere (5×2km layers) - realistic oceanic/continental crust
                is_foundry: false,
            },
            LayerConfig {
                cell_type: MaterialCompositeType::Silicate,
                cell_count: 14,
                height_km: 20.0, // 280km asthenosphere (14×20km layers) - deep mantle to 290km
                is_foundry: false,
            },
        ]
    }

    /// Create a thick atmosphere schedule for Venus-like planets
    pub fn create_thick_atmosphere_schedule() -> Vec<LayerConfig> {
        vec![
            LayerConfig {
                cell_type: MaterialCompositeType::Air,
                cell_count: 4,
                height_km: 20.0, // 80km thick atmosphere (4×20km layers)
                is_foundry: false,
            },
            LayerConfig {
                cell_type: MaterialCompositeType::Silicate,
                cell_count: 8,
                height_km: 5.0, // 40km lithosphere
                is_foundry: false,
            },
            LayerConfig {
                cell_type: MaterialCompositeType::Silicate,
                cell_count: 8,
                height_km: 10.0, // 80km asthenosphere
                is_foundry: false,
            },
        ]
    }

    /// Create a thin atmosphere schedule for Mars-like planets
    pub fn create_thin_atmosphere_schedule() -> Vec<LayerConfig> {
        vec![
            LayerConfig {
                cell_type: MaterialCompositeType::Air,
                cell_count: 4,
                height_km: 10.0, // 40km thin atmosphere (4×10km layers)
                is_foundry: false,
            },
            LayerConfig {
                cell_type: MaterialCompositeType::Silicate,
                cell_count: 10,
                height_km: 4.0, // 40km lithosphere
                is_foundry: false,
            },
            LayerConfig {
                cell_type: MaterialCompositeType::Silicate,
                cell_count: 10,
                height_km: 8.0, // 80km asthenosphere
                is_foundry: false,
            },
        ]
    }
    
    /// Get current atmospheric layers (above surface)
    pub fn atmospheric_layers(&self) -> impl Iterator<Item = &ThermalLayer> {
        self.layers_t.iter().map(|(current, _)| current).filter(|layer| layer.is_atmospheric())
    }

    /// Get mutable current atmospheric layers
    pub fn atmospheric_layers_mut(&mut self) -> impl Iterator<Item = &mut ThermalLayer> {
        self.layers_t.iter_mut().map(|(current, _)| current).filter(|layer| layer.is_atmospheric())
    }

    /// Get current surface layers (crossing surface boundary)
    pub fn surface_layers(&self) -> impl Iterator<Item = &ThermalLayer> {
        self.layers_t.iter().map(|(current, _)| current).filter(|layer| layer.is_surface())
    }

    /// Calculate pressure for each layer based on overlying mass (using current state)
    pub fn calculate_layer_pressures(&self) -> Vec<f64> {
        let mut pressures = Vec::with_capacity(self.layers_t.len());
        let gravity = self.planet.gravity_m_s2; // Use planet-specific gravity
        let surface_area_m2 = self.surface_area_km2() * 1e6; // km² to m²

        let mut cumulative_mass = 0.0;

        for (current, _) in &self.layers_t {
            // Pressure at center of layer
            let layer_center_mass = current.mass_kg() / 2.0;
            let pressure_pa = (cumulative_mass + layer_center_mass) * gravity / surface_area_m2;
            pressures.push(pressure_pa);

            cumulative_mass += current.mass_kg();
        }

        pressures
    }
    
    /// Process space radiation for thin atmospheric layers (modifies current state)
    pub fn process_space_radiation(&mut self, years: f64) -> f64 {
        const STEFAN_BOLTZMANN: f64 = 5.670374419e-8; // W⋅m⁻²⋅K⁻⁴
        const SPACE_TEMPERATURE: f64 = 2.7; // Cosmic background radiation
        const EMISSIVITY: f64 = 0.95; // Atmospheric emissivity

        let mut total_radiated = 0.0;
        let surface_area_m2 = self.surface_area_km2() * 1e6; // km² to m²

        for (current, _) in &mut self.layers_t {
            if current.is_thin_atmospheric() {
                let temp_k = current.temperature_k();

                // Stefan-Boltzmann radiation to space
                let radiated_power_w_m2 = EMISSIVITY * STEFAN_BOLTZMANN *
                                         (temp_k.powi(4) - SPACE_TEMPERATURE.powi(4));

                let energy_loss_j = radiated_power_w_m2 * surface_area_m2 * years * 365.25 * 24.0 * 3600.0;

                current.remove_energy(energy_loss_j);
                total_radiated += energy_loss_j;
            }
        }

        total_radiated
    }

    /// Process phase transitions for all layers (modifies current state)
    pub fn process_phase_transitions(&mut self) -> Vec<(usize, MaterialPhase, MaterialPhase)> {
        let pressures = self.calculate_layer_pressures();
        let mut transitions = Vec::new();

        for (i, (current, _)) in self.layers_t.iter_mut().enumerate() {
            let old_phase = current.phase();

            // Update phase based on temperature and pressure
            current.update_phase(pressures[i]);

            let new_phase = current.phase();
            if old_phase != new_phase {
                transitions.push((i, old_phase, new_phase));
            }
        }

        transitions
    }
    
    /// Get total atmospheric mass (gas phase layers above surface)
    pub fn total_atmospheric_mass(&self) -> f64 {
        self.atmospheric_layers()
            .filter(|layer| matches!(layer.phase(), MaterialPhase::Gas))
            .map(|layer| layer.mass_kg())
            .sum()
    }

    /// Get average surface temperature
    pub fn average_surface_temperature(&self) -> f64 {
        let surface_temps: Vec<f64> = self.surface_layers()
            .map(|layer| layer.temperature_k())
            .collect();

        if surface_temps.is_empty() {
            280.0 // Default surface temperature
        } else {
            surface_temps.iter().sum::<f64>() / surface_temps.len() as f64
        }
    }


    /// Get density profile for all current layers
    pub fn get_density_profile(&self) -> Vec<f64> {
        self.layers_t.iter()
            .map(|(current, _)| current.current_density_kg_m3())
            .collect()
    }

    /// Get compaction status for all current layers
    pub fn get_compaction_status(&self) -> Vec<bool> {
        self.layers_t.iter()
            .map(|(current, _)| current.is_compacted())
            .collect()
    }
}

impl std::fmt::Display for SimCell {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GlobalH3Cell[{:?}, {}, {:.1}km², {} layers, {:.0}K surface, {:.2}m/s² gravity]",
               self.h3_index,
               self.planet.name,
               self.surface_area_km2(),
               self.layers_t.len(),
               self.average_surface_temperature(),
               self.planet.gravity_m_s2)
    }
}
