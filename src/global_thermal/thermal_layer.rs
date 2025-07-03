/// Unified thermal layer that can represent atmospheric, lithospheric, or asthenospheric material
/// based on its physical properties and phase state rather than artificial categorization.

use crate::energy_mass_composite::{StandardEnergyMassComposite, MaterialPhase, MaterialCompositeType, EnergyMassParams, EnergyMassComposite};

/// A unified thermal layer with flexible depth positioning and natural phase transitions
#[derive(Debug, Clone)]
pub struct ThermalLayer {
    /// Depth from surface in km (negative = atmosphere, positive = subsurface)
    pub start_depth_km: f64,
    
    /// Layer thickness in km
    pub height_km: f64,
    
    /// Unified energy and mass system that handles all material phases
    pub energy_mass: StandardEnergyMassComposite,

    /// Surface area of this layer in km²
    pub surface_area_km2: f64,

    /// Index of this layer (0 = top, higher = deeper)
    pub layer_index: usize,

    /// Whether this is the surface layer (first non-atmospheric layer)
    pub is_surface_layer: bool,
}

impl ThermalLayer {

    /// Create a new atmospheric layer with reasonable volume and gaseous temperature
    pub fn new_atmospheric(start_depth_km: f64, height_km: f64, surface_area_km2: f64, layer_index: usize) -> Self {
        // Use volume = 5 km³ as suggested to avoid zero density issues
        let volume_km3 = 5.0;

        // Calculate reasonable atmospheric temperature based on altitude
        // Use standard atmospheric lapse rate: surface temp - 6.5K per km altitude
        let surface_temp_k = 288.0; // Standard surface temperature (15°C)
        let lapse_rate_k_per_km = 6.5;
        let altitude_km = (-start_depth_km + height_km / 2.0).max(0.0); // Center altitude of layer
        let atmospheric_temp_k = (surface_temp_k - altitude_km * lapse_rate_k_per_km).max(200.0); // Minimum 200K

        let mut energy_mass = StandardEnergyMassComposite::new_with_params(
            EnergyMassParams {
                material_type: MaterialCompositeType::Air,
                initial_phase: MaterialPhase::Gas,
                energy_joules: 0.0, // Will be set by temperature
                volume_km3,
                height_km,
                pressure_gpa: 0.0,
            }
        );

        // Set the atmospheric temperature
        energy_mass.set_kelvin(atmospheric_temp_k);

        Self {
            start_depth_km,
            height_km,
            energy_mass,
            surface_area_km2,
            layer_index,
            is_surface_layer: false, // Atmospheric layers are never surface layers
        }
    }

    /// Create a new solid layer with realistic material density (will be compacted by pressure)
    pub fn new_solid(start_depth_km: f64, height_km: f64, surface_area_km2: f64, material_type: MaterialCompositeType, layer_index: usize, is_surface_layer: bool) -> Self {
        // Calculate full volume for this layer
        let _volume_m3 = surface_area_km2 * height_km * 1e9; // km³ to m³

        // Create with standard material density
        let energy_mass = StandardEnergyMassComposite::new_with_params(EnergyMassParams {
            material_type,
            initial_phase: MaterialPhase::Solid,
            energy_joules: 0.0,
            volume_km3: surface_area_km2 * height_km, 
            height_km,
            pressure_gpa: 0.0,
        });

        Self {
            start_depth_km,
            height_km,
            energy_mass,
            surface_area_km2,
            layer_index,
            is_surface_layer,
        }
    }

    /// Create a new thermal layer (legacy method)
    pub fn new(start_depth_km: f64, height_km: f64, surface_area_km2: f64, material_type: MaterialCompositeType) -> Self {
        if material_type == MaterialCompositeType::Air {
            Self::new_atmospheric(start_depth_km, height_km, surface_area_km2, 0)
        } else {
            Self::new_solid(start_depth_km, height_km, surface_area_km2, material_type, 0, false)
        }
    }
    
    /// End depth of this layer
    pub fn end_depth_km(&self) -> f64 {
        self.start_depth_km + self.height_km
    }
    
    /// Check if this layer is in the atmosphere (above surface)
    pub fn is_atmospheric(&self) -> bool {
        self.energy_mass.material_composite_type() == MaterialCompositeType::Air
    }
    
    /// Check if this layer crosses the surface boundary
    pub fn is_surface(&self) -> bool {
        self.start_depth_km <= 0.0 && self.end_depth_km() >= 0.0
    }
    
    /// Check if this layer is thin enough to radiate directly to space
    pub fn is_thin_atmospheric(&self) -> bool {
        self.is_atmospheric() && self.height_km < 2.0 // < 2km = thin atmospheric layer
    }
    
    /// Check if this layer is in the deep subsurface
    pub fn is_deep_subsurface(&self) -> bool {
        self.start_depth_km > 50.0 // > 50km depth
    }
    
    /// Get the center depth of this layer
    pub fn center_depth_km(&self) -> f64 {
        self.start_depth_km + (self.height_km / 2.0)
    }
    
    /// Calculate pressure at the center of this layer based on overlying mass
    pub fn calculate_pressure_pa(&self, overlying_layers: &[ThermalLayer]) -> f64 {
        const GRAVITY: f64 = 9.81; // m/s²
        
        let mut total_mass_kg = 0.0;
        
        // Sum mass of all layers above this one
        for layer in overlying_layers {
            if layer.end_depth_km() <= self.start_depth_km {
                total_mass_kg += layer.energy_mass.mass_kg();
            }
        }
        
        // Pressure = (mass * gravity) / surface_area
        let surface_area_m2 = self.surface_area_km2 * 1e6; // km² to m²
        (total_mass_kg * GRAVITY) / surface_area_m2
    }
    
    /// Get current material phase
    pub fn phase(&self) -> MaterialPhase {
        self.energy_mass.phase()
    }
    
    /// Get current temperature
    pub fn temperature_k(&self) -> f64 {
        self.energy_mass.temperature()
    }
    
    /// Get layer mass
    pub fn mass_kg(&self) -> f64 {
        self.energy_mass.mass_kg()
    }

    /// Get the volume of this layer in km³
    pub fn volume_km3(&self) -> f64 {
        self.surface_area_km2 * self.height_km
    }
    
    /// Get layer energy
    pub fn energy_j(&self) -> f64 {
        self.energy_mass.energy()
    }
    
    /// Add energy to this layer
    pub fn add_energy(&mut self, energy_j: f64) {
        self.energy_mass.add_energy(energy_j);
    }
    
    /// Remove energy from this layer
    pub fn remove_energy(&mut self, energy_j: f64) {
        self.energy_mass.remove_energy(energy_j);
    }
    
    /// Calculate thermal conductivity based on current phase and material
    pub fn thermal_conductivity(&self) -> f64 {
        self.energy_mass.thermal_conductivity()
    }
    
    /// Calculate density based on current phase and temperature
    pub fn density_kg_m3(&self) -> f64 {
        let volume_m3 = self.surface_area_km2 * self.height_km * 1e9; // km³ to m³
        self.mass_kg() / volume_m3
    }
    
    /// Update phase based on current temperature and pressure
    pub fn update_phase(&mut self, pressure_pa: f64) {
        // Convert Pa to GPa and update phase considering pressure effects
        let pressure_gpa = pressure_pa / 1e9;
        self.energy_mass.set_pressure_gpa(pressure_gpa);
    }
    
    /// Check if this layer has undergone a phase transition
    pub fn check_phase_transition(&mut self, pressure_pa: f64) -> Option<(MaterialPhase, MaterialPhase)> {
        let old_phase = self.phase();
        self.update_phase(pressure_pa);
        let new_phase = self.phase();
        
        if old_phase != new_phase {
            Some((old_phase, new_phase))
        } else {
            None
        }
    }
    
    /// Calculate the effective height for diffusion (pressure-dependent)
    pub fn effective_height_for_diffusion(&self, pressure_pa: f64) -> f64 {
        // Higher pressure reduces effective height for diffusion
        let pressure_factor = 1.0 / (1.0 + pressure_pa / 1e8); // Normalize by ~1 GPa
        self.height_km * pressure_factor.max(0.1) // Minimum 10% of original height
    }

    /// Compact this layer based on overlying pressure (for solid layers)
    pub fn apply_pressure_compaction(&mut self, pressure_pa: f64) {
        // Only compact solid materials (not atmospheric)
        if self.is_atmospheric() {
            return;
        }

        // Pressure compaction formula: density increases with pressure
        // Using bulk modulus approximation: ΔV/V = -ΔP/K
        // where K is bulk modulus (~100 GPa for silicates)

        const BULK_MODULUS_PA: f64 = 1e11; // 100 GPa for silicate materials
        const REFERENCE_PRESSURE_PA: f64 = 1e5; // 1 atm reference pressure

        // Calculate compression ratio
        let pressure_increase = pressure_pa - REFERENCE_PRESSURE_PA;
        let compression_ratio = 1.0 + (pressure_increase / BULK_MODULUS_PA).max(0.0);

        // Increase density (decrease volume) due to compression
        // Mass stays the same, but volume decreases
        let _original_mass = self.mass_kg();
        let original_volume_m3 = self.surface_area_km2 * self.height_km * 1e9;
        let compressed_volume_m3 = original_volume_m3 / compression_ratio;

        // Update the energy mass with new compressed volume
        let compressed_volume_km3 = compressed_volume_m3 / 1e9; // Convert back to km³
        self.energy_mass = StandardEnergyMassComposite::new_with_params(EnergyMassParams {
            material_type: self.energy_mass.material_composite_type(),
            initial_phase: self.energy_mass.phase(),
            energy_joules: self.energy_mass.energy_joules,
            volume_km3: compressed_volume_km3,
            height_km: self.height_km,
            pressure_gpa: pressure_pa / 1e9, // Convert Pa to GPa
        });

        // Preserve temperature during compression (adiabatic compression would increase temp slightly)
        let temp = self.temperature_k();
        self.energy_mass.set_temperature(temp * compression_ratio.powf(0.1)); // Slight temperature increase
    }

    /// Get current density of this layer
    pub fn current_density_kg_m3(&self) -> f64 {
        let current_volume_m3 = self.surface_area_km2 * self.height_km * 1e9;
        self.mass_kg() / current_volume_m3
    }

    /// Check if this layer is significantly compacted
    pub fn is_compacted(&self) -> bool {
        // Compare current density to standard material density
        let standard_density = self.energy_mass.material_composite_profile().density_kg_m3;
        let current_density = self.current_density_kg_m3();
        current_density > standard_density * 1.1 // 10% increase indicates compaction
    }
}

impl std::fmt::Display for ThermalLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ThermalLayer[{:.1}-{:.1}km, {:.0}K, {:?}, {:.2e}kg]", 
               self.start_depth_km, 
               self.end_depth_km(), 
               self.temperature_k(),
               self.phase(),
               self.mass_kg())
    }
}
