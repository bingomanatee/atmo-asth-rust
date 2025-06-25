use serde::{Deserialize, Serialize};
use crate::energy_mass::{EnergyMass, StandardEnergyMass};
use crate::material::MaterialType;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AsthCellLithosphere {
    pub height_km: f64,
    energy_mass: StandardEnergyMass,
}

impl AsthCellLithosphere {
    /// Create a new lithosphere with specified material, height, and volume
    /// Temperature defaults to 0K (will be set by thermal processes)
    pub fn new(height_km: f64, material: MaterialType, volume_km3: f64) -> AsthCellLithosphere {
        AsthCellLithosphere {
            height_km,
            energy_mass: StandardEnergyMass::new_with_material(material, 0.0, volume_km3),
        }
    }

    /// Create a new lithosphere with specified material, height, volume, and temperature
    pub fn new_with_temp(height_km: f64, material: MaterialType, volume_km3: f64, temperature_k: f64) -> AsthCellLithosphere {
        AsthCellLithosphere {
            height_km,
            energy_mass: StandardEnergyMass::new_with_material(material, temperature_k, volume_km3),
        }
    }

    /// Get the material type
    pub fn material(&self) -> MaterialType {
        self.energy_mass.material_type()
    }

    /// Get the volume in km³
    pub fn volume_km3(&self) -> f64 {
        self.energy_mass.volume()
    }

    /// Set the volume in km³ (maintains temperature)
    pub fn set_volume_km3(&mut self, volume_km3: f64) {
        let current_temp = self.energy_mass.temperature();
        self.energy_mass.remove_volume_internal(self.energy_mass.volume() - volume_km3);
        if volume_km3 > self.energy_mass.volume() {
            // Need to add volume - create new energy mass and merge
            let additional_volume = volume_km3 - self.energy_mass.volume();
            let mut additional = StandardEnergyMass::new_with_material(
                self.energy_mass.material_type(),
                current_temp,
                additional_volume
            );
            self.energy_mass.merge_em(&additional);
        }
    }

    /// Get the temperature in Kelvin
    pub fn temperature(&self) -> f64 {
        self.energy_mass.temperature()
    }

    /// Set the temperature in Kelvin
    pub fn set_temperature(&mut self, temperature_k: f64) {
        self.energy_mass.set_temperature(temperature_k);
    }

    /// Get the energy in Joules
    pub fn energy(&self) -> f64 {
        self.energy_mass.energy()
    }

    /// Add energy (temperature will increase)
    pub fn add_energy(&mut self, energy_joules: f64) {
        // Since we removed add_energy, we need to set temperature directly
        let current_energy = self.energy_mass.energy();
        let mass_kg = self.energy_mass.mass_kg();
        let specific_heat = self.energy_mass.specific_heat();
        if mass_kg > 0.0 && specific_heat > 0.0 {
            let new_temp = (current_energy + energy_joules) / (mass_kg * specific_heat);
            self.energy_mass.set_temperature(new_temp);
        }
    }

    /// Remove energy (temperature will decrease)
    pub fn remove_energy(&mut self, energy_joules: f64) {
        self.energy_mass.remove_heat(energy_joules);
    }

    /// Get the material profile for this layer's material
    /// Panics if the profile cannot be found (game-ending error)
    pub fn profile(&self) -> &'static crate::material::MaterialProfile {
        self.energy_mass.material_profile()
    }

    /// Get the mass in kg
    pub fn mass_kg(&self) -> f64 {
        self.energy_mass.mass_kg()
    }

    /// Get the density in kg/m³
    pub fn density(&self) -> f64 {
        self.energy_mass.density()
    }

    /// Get the thermal conductivity in W/(m·K)
    pub fn thermal_conductivity(&self) -> f64 {
        self.energy_mass.thermal_conductivity()
    }

    /// Get the specific heat in J/(kg·K)
    pub fn specific_heat(&self) -> f64 {
        self.energy_mass.specific_heat()
    }
}