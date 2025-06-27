use serde::{Deserialize, Serialize};
use crate::energy_mass::{EnergyMass, StandardEnergyMass};
use crate::material::MaterialType;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AsthCellLithosphere {
    pub height_km: f64,
    pub(crate) energy_mass: StandardEnergyMass,
}

impl AsthCellLithosphere {
    pub fn growth_per_year(&self, surface_temp_k: f64) -> f64 {
        let profile = self.profile();
        profile.max_lith_growth_km_per_year
            * if surface_temp_k >= profile.peak_lith_growth_temp_kv {
            // At or below peak growth temperature - maximum growth
            1.0
        } else {
            // Between peak and formation temperature - reduced growth
            let temp_range = profile.max_lith_formation_temp_kv
                - profile.peak_lith_growth_temp_kv;
            let temp_above_peak = surface_temp_k - profile.peak_lith_growth_temp_kv;
            (1.0 - (temp_above_peak / temp_range)).max(0.0)
        }
    }

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
    pub fn material_type(&self) -> MaterialType {
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

    /// Get the energy in Joules
    pub fn energy_joules(&self) -> f64 {
        self.energy_mass.energy()
    }

    /// Add energy (temperature will increase)
    pub fn add_energy(&mut self, energy_joules: f64) {
        // Since we removed add_energy, we need to set temperature directly
        let current_energy = self.energy_mass.energy();
        let mass_kg = self.energy_mass.mass_kg();
        let specific_heat = self.energy_mass.specific_heat_j_kg_k();
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
        self.energy_mass.density_kgm3()
    }

    /// Get the thermal conductivity in W/(m·K)
    pub fn thermal_conductivity(&self) -> f64 {
        self.energy_mass.thermal_conductivity()
    }

    /// Get the specific heat in J/(kg·K)
    pub fn specific_heat(&self) -> f64 {
        self.energy_mass.specific_heat_j_kg_k()
    }

    /// Get the temperature in Kelvin
    pub fn kelvin(&self) -> f64 {
        self.energy_mass.kelvin()
    }

    /// Get mutable reference to the internal EnergyMass for direct thermal operations
    pub fn energy_mass_mut(&mut self) -> &mut dyn crate::energy_mass::EnergyMass {
        &mut self.energy_mass
    }

    /// this is the amount of material that will melt off per year
    /// based on the temperature of the asthenosphere below it.
    pub fn  melt_from_below_km_per_year(
        &self,
        temp_asth_k: f64,
    ) -> f64 {

        let material = self.profile();
        if temp_asth_k <= material.max_lith_formation_temp_kv {
            return 0.0;
        }

        let delta_t = temp_asth_k - material.max_lith_formation_temp_kv;

        let energy_per_m3 = material.specific_heat_capacity_j_per_kg_k
            * material.density_kg_m3
            * delta_t;

        // Joules per year per m²
        let thickness_m = self.height_km * 1000.0;
        let flux_w_m2 = material.thermal_conductivity_w_m_k * delta_t / thickness_m; // W/m²
        let joules_per_year = flux_w_m2 * 31_536_000.0;


        // Convert to km/year
        let melt_m = joules_per_year / energy_per_m3;

        // Return in km/year
        melt_m / 1000.0
    }

    /// Grow this lithosphere layer by the specified amount
    /// Returns the excess height if growth exceeds max individual layer height
    pub fn grow(&mut self, growth_km: f64, area_km2: f64, max_layer_height_km: f64, formation_temp_k: f64) -> f64 {
        if growth_km <= 0.0 {
            return 0.0;
        }

        let new_height = self.height_km + growth_km;

        if new_height <= max_layer_height_km {
            // Growth fits within this layer
            self.height_km = new_height;
            let new_volume = area_km2 * self.height_km;
            self.set_volume_km3(new_volume);

            // Set temperature to formation temperature if layer has no energy
            if self.energy_joules() <= 0.0 {
                let mass_kg = self.mass_kg();
                let specific_heat = self.specific_heat();
                if mass_kg > 0.0 && specific_heat > 0.0 {
                    let target_energy = mass_kg * specific_heat * formation_temp_k;
                    self.add_energy(target_energy);
                }
            }

            0.0 // No excess
        } else {
            // Growth exceeds max layer height - cap this layer and return excess
            let excess = new_height - max_layer_height_km;
            self.height_km = max_layer_height_km;
            let new_volume = area_km2 * self.height_km;
            self.set_volume_km3(new_volume);

            // Set temperature to formation temperature if layer has no energy
            if self.energy_joules() <= 0.0 {
                let mass_kg = self.mass_kg();
                let specific_heat = self.specific_heat();
                if mass_kg > 0.0 && specific_heat > 0.0 {
                    let target_energy = mass_kg * specific_heat * formation_temp_k;
                    self.add_energy(target_energy);
                }
            }

            excess
        }
    }

    /// Melt this lithosphere layer by the specified amount
    /// Returns the amount actually melted (may be less than requested if layer is too thin)
    pub fn melt(&mut self, melt_km: f64) -> f64 {
        if melt_km <= 0.0 || self.height_km <= 0.0 {
            return 0.0;
        }
        let actual_melt = melt_km.min(self.height_km);
        self.height_km -= actual_melt;
        actual_melt
    }

    /// Process melting from asthenosphere temperature and add melted energy back to asthenosphere
    /// Returns energy removed
    pub fn process_melting(&mut self, asthenosphere_temp_k: f64, years_per_step: u32) -> f64 {
        if self.height_km <= 0.0 {
            return 0.0;
        }

        // Store original height and energy before melting
        let original_height = self.height_km;
        let original_energy = self.energy_joules();

        // Calculate melting rate using the existing melt_from_below_km_per_year function
        let melt_rate_km_per_year = self.melt_from_below_km_per_year(asthenosphere_temp_k);

        if melt_rate_km_per_year > 0.0 {
            let melt_rate_km_per_step = melt_rate_km_per_year * years_per_step as f64;

            let melted_height = self.melt(melt_rate_km_per_step);

            // Add melted energy back to asthenosphere
            if melted_height > 0.0 && original_height > 0.0 {
                // Melted energy is proportional to the fraction of layer that melted
                let new_volume = self.volume_km3() * (self.height_km / original_height);
                self.set_volume_km3(new_volume);
                let melted_energy = original_energy * (melted_height / original_height);
                self.remove_energy(melted_energy);
                return melted_energy;
            }
        }

        0.0
    }
}