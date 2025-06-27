use crate::constants::{MANTLE_DENSITY_KGM3, SPECIFIC_HEAT_CAPACITY_MANTLE_J_PER_KG_K};
use crate::material::{MaterialType, get_profile};
use serde::{Deserialize, Serialize};

/// Trait for objects that manage the relationship between energy, mass, and temperature
/// Maintains consistency between these properties using thermodynamic relationships
pub trait EnergyMass: std::any::Any {
    /// Get the current temperature in Kelvin
    fn kelvin(&self) -> f64;

    /// Get the current temperature in Kelvin (alias for kelvin() - deprecated, use kelvin())
    fn temperature(&self) -> f64 {
        self.kelvin()
    }

    /// Get the current energy in Joules (read-only)
    fn energy(&self) -> f64;

    /// Get the current volume in km³ (read-only)
    fn volume(&self) -> f64;

    /// Set the temperature in Kelvin (internal use only - maintains thermodynamic consistency)
    fn set_kelvin(&mut self, kelvin: f64);

    /// Set the temperature in Kelvin (alias for set_kelvin() - deprecated, use set_kelvin())
    fn set_temperature(&mut self, temperature_k: f64) {
        self.set_kelvin(temperature_k);
    }

    /// Get the mass in kg (derived from volume and density)
    fn mass_kg(&self) -> f64;

    /// Get the density in kg/m³
    fn density(&self) -> f64;

    /// Get the specific heat in J/(kg·K)
    fn specific_heat(&self) -> f64;

    /// Get the thermal conductivity in W/(m·K)
    fn thermal_conductivity(&self) -> f64;

    /// Get the material type
    fn material_type(&self) -> MaterialType;

    /// Get the material profile
    fn material_profile(&self) -> &'static crate::material::MaterialProfile;

    /// Scale the entire EnergyMass by a factor (useful for splitting/combining)
    fn scale(&mut self, factor: f64);

    /// Remove heat energy (temperature will decrease, enforces zero minimum)
    fn remove_heat(&mut self, heat_joules: f64);

    /// Add energy (temperature will increase)
    fn add_energy(&mut self, energy_joules: f64);

    /// Radiate energy to another EnergyMass using conductive transfer
    /// Returns the amount of energy transferred (positive = energy flows to other)
    fn radiate_to(&mut self, other: &mut dyn EnergyMass, distance_km: f64, area_km2: f64, time_years: f64) -> f64;

    /// Remove volume (enforces zero minimum, maintains temperature)
    fn remove_volume_internal(&mut self, volume_to_remove: f64);

    /// Merge another EnergyMass into this one
    /// Energy and volume are directly added, resulting in a new blended temperature
    /// The other EnergyMass must have the same material type
    fn merge_em(&mut self, other: &dyn EnergyMass);

    /// Remove a specified volume from this EnergyMass, returning a new EnergyMass with that volume
    /// The removed EnergyMass will have the same temperature as the original
    /// This EnergyMass will have proportionally less volume and energy but maintain the same temperature
    fn remove_volume(&mut self, volume_to_remove: f64) -> Box<dyn EnergyMass>;

    /// Split this EnergyMass into two parts by volume fraction
    /// Returns a new EnergyMass with the specified fraction, this one keeps the remainder
    /// Both will have the same temperature as the original
    fn split_by_fraction(&mut self, fraction: f64) -> Box<dyn EnergyMass>;

    /// Get the R0 thermal transmission coefficient for this material
    /// This controls energy transfer efficiency between layers (tunable for equilibrium)
    fn thermal_transmission_r0(&self) -> f64;

    /// Add core radiance energy influx (2.52e12 J per km² per year)
    /// Only applies to the bottom-most asthenosphere layer
    fn add_core_radiance(&mut self, area_km2: f64, years: f64) {
        // Earth's core radiance: 2.52e12 J per km² per year
        let core_radiance_per_km2_per_year = 2.52e12;
        let energy_influx = core_radiance_per_km2_per_year * area_km2 * years;

        // Add energy by calculating new temperature
        let current_energy = self.energy();
        let mass_kg = self.mass_kg();
        let specific_heat = self.specific_heat();

        if mass_kg > 0.0 && specific_heat > 0.0 {
            let new_temp = (current_energy + energy_influx) / (mass_kg * specific_heat);
            self.set_kelvin(new_temp);
        }
    }

    /// Radiate energy to space based on material thermal properties
    /// The top layer (lithosphere or asthenosphere) radiates as much as possible
    /// Returns the amount of energy radiated (J)
    fn radiate_to_space(&mut self, area_km2: f64, time_years: f64) -> f64 {
        let conductivity = self.thermal_conductivity();
        let temp_k = self.kelvin();
        let r0 = self.thermal_transmission_r0();

        // Maximum radiation based on material properties and temperature
        // Higher conductivity + higher R0 + higher temperature = more radiation
        let radiation_coefficient = conductivity * r0 * 1e12; // Scaling factor
        let radiation_energy = radiation_coefficient * temp_k * area_km2 * time_years;

        // Remove the radiated energy (cooling effect)
        let actual_radiated = radiation_energy.min(self.energy() * 0.5); // Max 50% per step
        self.remove_heat(actual_radiated);

        actual_radiated
    }

    /// Calculate thermal energy transfer between two materials based on conductivity and temperature difference
    /// Scales transfer based on the volume of the hotter material
    /// Returns the energy transfer amount (J)
    fn calculate_thermal_transfer(&self, other: &dyn EnergyMass, diffusion_rate: f64, years: f64) -> f64 {
        let interface_conductivity = 2.0 * self.thermal_conductivity() * other.thermal_conductivity()
            / (self.thermal_conductivity() + other.thermal_conductivity());

        // Calculate interface R0 (harmonic mean like conductivity)
        let interface_r0 = 2.0 * self.thermal_transmission_r0() * other.thermal_transmission_r0()
            / (self.thermal_transmission_r0() + other.thermal_transmission_r0());

        let temp_diff = self.kelvin() - other.kelvin();

        // Determine which material is hotter and use its volume for scaling
        let hot_volume_km3 = if temp_diff > 0.0 {
            self.volume_km3()  // self is hotter
        } else {
            other.volume_km3() // other is hotter
        };

        // Volume scaling factor: larger hot volumes can transfer more energy
        // Use cube root to prevent excessive scaling for very large volumes
        let volume_scale = (hot_volume_km3 / 100.0).powf(0.33).max(0.1); // Baseline 100 km³, minimum 0.1x

        let base_transfer_rate = interface_conductivity * interface_r0 * diffusion_rate * years * 1e12; // Scaling factor
        let energy_transfer = temp_diff * base_transfer_rate * volume_scale;

        energy_transfer
    }

    /// Calculate bulk thermal energy transfer for large volume objects
    /// Uses volume-based thermal diffusion physics appropriate for thick layers
    /// Returns the energy transfer amount (J)
    fn calculate_bulk_thermal_transfer(&self, other: &dyn EnergyMass, layer_thickness_km: f64, years: f64) -> f64 {
        let temp_diff = self.kelvin() - other.kelvin();

        // Calculate thermal diffusivity (m²/s) = conductivity / (density × specific_heat)
        let self_diffusivity = self.thermal_conductivity() / (self.density_kg_m3() * self.specific_heat_capacity_j_per_kg_k());
        let other_diffusivity = other.thermal_conductivity() / (other.density_kg_m3() * other.specific_heat_capacity_j_per_kg_k());

        // Use average diffusivity for the transfer
        let avg_diffusivity = (self_diffusivity + other_diffusivity) / 2.0;

        // Convert to km²/year for our units
        let diffusivity_km2_per_year = avg_diffusivity * 3.15576e7 / 1e6; // seconds/year / m²/km²

        // Calculate diffusion length scale: sqrt(diffusivity × time)
        let diffusion_length_km = (diffusivity_km2_per_year * years).sqrt();

        // Energy transfer efficiency based on diffusion vs layer thickness
        let transfer_efficiency = (diffusion_length_km / layer_thickness_km).min(1.0);

        // Volume-based energy transfer: larger volumes can transfer more energy
        let transfer_volume_km3 = (self.volume_km3() + other.volume_km3()) / 2.0;

        // Base energy transfer rate (J/K/km³/year)
        let base_rate = avg_diffusivity * 1e15; // Scaling factor for realistic energy transfer

        // Total energy transfer
        let energy_transfer = temp_diff * transfer_volume_km3 * transfer_efficiency * base_rate * years;

        energy_transfer
    }
}

/// Standard implementation of EnergyMass using material profiles
/// Material properties are derived from the assigned material type
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct StandardEnergyMass {
    energy_joules: f64,
    volume_km3: f64,
    material_type: MaterialType,
    // Cached properties from material profile for performance
    specific_heat: f64,        // J/(kg·K)
    density: f64,              // kg/m³
    thermal_conductivity: f64, // W/(m·K)
    thermal_transmission_r0: f64, // R0 thermal transmission coefficient
}

impl StandardEnergyMass {
    /// Create a new StandardEnergyMass with specified material, temperature and volume
    pub fn new_with_material(
        material_type: MaterialType,
        temperature_k: f64,
        volume_km3: f64,
    ) -> Self {
        let profile = get_profile(material_type).expect(&format!(
            "Material profile not found for {:?}",
            material_type
        ));

        // Generate random R0 value within material's range
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let r0_range = profile.thermal_transmission_r0_max - profile.thermal_transmission_r0_min;
        let random_r0 = profile.thermal_transmission_r0_min + rng.random::<f64>() * r0_range;

        let mut energy_mass = Self {
            energy_joules: 0.0,
            volume_km3,
            material_type,
            specific_heat: profile.specific_heat_capacity_j_per_kg_k,
            density: profile.density_kg_m3,
            thermal_conductivity: profile.thermal_conductivity_w_m_k,
            thermal_transmission_r0: random_r0,
        };
        energy_mass.set_temperature(temperature_k);
        energy_mass
    }

    /// Create a new StandardEnergyMass with specified material, energy and volume
    pub fn new_with_material_energy(
        material_type: MaterialType,
        energy_joules: f64,
        volume_km3: f64,
    ) -> Self {
        let profile = get_profile(material_type).expect(&format!(
            "Material profile not found for {:?}",
            material_type
        ));

        // Generate random R0 value within material's range
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let r0_range = profile.thermal_transmission_r0_max - profile.thermal_transmission_r0_min;
        let random_r0 = profile.thermal_transmission_r0_min + rng.random::<f64>() * r0_range;

        Self {
            energy_joules,
            volume_km3,
            material_type,
            specific_heat: profile.specific_heat_capacity_j_per_kg_k,
            density: profile.density_kg_m3,
            thermal_conductivity: profile.thermal_conductivity_w_m_k,
            thermal_transmission_r0: random_r0,
        }
    }
}

impl EnergyMass for StandardEnergyMass {
    /// Get the current temperature in Kelvin
    fn kelvin(&self) -> f64 {
        let mass_kg = self.mass_kg();
        if mass_kg == 0.0 {
            return 0.0;
        }
        self.energy_joules / (mass_kg * self.specific_heat)
    }

    /// Set the temperature in Kelvin, updating energy accordingly (volume stays constant)
    fn set_kelvin(&mut self, kelvin: f64) {
        let mass_kg = self.mass_kg();
        self.energy_joules = mass_kg * self.specific_heat * kelvin;
    }

    /// Get the current energy in Joules (read-only)
    fn energy(&self) -> f64 {
        self.energy_joules
    }

    /// Get the current volume in km³ (read-only)
    fn volume(&self) -> f64 {
        self.volume_km3
    }

    /// Get the mass in kg (derived from volume and density)
    fn mass_kg(&self) -> f64 {
        const KM3_TO_M3: f64 = 1.0e9;
        let volume_m3 = self.volume_km3 * KM3_TO_M3;
        volume_m3 * self.density
    }

    /// Get the density in kg/m³
    fn density(&self) -> f64 {
        self.density
    }

    /// Get the specific heat in J/(kg·K)
    fn specific_heat(&self) -> f64 {
        self.specific_heat
    }

    /// Get the thermal conductivity in W/(m·K)
    fn thermal_conductivity(&self) -> f64 {
        self.thermal_conductivity
    }

    /// Get the material type
    fn material_type(&self) -> MaterialType {
        self.material_type
    }

    /// Get the material profile
    fn material_profile(&self) -> &'static crate::material::MaterialProfile {
        get_profile(self.material_type).expect(&format!(
            "Material profile not found for {:?}",
            self.material_type
        ))
    }

    /// Scale the entire EnergyMass by a factor (useful for splitting/combining)
    fn scale(&mut self, factor: f64) {
        self.energy_joules *= factor;
        self.volume_km3 *= factor;
        // Temperature remains the same, mass and energy scale proportionally
    }

    /// Remove heat energy (temperature will decrease, enforces zero minimum)
    fn remove_heat(&mut self, heat_joules: f64) {
        self.energy_joules = (self.energy_joules - heat_joules).max(0.0);
    }

    /// Add energy (temperature will increase)
    fn add_energy(&mut self, energy_joules: f64) {
        self.energy_joules += energy_joules;
    }

    /// Radiate energy to another EnergyMass using conductive transfer (Fourier's law)
    /// Returns the amount of energy transferred (positive = energy flows to other)
    fn radiate_to(&mut self, other: &mut dyn EnergyMass, distance_km: f64, area_km2: f64, time_years: f64) -> f64 {
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
        let area_m2 = area_km2 * 1e6;  // km² to m²
        let distance_m = distance_km * 1000.0;  // km to m
        let time_seconds = time_years * 365.25 * 24.0 * 3600.0;  // years to seconds

        // Fourier's law: Q = k * A * ΔT * t / d
        let energy_transfer = interface_conductivity * area_m2 * temp_diff * time_seconds / distance_m;

        // Limit transfer to prevent temperature inversion
        let max_transfer = self.energy() * 0.1; // Max 10% per step
        let actual_transfer = energy_transfer.abs().min(max_transfer);

        // Apply the transfer
        if temp_diff > 0.0 {
            // I'm hotter - I lose energy, other gains energy
            self.remove_heat(actual_transfer);
            other.add_energy(actual_transfer);
            actual_transfer
        } else {
            // Other is hotter - other loses energy, I gain energy
            other.add_energy(-actual_transfer);
            self.remove_heat(-actual_transfer);  // Remove negative energy = add positive energy
            -actual_transfer
        }
    }

    /// Remove volume (enforces zero minimum, maintains temperature)
    fn remove_volume_internal(&mut self, volume_to_remove: f64) {
        if volume_to_remove <= 0.0 {
            return; // Nothing to remove
        }

        let current_temp = self.temperature();
        self.volume_km3 = (self.volume_km3 - volume_to_remove).max(0.0);

        // Recalculate energy to maintain temperature with new volume
        if self.volume_km3 > 0.0 {
            self.set_temperature(current_temp);
        } else {
            self.energy_joules = 0.0; // No volume means no energy
        }
    }

    /// Merge another EnergyMass into this one
    /// Energy and volume are directly added, resulting in a new blended temperature
    /// The other EnergyMass must have the same material type
    fn merge_em(&mut self, other: &dyn EnergyMass) {
        // Verify compatible material types
        assert_eq!(
            self.material_type,
            other.material_type(),
            "Cannot merge EnergyMass with different materials: {:?} vs {:?}",
            self.material_type,
            other.material_type()
        );

        // Add energy and volume directly
        self.energy_joules += other.energy();
        self.volume_km3 += other.volume();
        // Temperature will be automatically recalculated based on new energy/mass ratio
    }

    /// Remove a specified volume from this EnergyMass, returning a new EnergyMass with that volume
    /// The removed EnergyMass will have the same temperature as the original
    /// This EnergyMass will have proportionally less volume and energy but maintain the same temperature
    fn remove_volume(&mut self, volume_to_remove: f64) -> Box<dyn EnergyMass> {
        if volume_to_remove <= 0.0 {
            panic!("Cannot remove zero or negative volume");
        }
        if volume_to_remove >= self.volume_km3 {
            panic!(
                "Cannot remove more volume ({}) than available ({})",
                volume_to_remove, self.volume_km3
            );
        }

        let current_temp = self.temperature();
        let fraction_removed = volume_to_remove / self.volume_km3;
        let energy_to_remove = self.energy_joules * fraction_removed;

        // Create the removed EnergyMass with same temperature and material
        let removed = StandardEnergyMass::new_with_material(
            self.material_type,
            current_temp,
            volume_to_remove,
        );

        // Update this EnergyMass
        self.volume_km3 -= volume_to_remove;
        self.energy_joules -= energy_to_remove;

        Box::new(removed)
    }

    /// Split this EnergyMass into two parts by volume fraction
    /// Returns a new EnergyMass with the specified fraction, this one keeps the remainder
    /// Both will have the same temperature as the original
    fn split_by_fraction(&mut self, fraction: f64) -> Box<dyn EnergyMass> {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_create_with_temperature() {
        let energy_mass =
            StandardEnergyMass::new_with_material(MaterialType::Silicate, 1500.0, 100.0);

        assert_abs_diff_eq!(energy_mass.temperature(), 1500.0, epsilon = 0.01);
        assert_eq!(energy_mass.volume(), 100.0);
        assert!(energy_mass.energy() > 0.0);
    }

    #[test]
    fn test_temperature_energy_roundtrip() {
        let mut energy_mass =
            StandardEnergyMass::new_with_material(MaterialType::Silicate, 1673.15, 100.0);

        // Get the energy
        let energy = energy_mass.energy();

        // Set energy back
        // Set energy by calculating the required temperature
        let mass_kg = energy_mass.mass_kg();
        let specific_heat = energy_mass.specific_heat();
        if mass_kg > 0.0 && specific_heat > 0.0 {
            let new_temp = energy / (mass_kg * specific_heat);
            energy_mass.set_temperature(new_temp);
        }

        // Temperature should be the same
        assert_abs_diff_eq!(energy_mass.temperature(), 1673.15, epsilon = 0.01);
    }

    #[test]
    fn test_volume_changes_maintain_temperature() {
        let mut energy_mass =
            StandardEnergyMass::new_with_material(MaterialType::Silicate, 1500.0, 100.0);
        let initial_energy = energy_mass.energy();

        // Double the volume
        // Set volume by creating a new energy mass with the desired volume and same temperature
        let current_temp = energy_mass.temperature();
        energy_mass =
            StandardEnergyMass::new_with_material(MaterialType::Silicate, current_temp, 200.0);

        // Temperature should stay the same
        assert_abs_diff_eq!(energy_mass.temperature(), 1500.0, epsilon = 0.01);

        // Energy should double (more mass at same temperature)
        assert_abs_diff_eq!(energy_mass.energy(), initial_energy * 2.0, epsilon = 1.0);
    }

    #[test]
    fn test_energy_changes_affect_temperature() {
        let mut energy_mass =
            StandardEnergyMass::new_with_material(MaterialType::Silicate, 1500.0, 100.0);
        let initial_energy = energy_mass.energy();

        // Double the energy
        // Set energy by calculating the required temperature
        let mass_kg = energy_mass.mass_kg();
        let specific_heat = energy_mass.specific_heat();
        if mass_kg > 0.0 && specific_heat > 0.0 {
            let new_temp = (initial_energy * 2.0) / (mass_kg * specific_heat);
            energy_mass.set_temperature(new_temp);
        }

        // Temperature should double
        assert_abs_diff_eq!(energy_mass.temperature(), 3000.0, epsilon = 0.01);

        // Volume should stay the same
        assert_eq!(energy_mass.volume(), 100.0);
    }

    #[test]
    fn test_add_remove_energy() {
        let mut energy_mass =
            StandardEnergyMass::new_with_material(MaterialType::Silicate, 1500.0, 100.0);
        let initial_temp = energy_mass.temperature();
        let initial_energy = energy_mass.energy();

        // Add energy
        // Add energy by calculating the new temperature
        let current_energy = energy_mass.energy();
        let mass_kg = energy_mass.mass_kg();
        let specific_heat = energy_mass.specific_heat();
        if mass_kg > 0.0 && specific_heat > 0.0 {
            let new_temp = (current_energy + initial_energy) / (mass_kg * specific_heat);
            energy_mass.set_temperature(new_temp);
        }
        assert_abs_diff_eq!(
            energy_mass.temperature(),
            initial_temp * 2.0,
            epsilon = 0.01
        );

        // Remove energy back to original
        energy_mass.remove_heat(initial_energy);
        assert_abs_diff_eq!(energy_mass.temperature(), initial_temp, epsilon = 0.01);
    }

    #[test]
    fn test_scaling() {
        let mut energy_mass =
            StandardEnergyMass::new_with_material(MaterialType::Silicate, 1500.0, 100.0);
        let initial_temp = energy_mass.temperature();
        let initial_energy = energy_mass.energy();
        let initial_volume = energy_mass.volume();

        // Scale by 0.5
        energy_mass.scale(0.5);

        // Temperature should stay the same
        assert_abs_diff_eq!(energy_mass.temperature(), initial_temp, epsilon = 0.01);

        // Energy and volume should be halved
        assert_abs_diff_eq!(energy_mass.energy(), initial_energy * 0.5, epsilon = 1.0);
        assert_abs_diff_eq!(energy_mass.volume(), initial_volume * 0.5, epsilon = 0.01);
    }

    #[test]
    fn test_lithosphere_relevant_temperatures() {
        let test_temps = vec![1673.15, 1773.15, 1873.15]; // Silicate temps

        for temp in test_temps {
            let energy_mass =
                StandardEnergyMass::new_with_material(MaterialType::Silicate, temp, 100.0);
            assert_abs_diff_eq!(energy_mass.temperature(), temp, epsilon = 0.01);

            println!(
                "Temp: {:.2} K, Energy: {:.2e} J, Mass: {:.2e} kg",
                temp,
                energy_mass.energy(),
                energy_mass.mass_kg()
            );
        }
    }

    #[test]
    fn test_merge_em() {
        let mut em1 = StandardEnergyMass::new_with_material(MaterialType::Silicate, 1500.0, 100.0); // Hot
        let em2 = StandardEnergyMass::new_with_material(MaterialType::Silicate, 1000.0, 50.0); // Cool

        let initial_energy1 = em1.energy();
        let initial_energy2 = em2.energy();
        let initial_volume1 = em1.volume();
        let initial_volume2 = em2.volume();

        // Calculate expected blended temperature
        let total_mass = em1.mass_kg() + em2.mass_kg();
        let total_energy = initial_energy1 + initial_energy2;
        let expected_temp = total_energy / (total_mass * em1.specific_heat());

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
        assert_abs_diff_eq!(em1.temperature(), expected_temp, epsilon = 0.1);

        println!(
            "Merged: {:.2} K (expected {:.2} K), Volume: {:.1} km³, Energy: {:.2e} J",
            em1.temperature(),
            expected_temp,
            em1.volume(),
            em1.energy()
        );
    }

    #[test]
    fn test_remove_volume() {
        let mut original =
            StandardEnergyMass::new_with_material(MaterialType::Silicate, 1600.0, 200.0);
        let initial_temp = original.temperature();
        let initial_energy = original.energy();
        let initial_volume = original.volume();

        // Remove 1/4 of the volume
        let removed = original.remove_volume(50.0);

        // Check removed EnergyMass
        assert_abs_diff_eq!(removed.temperature(), initial_temp, epsilon = 0.01);
        assert_abs_diff_eq!(removed.volume(), 50.0, epsilon = 0.01);
        assert_abs_diff_eq!(removed.energy(), initial_energy * 0.25, epsilon = 1.0);

        // Check remaining EnergyMass
        assert_abs_diff_eq!(original.temperature(), initial_temp, epsilon = 0.01);
        assert_abs_diff_eq!(original.volume(), 150.0, epsilon = 0.01);
        assert_abs_diff_eq!(original.energy(), initial_energy * 0.75, epsilon = 1.0);

        println!(
            "Original: {:.2} K, {:.1} km³, {:.2e} J",
            original.temperature(),
            original.volume(),
            original.energy()
        );
        println!(
            "Removed: {:.2} K, {:.1} km³, {:.2e} J",
            removed.temperature(),
            removed.volume(),
            removed.energy()
        );
    }

    #[test]
    fn test_split_by_fraction() {
        let mut original =
            StandardEnergyMass::new_with_material(MaterialType::Silicate, 1700.0, 100.0);
        let initial_temp = original.temperature();
        let initial_energy = original.energy();
        let initial_volume = original.volume();

        // Split off 30%
        let split_off = original.split_by_fraction(0.3);

        // Check split-off part
        assert_abs_diff_eq!(split_off.temperature(), initial_temp, epsilon = 0.01);
        assert_abs_diff_eq!(split_off.volume(), 30.0, epsilon = 0.01);
        assert_abs_diff_eq!(split_off.energy(), initial_energy * 0.3, epsilon = 1.0);

        // Check remaining part
        assert_abs_diff_eq!(original.temperature(), initial_temp, epsilon = 0.01);
        assert_abs_diff_eq!(original.volume(), 70.0, epsilon = 0.01);
        assert_abs_diff_eq!(original.energy(), initial_energy * 0.7, epsilon = 1.0);

        println!(
            "Remaining: {:.2} K, {:.1} km³, {:.2e} J",
            original.temperature(),
            original.volume(),
            original.energy()
        );
        println!(
            "Split off: {:.2} K, {:.1} km³, {:.2e} J",
            split_off.temperature(),
            split_off.volume(),
            split_off.energy()
        );
    }

    #[test]
    fn test_merge_and_split_roundtrip() {
        let mut em1 = StandardEnergyMass::new_with_material(MaterialType::Silicate, 1500.0, 80.0);
        let em2 = StandardEnergyMass::new_with_material(MaterialType::Silicate, 1800.0, 20.0);

        let initial_temp1 = em1.temperature();
        let initial_volume1 = em1.volume();
        let initial_energy1 = em1.energy();

        // Merge em2 into em1
        em1.merge_em(&em2);
        let merged_temp = em1.temperature();

        // Split off the same volume that was added
        let split_off = em1.remove_volume(20.0);

        // The remaining should be close to original (but temperature will be different due to mixing)
        assert_abs_diff_eq!(em1.volume(), initial_volume1, epsilon = 0.01);

        // The split-off should have the merged temperature
        assert_abs_diff_eq!(split_off.temperature(), merged_temp, epsilon = 0.01);
        assert_abs_diff_eq!(split_off.volume(), 20.0, epsilon = 0.01);

        println!(
            "Original: {:.2} K, {:.1} km³",
            initial_temp1, initial_volume1
        );
        println!("After merge: {:.2} K, {:.1} km³", merged_temp, em1.volume());
        println!(
            "After split: {:.2} K, {:.1} km³",
            em1.temperature(),
            em1.volume()
        );
        println!(
            "Split off: {:.2} K, {:.1} km³",
            split_off.temperature(),
            split_off.volume()
        );
    }
}
