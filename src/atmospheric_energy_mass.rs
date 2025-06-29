use serde::{Deserialize, Serialize};
use crate::constants::{SECONDS_PER_YEAR, SIGMA_KM2_YEAR};
use crate::energy_mass_composite::{get_profile_fast, EnergyMassComposite, MaterialCompositeType, MaterialPhase, MaterialStateProfile};
use crate::material_composite::{MaterialComposite, MATERIAL_COMPOSITES};

/// Atmospheric EnergyMass implementation using composite material system
/// Uses dynamic property lookup based on current phase state
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AtmosphericEnergyMass {
    energy_joules: f64,
    volume_km3: f64,
    height_km: f64,
    material_type: MaterialCompositeType, // Always Air for atmospheric materials
    phase: MaterialPhase,         // Current atmospheric state (usually Gas)
    custom_density_kg_m3: f64,            // Custom density for this atmospheric layer
    thermal_transmission_r0: f64,         // R0 thermal transmission coefficient (set at creation)
}

impl AtmosphericEnergyMass {
    /// Create a new AtmosphericEnergyMass with custom density and phase
    pub fn new_with_phase(
        temperature_k: f64,
        volume_km3: f64,
        height_km: f64,
        density_kg_m3: f64,
        phase: MaterialPhase,
    ) -> Self {
        // Use Air material type for atmospheric materials
        let phase = phase.clone();
        let profile = get_profile_fast(&MaterialCompositeType::Air, &phase);

        // Generate random R0 value within material's range
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let r0_range = profile.thermal_transmission_r0_max - profile.thermal_transmission_r0_min;
        let random_r0 = profile.thermal_transmission_r0_min + rng.random::<f64>() * r0_range;

        let mut atm = Self {
            energy_joules: 0.0,
            volume_km3,
            height_km,
            material_type: MaterialCompositeType::Air,
            phase,
            custom_density_kg_m3: density_kg_m3,
            thermal_transmission_r0: random_r0,
        };
        atm.set_kelvin(temperature_k);
        atm
    }

    /// Create a new AtmosphericEnergyMass with custom density (defaults to Gas phase)
    pub fn new(temperature_k: f64, volume_km3: f64, height_km: f64, density_kg_m3: f64) -> Self {
        Self::new_with_phase(
            temperature_k,
            volume_km3,
            height_km,
            density_kg_m3,
            MaterialPhase::Gas,
        )
    }
}

impl EnergyMassComposite for AtmosphericEnergyMass {
    fn kelvin(&self) -> f64 {
        let mass_kg = self.mass_kg();
        if mass_kg == 0.0 {
            return 0.0;
        }
        self.energy_joules / (mass_kg * self.specific_heat_j_kg_k())
    }

    fn set_kelvin(&mut self, kelvin: f64) {
        let mass_kg = self.mass_kg();
        self.energy_joules = mass_kg * self.specific_heat_j_kg_k() * kelvin;
    }

    fn energy(&self) -> f64 {
        self.energy_joules
    }

    fn volume(&self) -> f64 {
        self.volume_km3
    }

    fn height_km(&self) -> f64 {
        self.height_km
    }

    fn mass_kg(&self) -> f64 {
        const KM3_TO_M3: f64 = 1.0e9;
        let volume_m3 = self.volume_km3 * KM3_TO_M3;
        volume_m3 * self.density_kgm3()
    }

    fn density_kgm3(&self) -> f64 {
        get_profile_fast(&self.material_type, &self.phase).density_kg_m3
    }

    fn specific_heat_j_kg_k(&self) -> f64 {
        get_profile_fast(&self.material_type, &self.phase).specific_heat_capacity_j_per_kg_k
    }

    fn thermal_conductivity(&self) -> f64 {
        get_profile_fast(&self.material_type, &self.phase).thermal_conductivity_w_m_k
    }

    fn material_composite_type(&self) -> MaterialCompositeType {
        self.material_type
    }

    fn material_composite_profile(&self) -> &'static MaterialStateProfile {
        get_profile_fast(&self.material_type, &self.phase)
    }

    fn material_composite(&self) -> MaterialComposite {
        match MATERIAL_COMPOSITES.get(&self.material_composite_type()) {
            None => panic!("cannot find material type: {:?}", self.material_composite_type()),
            Some(composite) => composite.clone()
        }
    }

    fn scale(&mut self, factor: f64) {
        self.energy_joules *= factor;
        self.volume_km3 *= factor;
    }

    fn remove_joules(&mut self, heat_joules: f64) {
        self.energy_joules = (self.energy_joules - heat_joules).max(0.0);
    }

    fn add_joules(&mut self, energy_joules: f64) {
        self.energy_joules += energy_joules;
    }

    // Implement remaining required trait methods with simplified atmospheric behavior
    fn radiate_to(
        &mut self,
        other: &mut dyn EnergyMassComposite,
        distance_km: f64,
        area_km2: f64,
        time_years: f64,
    ) -> f64 {
        // Simplified atmospheric radiation - use same logic as StandardEnergyMass
        let my_temp = self.kelvin();
        let other_temp = other.kelvin();

        if (my_temp - other_temp).abs() < 0.1 {
            return 0.0;
        }

        let my_conductivity = self.thermal_conductivity();
        let other_conductivity = other.thermal_conductivity();

        let interface_conductivity = if my_conductivity > 0.0 && other_conductivity > 0.0 {
            2.0 * my_conductivity * other_conductivity / (my_conductivity + other_conductivity)
        } else {
            0.0
        };

        let temp_diff = my_temp - other_temp;
        let area_m2 = area_km2 * 1e6;
        let distance_m = distance_km * 1000.0;
        let time_seconds = time_years * 365.25 * 24.0 * 3600.0;

        let energy_transfer =
            interface_conductivity * area_m2 * temp_diff * time_seconds / distance_m;
        let max_transfer = self.energy() * 0.1;
        let actual_transfer = energy_transfer.abs().min(max_transfer);

        if temp_diff > 0.0 {
            self.remove_joules(actual_transfer);
            other.add_joules(actual_transfer);
            actual_transfer
        } else {
            other.add_joules(-actual_transfer);
            self.remove_joules(-actual_transfer);
            -actual_transfer
        }
    }

    fn radiate_to_space(&mut self, area_km2: f64, time_years: f64) -> f64 {
        self.radiate_to_space_with_skin_depth(area_km2, time_years, 1.0)
    }

    fn radiate_to_space_with_skin_depth(
        &mut self,
        area_km2: f64,
        time_years: f64,
        _energy_throttle: f64,
    ) -> f64 {
        let surface_temp = self.kelvin();
        let skin_depth = self.skin_depth_km(time_years);
        let effective_skin_depth = skin_depth.min(self.height_km());
        let radiation_fraction = if self.height_km() > 0.0 {
            effective_skin_depth / self.height_km()
        } else {
            1.0
        };

        let radiated_energy_per_km2 = SIGMA_KM2_YEAR * surface_temp.powi(4) * time_years;
        let total_radiated_energy = radiated_energy_per_km2 * area_km2;
        let skin_energy = self.energy() * radiation_fraction;
        let energy_to_remove = total_radiated_energy.min(skin_energy);

        self.remove_joules(energy_to_remove);
        energy_to_remove
    }

    fn remove_volume_internal(&mut self, volume_to_remove: f64) {
        if volume_to_remove <= 0.0 {
            return;
        }

        let current_temp = self.kelvin();
        self.volume_km3 = (self.volume_km3 - volume_to_remove).max(0.0);

        if self.volume_km3 > 0.0 {
            self.set_kelvin(current_temp);
        } else {
            self.energy_joules = 0.0;
        }
    }

    fn merge_em(&mut self, other: &dyn EnergyMassComposite) {
        assert_eq!(
            self.material_composite_type(),
            other.material_composite_type(),
            "Cannot merge EnergyMass with different materials"
        );

        self.energy_joules += other.energy();
        self.volume_km3 += other.volume();
    }

    fn remove_volume(&mut self, volume_to_remove: f64) -> Box<dyn EnergyMassComposite> {
        if volume_to_remove <= 0.0 {
            panic!("Cannot remove zero or negative volume");
        }
        if volume_to_remove >= self.volume_km3 {
            panic!("Cannot remove more volume than available");
        }

        let current_temp = self.kelvin();
        let fraction_removed = volume_to_remove / self.volume_km3;
        let energy_to_remove = self.energy_joules * fraction_removed;

        let removed = AtmosphericEnergyMass::new_with_phase(
            current_temp,
            volume_to_remove,
            self.height_km,
            self.custom_density_kg_m3,
            self.phase,
        );

        self.volume_km3 -= volume_to_remove;
        self.energy_joules -= energy_to_remove;

        Box::new(removed)
    }

    fn split_by_fraction(&mut self, fraction: f64) -> Box<dyn EnergyMassComposite> {
        if fraction <= 0.0 || fraction >= 1.0 {
            panic!("Fraction must be between 0 and 1");
        }

        let volume_to_remove = self.volume_km3 * fraction;
        self.remove_volume(volume_to_remove)
    }

    fn thermal_transmission_r0(&self) -> f64 {
        self.thermal_transmission_r0
    }

    fn skin_depth_km(&self, time_years: f64) -> f64 {
        let kappa =
            self.thermal_conductivity() / (self.density_kgm3() * self.specific_heat_j_kg_k());
        let dt_secs = time_years * SECONDS_PER_YEAR;
        (kappa * dt_secs).sqrt() / 1000.0
    }

    fn add_energy(&mut self, energy_joules: f64) {
        self.energy_joules += energy_joules;
    }

    fn remove_energy(&mut self, energy_joules: f64) {
        self.energy_joules = (self.energy_joules - energy_joules).max(0.0);
    }
}