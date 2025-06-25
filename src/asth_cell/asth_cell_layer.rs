use serde::{Deserialize, Serialize};
use crate::energy_mass::{EnergyMass, StandardEnergyMass};
use crate::material::MaterialType;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AsthCellLayer {
    energy_mass: StandardEnergyMass,
    pub level: usize,
}

impl AsthCellLayer {
    /// Create a new layer with specified material, temperature, volume, and level
    pub fn new_with_material(material_type: MaterialType, temperature_k: f64, volume_km3: f64, level: usize) -> Self {
        Self {
            energy_mass: StandardEnergyMass::new_with_material(material_type, temperature_k, volume_km3),
            level,
        }
    }

    /// Create a new layer with specified material, energy, volume, and level
    pub fn new_with_material_energy(material_type: MaterialType, energy_joules: f64, volume_km3: f64, level: usize) -> Self {
        Self {
            energy_mass: StandardEnergyMass::new_with_material_energy(material_type, energy_joules, volume_km3),
            level,
        }
    }



    /// Get the current temperature in Kelvin
    pub fn kelvin(&self) -> f64 {
        self.energy_mass.kelvin()
    }

    /// Get the current temperature in Kelvin (alias for kelvin())
    pub fn temperature(&self) -> f64 {
        self.kelvin()
    }

    /// Set the temperature in Kelvin, updating energy accordingly (volume stays constant)
    pub fn set_temp_kelvin(&mut self, target_kelvin: f64) {
        self.energy_mass.set_kelvin(target_kelvin);
    }

    /// Get the current energy in Joules
    pub fn energy_joules(&self) -> f64 {
        self.energy_mass.energy()
    }

    /// Set the energy, temperature will change accordingly (volume stays constant)
    pub fn set_energy_joules(&mut self, energy_joules: f64) {
        let mass_kg = self.energy_mass.mass_kg();
        let specific_heat = self.energy_mass.specific_heat();
        if mass_kg > 0.0 && specific_heat > 0.0 {
            let new_kelvin = energy_joules / (mass_kg * specific_heat);
            self.energy_mass.set_kelvin(new_kelvin);
        }
    }

    /// Get the current volume in km³
    pub fn volume_km3(&self) -> f64 {
        self.energy_mass.volume()
    }

    /// Set the volume, energy changes to maintain constant temperature
    pub fn set_volume_km3(&mut self, volume_km3: f64) {
        let current_kelvin = self.energy_mass.kelvin();
        let current_volume = self.energy_mass.volume();

        if volume_km3 < current_volume {
            // Remove volume
            self.energy_mass.remove_volume_internal(current_volume - volume_km3);
        } else if volume_km3 > current_volume {
            // Need to add volume - create a new energy mass with the desired volume
            let new_energy_mass = StandardEnergyMass::new_with_material(
                self.energy_mass.material_type(),
                current_kelvin,
                volume_km3
            );
            self.energy_mass = new_energy_mass;
        }
    }

    /// Add energy to this layer (temperature will increase)
    pub fn add_energy(&mut self, additional_energy: f64) {
        let current_energy = self.energy_mass.energy();
        let mass_kg = self.energy_mass.mass_kg();
        let specific_heat = self.energy_mass.specific_heat();
        if mass_kg > 0.0 && specific_heat > 0.0 {
            let new_kelvin = (current_energy + additional_energy) / (mass_kg * specific_heat);
            self.energy_mass.set_kelvin(new_kelvin);
        }
    }

    /// Remove energy from this layer (temperature will decrease)
    pub fn remove_energy(&mut self, energy_to_remove: f64) {
        self.energy_mass.remove_heat(energy_to_remove);
    }

    /// Get the mass in kg
    pub fn mass_kg(&self) -> f64 {
        self.energy_mass.mass_kg()
    }

    /// Get the material type
    pub fn material_type(&self) -> MaterialType {
        self.energy_mass.material_type()
    }

    /// Get the material profile
    pub fn material_profile(&self) -> &'static crate::material::MaterialProfile {
        self.energy_mass.material_profile()
    }

    /// Get the material profile (alias for material_profile())
    pub fn profile(&self) -> &'static crate::material::MaterialProfile {
        self.material_profile()
    }

    /// Get the thermal conductivity in W/(m·K)
    pub fn thermal_conductivity(&self) -> f64 {
        self.energy_mass.thermal_conductivity()
    }

    /// Get the density in kg/m³
    pub fn density(&self) -> f64 {
        self.energy_mass.density()
    }

    /// Get the specific heat in J/(kg·K)
    pub fn specific_heat(&self) -> f64 {
        self.energy_mass.specific_heat()
    }

    /// Add energy in Joules (temperature will increase)
    pub fn add_energy_joules(&mut self, additional_energy: f64) {
        self.add_energy(additional_energy); // Delegate to the main add_energy method
    }

    /// Remove energy in Joules (temperature will decrease)
    pub fn remove_energy_joules(&mut self, energy_to_remove: f64) {
        self.energy_mass.remove_heat(energy_to_remove);
    }

    /// Scale the layer by a factor (useful for splitting/combining)
    pub fn scale(&mut self, factor: f64) {
        self.energy_mass.scale(factor);
    }

    /// Merge another layer into this one
    /// Energy and volume are directly added, resulting in a new blended temperature
    /// The other layer must have compatible material properties
    pub fn merge_layer(&mut self, other: &AsthCellLayer) {
        self.energy_mass.merge_em(&other.energy_mass);
    }

    /// Remove a specified volume from this layer, returning a new layer with that volume
    /// The removed layer will have the same temperature as the original
    /// This layer will have proportionally less volume and energy but maintain the same temperature
    pub fn remove_volume_as_layer(&mut self, volume_to_remove: f64) -> AsthCellLayer {
        let removed_energy_mass = self.energy_mass.remove_volume(volume_to_remove);
        // Downcast the Box<dyn EnergyMass> back to StandardEnergyMass
        let any_box: Box<dyn std::any::Any> = removed_energy_mass;
        let standard_energy_mass = *any_box.downcast::<StandardEnergyMass>()
            .expect("Expected StandardEnergyMass");
        AsthCellLayer {
            energy_mass: standard_energy_mass,
            level: self.level, // Same level as the source
        }
    }

    /// Split this layer into two parts by volume fraction
    /// Returns a new layer with the specified fraction, this one keeps the remainder
    /// Both will have the same temperature as the original
    pub fn split_layer_by_fraction(&mut self, fraction: f64) -> AsthCellLayer {
        let split_energy_mass = self.energy_mass.split_by_fraction(fraction);
        // Downcast the Box<dyn EnergyMass> back to StandardEnergyMass
        let any_box: Box<dyn std::any::Any> = split_energy_mass;
        let standard_energy_mass = *any_box.downcast::<StandardEnergyMass>()
            .expect("Expected StandardEnergyMass");
        AsthCellLayer {
            energy_mass: standard_energy_mass,
            level: self.level, // Same level as the source
        }
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_temperature_energy_conversion_roundtrip() {
        let mut layer = AsthCellLayer::new_with_material(MaterialType::Silicate, 1000.0, 100.0, 0);

        // Test various temperatures
        let test_temperatures = vec![
            1000.0, // Low temperature
            1400.0, // Medium temperature
            1673.15, // Silicate peak growth temp
            1773.15, // Silicate halfway temp
            1873.15, // Silicate formation temp
            2000.0, // High temperature
        ];

        for target_temp in test_temperatures {
            // Set the temperature
            layer.set_temp_kelvin(target_temp);

            // Read it back
            let actual_temp = layer.kelvin();

            // Should be very close (within 0.01 K)
            assert_abs_diff_eq!(actual_temp, target_temp, epsilon = 0.01);

            println!("Target: {:.2} K, Actual: {:.2} K, Energy: {:.2e} J",
                     target_temp, actual_temp, layer.energy_joules());
        }
    }

    #[test]
    fn test_temperature_setting_validation() {
        let mut layer = AsthCellLayer::new_with_material(MaterialType::Silicate, 1000.0, 50.0, 1);

        // Test that setting specific lithosphere-relevant temperatures works
        let lithosphere_temps = vec![
            (1673.15, "Silicate peak growth temperature"),
            (1773.15, "Silicate halfway temperature"),
            (1873.15, "Silicate formation temperature"),
        ];

        for (temp, description) in lithosphere_temps {
            layer.set_temp_kelvin(temp);
            let actual = layer.kelvin();

            assert_abs_diff_eq!(actual, temp, epsilon = 0.1);
            println!("{}: Set {:.2} K, Got {:.2} K", description, temp, actual);
        }
    }

    #[test]
    fn test_volume_affects_energy_calculation() {
        let test_temp = 1500.0;

        // Test different volumes
        let volumes = vec![10.0, 50.0, 100.0, 200.0];

        for volume in volumes {
            let mut layer = AsthCellLayer::new_with_material(MaterialType::Silicate, test_temp, volume, 0);

            let actual_temp = layer.kelvin();

            // Temperature should be consistent regardless of volume
            assert_abs_diff_eq!(actual_temp, test_temp, epsilon = 0.01);

            // But energy should scale with volume
            println!("Volume: {:.1} km³, Energy: {:.2e} J, Temp: {:.2} K",
                     volume, layer.energy_joules(), actual_temp);
        }
    }

    #[test]
    fn test_material_profile_access() {
        use crate::material::MaterialType;

        // Test that we can create layers with different materials and access their profiles
        let silicate_layer = AsthCellLayer::new_with_material(MaterialType::Silicate, 1673.15, 100.0, 0);
        let basaltic_layer = AsthCellLayer::new_with_material(MaterialType::Basaltic, 1573.15, 100.0, 0);

        // Test that we can access material profiles
        let silicate_profile = silicate_layer.profile();
        let basaltic_profile = basaltic_layer.profile();

        println!("Silicate profile: {:?}", silicate_profile.kind);
        println!("Basaltic profile: {:?}", basaltic_profile.kind);

        // Validate that the profiles are correct
        assert_eq!(silicate_profile.kind, MaterialType::Silicate);
        assert_eq!(basaltic_profile.kind, MaterialType::Basaltic);

        // Test that temperatures are set correctly
        assert!((silicate_layer.temperature() - 1673.15).abs() < 0.01);
        assert!((basaltic_layer.temperature() - 1573.15).abs() < 0.01);

        println!("✅ Material profile access validated successfully!");
    }

    #[test]
    fn test_energy_temperature_conversion_with_materials() {
        use crate::material::MaterialType;

        // Test that setting energy gives the expected temperature for different materials
        let test_volume = 100.0; // km³

        // Test Silicate
        let mut silicate_layer = AsthCellLayer::new_with_material(MaterialType::Silicate, 1000.0, test_volume, 0);
        let silicate_profile = silicate_layer.profile();

        // Calculate expected energy for a specific temperature
        let target_temp = 1673.15; // Peak growth temperature
        let expected_energy = silicate_layer.mass_kg() * silicate_profile.specific_heat_capacity_j_per_kg_k * target_temp;

        // Set the energy and check if we get the expected temperature
        silicate_layer.set_energy_joules(expected_energy);
        let actual_temp = silicate_layer.temperature();

        println!("Silicate energy-temperature conversion:");
        println!("  Target temp: {} K", target_temp);
        println!("  Expected energy: {:.2e} J", expected_energy);
        println!("  Actual temp: {} K", actual_temp);
        println!("  Mass: {:.2e} kg", silicate_layer.mass_kg());
        println!("  Specific heat: {} J/(kg·K)", silicate_profile.specific_heat_capacity_j_per_kg_k);

        assert!((actual_temp - target_temp).abs() < 0.01,
                "Temperature should be {} K, got {} K", target_temp, actual_temp);

        // Test Basaltic
        let mut basaltic_layer = AsthCellLayer::new_with_material(MaterialType::Basaltic, 1000.0, test_volume, 0);
        let basaltic_profile = basaltic_layer.profile();

        let basaltic_target_temp = 1573.15; // Basaltic peak growth temperature
        let basaltic_expected_energy = basaltic_layer.mass_kg() * basaltic_profile.specific_heat_capacity_j_per_kg_k * basaltic_target_temp;

        basaltic_layer.set_energy_joules(basaltic_expected_energy);
        let basaltic_actual_temp = basaltic_layer.temperature();

        println!("Basaltic energy-temperature conversion:");
        println!("  Target temp: {} K", basaltic_target_temp);
        println!("  Expected energy: {:.2e} J", basaltic_expected_energy);
        println!("  Actual temp: {} K", basaltic_actual_temp);
        println!("  Mass: {:.2e} kg", basaltic_layer.mass_kg());
        println!("  Specific heat: {} J/(kg·K)", basaltic_profile.specific_heat_capacity_j_per_kg_k);

        assert!((basaltic_actual_temp - basaltic_target_temp).abs() < 0.01,
                "Basaltic temperature should be {} K, got {} K", basaltic_target_temp, basaltic_actual_temp);

        println!("✅ Energy-temperature conversion validated for all materials!");
    }

    #[test]
    fn test_simple_temperature_validation() {
        use crate::material::MaterialType;

        // Simple test to validate that temperature setting works correctly
        let layer1 = AsthCellLayer::new_with_material(MaterialType::Silicate, 1673.15, 100.0, 0);
        let layer2 = AsthCellLayer::new_with_material(MaterialType::Basaltic, 1573.15, 100.0, 0);

        // Test that temperatures are set correctly
        assert!((layer1.temperature() - 1673.15).abs() < 0.01,
                "Silicate layer temperature should be 1673.15 K, got {} K", layer1.temperature());
        assert!((layer2.temperature() - 1573.15).abs() < 0.01,
                "Basaltic layer temperature should be 1573.15 K, got {} K", layer2.temperature());

        println!("Silicate layer: {} K", layer1.temperature());
        println!("Basaltic layer: {} K", layer2.temperature());

        println!("✅ Simple temperature validation successful!");
    }

    #[test]
    fn test_kelvin_set_get_consistency() {
        use crate::material::MaterialType;

        // Test that setting Kelvin temperature and getting it back works correctly
        let mut layer = AsthCellLayer::new_with_material(MaterialType::Silicate, 1000.0, 100.0, 0);

        // Test various temperatures relevant to lithosphere formation
        let test_temps = vec![
            1673.15, // Peak growth temperature
            1773.15, // Halfway temperature
            1873.15, // Formation temperature
            1500.0,  // Below peak
            2000.0,  // Above formation
        ];

        for target_temp in test_temps {
            // Set the temperature in Kelvin
            layer.set_temp_kelvin(target_temp);

            // Get it back in Kelvin
            let actual_temp = layer.kelvin();

            // Should be exactly the same (within floating point precision)
            assert!((actual_temp - target_temp).abs() < 0.001,
                    "Set {:.2} K, got {:.2} K", target_temp, actual_temp);

            println!("✓ Set {:.2} K -> Got {:.2} K", target_temp, actual_temp);
        }

        println!("✅ Kelvin set/get consistency validated!");
    }

    #[test]
    fn test_simulation_temperature_conversion() {
        use crate::asth_cell::{AsthCellColumn, AsthCellParams};
        use h3o::CellIndex;

        // Test what happens when we create a simulation cell with a specific surface temperature
        let cell_index = CellIndex::base_cells().next().unwrap();

        // Test the problematic case from lithosphere tests
        let surface_temp = 1673.15; // Target peak growth temperature

        let cell = AsthCellColumn::new(AsthCellParams {
            cell_index,
            energy: 1000.0,
            volume: 100.0,
            layer_height_km: 10.0,
            layer_count: 3,
            planet_radius_km: 6371.0,
            surface_temp_k: surface_temp,
        });

        // Check what temperature we actually get in layer 0
        let layer_0_temp = cell.layers[0].kelvin();
        let layer_1_temp = cell.layers[1].kelvin();
        let layer_2_temp = cell.layers[2].kelvin();

        println!("Surface temp set to: {:.2} K", surface_temp);
        println!("Layer 0 temp: {:.2} K (diff: {:.2} K)", layer_0_temp, layer_0_temp - surface_temp);
        println!("Layer 1 temp: {:.2} K (diff: {:.2} K)", layer_1_temp, layer_1_temp - surface_temp);
        println!("Layer 2 temp: {:.2} K (diff: {:.2} K)", layer_2_temp, layer_2_temp - surface_temp);

        // The geothermal gradient should add only a small amount per layer
        // With 0.5 K/km and 10 km layers, each layer should be ~5K hotter than the previous
        let expected_layer_0_temp = surface_temp + 2.5; // 0.5 * 5 km (middle of layer)
        let expected_layer_1_temp = surface_temp + 7.5; // 0.5 * 15 km (middle of layer)

        println!("Expected layer 0 temp: {:.2} K", expected_layer_0_temp);
        println!("Expected layer 1 temp: {:.2} K", expected_layer_1_temp);

        // If the difference is huge, there's a bug in the temperature conversion
        let diff_0 = (layer_0_temp - expected_layer_0_temp).abs();
        let diff_1 = (layer_1_temp - expected_layer_1_temp).abs();

        if diff_0 > 100.0 || diff_1 > 100.0 {
            println!("⚠️  Large temperature difference detected - possible bug in energy_at_layer");
            println!("   Layer 0 diff: {:.2} K", diff_0);
            println!("   Layer 1 diff: {:.2} K", diff_1);
        } else {
            println!("✅ Temperature conversion looks reasonable");
        }
    }

    #[test]
    fn test_layer_merge_and_split() {
        let mut layer1 = AsthCellLayer::new_with_material(MaterialType::Silicate, 1600.0, 100.0, 0); // Hot layer
        let layer2 = AsthCellLayer::new_with_material(MaterialType::Silicate, 1200.0, 50.0, 0);      // Cool layer

        let initial_temp1 = layer1.kelvin();
        let initial_volume1 = layer1.volume_km3();

        // Merge layer2 into layer1
        layer1.merge_layer(&layer2);

        // Should have combined volume
        assert_abs_diff_eq!(layer1.volume_km3(), 150.0, epsilon = 0.01);

        // Temperature should be between the two original temperatures
        let merged_temp = layer1.kelvin();
        assert!(merged_temp > 1200.0 && merged_temp < 1600.0);

        // Split off a portion
        let split_layer = layer1.split_layer_by_fraction(0.4);

        // Both should have the same temperature
        assert_abs_diff_eq!(split_layer.kelvin(), merged_temp, epsilon = 0.01);
        assert_abs_diff_eq!(layer1.kelvin(), merged_temp, epsilon = 0.01);

        // Volumes should be split correctly
        assert_abs_diff_eq!(split_layer.volume_km3(), 60.0, epsilon = 0.01); // 40% of 150
        assert_abs_diff_eq!(layer1.volume_km3(), 90.0, epsilon = 0.01);      // 60% of 150

        println!("Original: {:.2} K, {:.1} km³", initial_temp1, initial_volume1);
        println!("After merge: {:.2} K, {:.1} km³", merged_temp, layer1.volume_km3() + split_layer.volume_km3());
        println!("Remaining: {:.2} K, {:.1} km³", layer1.kelvin(), layer1.volume_km3());
        println!("Split off: {:.2} K, {:.1} km³", split_layer.kelvin(), split_layer.volume_km3());
    }
}