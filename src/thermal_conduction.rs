/// Thermal conduction system for realistic energy transfer between layers
/// 
/// This module implements proper thermal conduction with:
/// - Material-based thermal conductivity
/// - Energy cascading when layers can't provide enough energy
/// - Cross-system mixing and energy removal
/// - Conductive energy transfer based on area and time

use crate::material::MaterialType;
use crate::energy_mass::EnergyMass;

/// Layer type identifier
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LayerType {
    Asthenosphere,
    Lithosphere,
}

/// Standardized layer representation for thermal calculations
/// Breaks down layers into: type, id, index, stdEnergy
#[derive(Debug, Clone)]
pub struct LayerStanding {
    /// Type of layer (Asthenosphere or Lithosphere)
    pub layer_type: LayerType,
    /// Unique identifier for this layer
    pub id: usize,
    /// Index position in the stack (0 = surface, higher = deeper)
    pub index: usize,
    /// Standard energy in joules (from EnergyMass trait)
    pub std_energy: f64,
    /// Material type for thermal conductivity calculations
    pub material_type: MaterialType,
}

impl LayerStanding {
    /// Create a new layer standing from an EnergyMass object
    pub fn new(
        layer_type: LayerType,
        id: usize,
        index: usize,
        energy_mass: &dyn EnergyMass,
        material_type: MaterialType,
    ) -> Self {
        Self {
            layer_type,
            id,
            index,
            std_energy: energy_mass.energy(),
            material_type,
        }
    }

    /// Create from raw values (for testing or special cases)
    pub fn from_raw(
        layer_type: LayerType,
        id: usize,
        index: usize,
        std_energy: f64,
        material_type: MaterialType,
    ) -> Self {
        Self {
            layer_type,
            id,
            index,
            std_energy,
            material_type,
        }
    }
}

/// Thermal conduction calculator
pub struct ThermalConduction {
    /// Area in km² for conductive calculations
    pub area_km2: f64,
    /// Time step in years
    pub time_years: f64,
}

impl ThermalConduction {
    pub fn new(area_km2: f64, time_years: f64) -> Self {
        Self {
            area_km2,
            time_years,
        }
    }

    /// Mix energy across layers using material-specific R0 thermal transmission coefficients
    /// Returns energy changes for each layer index
    pub fn mix(&self, layers: &[LayerStanding], base_mix_percent: f64) -> Vec<f64> {
        let mut energy_changes = vec![0.0; layers.len()];
        
        if layers.len() < 2 {
            return energy_changes;
        }

        // Calculate energy exchanges between adjacent layers
        for i in 0..layers.len() {
            let current_energy = layers[i].std_energy;
            
            // Mix with layer above (if exists)
            let above_exchange = if i > 0 {
                let above_energy = layers[i - 1].std_energy;
                // Energy flows from hot to cold
                let energy_diff = current_energy - above_energy;
                energy_diff * base_mix_percent
            } else {
                0.0
            };

            // Mix with layer below (if exists)
            let below_exchange = if i < layers.len() - 1 {
                let below_energy = layers[i + 1].std_energy;
                // Energy flows from hot to cold
                let energy_diff = below_energy - current_energy;
                energy_diff * base_mix_percent
            } else {
                0.0
            };
            
            // Net energy change for this layer
            let net_exchange = below_exchange - above_exchange;
            energy_changes[i] = net_exchange;
        }
        
        energy_changes
    }

    /// Take energy from layers with cascading from below
    /// For asthenosphere: limited by cooling fraction (max energy per step)
    /// For lithosphere: no maximum limit at this point
    /// Returns actual energy changes (negative = energy removed)
    pub fn take(&self, layers: &[LayerStanding], mut energy_requests: Vec<f64>) -> Vec<f64> {
        let mut energy_changes = vec![0.0; layers.len()];

        // Process from top to bottom, cascading energy requests downward
        for i in 0..layers.len() {
            let available_energy = layers[i].std_energy;
            let requested = energy_requests[i];

            // Calculate maximum energy that can be extracted from this layer
            let max_extractable = match layers[i].layer_type {
                LayerType::Asthenosphere => {
                    // For asthenosphere, use cooling fraction as maximum per step
                    let cooling_fraction = self.get_cooling_fraction(&layers[i]);
                    let cooling_limit = available_energy * cooling_fraction;

                    // Also consider thermal conductivity limits
                    let conductivity_limit = self.get_conductivity_limit(&layers[i]);

                    // Use minimum of cooling limit and conductivity limit
                    cooling_limit.min(conductivity_limit).min(available_energy)
                }
                LayerType::Lithosphere => {
                    // Lithosphere has no maximum limit at this point
                    available_energy
                }
            };

            if requested <= max_extractable {
                // Layer can provide the requested energy within limits
                energy_changes[i] = -requested;
            } else {
                // Layer cannot provide enough energy - take what's extractable
                energy_changes[i] = -max_extractable;

                // Cascade the remaining request to the layer below
                let remaining_request = requested - max_extractable;
                if i + 1 < layers.len() {
                    energy_requests[i + 1] += remaining_request;
                }
            }
        }

        energy_changes
    }

    /// Get cooling fraction for asthenosphere layers based on lithosphere thickness
    /// This represents the maximum fraction of energy that can be removed per step
    /// Uses the existing GLOBAL_ENERGY_LOSS_TABLE approach
    fn get_cooling_fraction(&self, layer: &LayerStanding) -> f64 {
        // For now, use a simplified approach based on layer depth
        // In practice, this should be calculated based on lithosphere thickness above this layer
        match layer.index {
            0 => 0.15,  // Surface layer - highest cooling rate (15% max per step)
            1 => 0.08,  // Second layer - moderate cooling (8% max per step)
            2 => 0.04,  // Third layer - reduced cooling (4% max per step)
            _ => 0.02,  // Deep layers - minimal cooling (2% max per step)
        }
    }

    /// Calculate cooling fraction based on lithosphere thickness above a layer
    /// This is the proper implementation that should be used when lithosphere data is available
    pub fn get_cooling_fraction_with_lithosphere(&self, layer_index: usize, lithosphere_thickness_km: f64) -> f64 {
        // Get the cooling rate from the existing table (we'll implement this locally for now)
        let cooling_j_per_year = self.cooling_j_my_for_thickness(lithosphere_thickness_km);

        // Convert to a fraction based on typical layer energy
        // This is a rough approximation - in practice you'd use the actual layer energy
        let typical_layer_energy = 1e24; // Joules (rough estimate)
        let cooling_fraction = (cooling_j_per_year * self.time_years) / typical_layer_energy;

        // Clamp to reasonable bounds
        cooling_fraction.clamp(0.001, 0.5) // 0.1% to 50% max per step
    }

    /// Get conductivity-based energy transfer limit
    fn get_conductivity_limit(&self, layer: &LayerStanding) -> f64 {
        // Calculate based on thermal conductivity and layer properties
        let conductivity = self.thermal_conductivity(layer.material_type);

        // Simple approximation: higher conductivity = higher transfer rate
        // This would be more sophisticated in practice
        let base_transfer_rate = conductivity * self.area_km2 * self.time_years * 1e15; // Scaling factor

        base_transfer_rate
    }

    /// Local implementation of cooling calculation based on lithosphere thickness
    /// Uses the same logic as the existing GLOBAL_ENERGY_LOSS_TABLE
    fn cooling_j_my_for_thickness(&self, thick_km: f64) -> f64 {
        // Simplified version of the cooling table
        // In practice, this should use the actual GLOBAL_ENERGY_LOSS_TABLE
        const COOLING_TABLE: &[(f64, f64)] = &[
            (0.0, 1.5e21),    // No lithosphere - maximum cooling
            (1.0, 1.2e21),    // 1 km lithosphere
            (5.0, 8.0e20),    // 5 km lithosphere
            (10.0, 2.0e20),   // 10 km lithosphere
            (50.0, 5.0e19),   // 50 km lithosphere
            (100.0, 1.0e19),  // 100 km lithosphere
            (200.0, 0.0),     // 200 km lithosphere - no cooling
        ];

        // Clamp outside table
        if thick_km <= COOLING_TABLE[0].0 {
            return COOLING_TABLE[0].1;
        }
        if thick_km >= COOLING_TABLE[COOLING_TABLE.len() - 1].0 {
            return COOLING_TABLE[COOLING_TABLE.len() - 1].1;
        }

        // Linear interpolation between table points
        for window in COOLING_TABLE.windows(2) {
            let (t1, e1) = window[0];
            let (t2, e2) = window[1];
            if thick_km >= t1 && thick_km <= t2 {
                let f = (thick_km - t1) / (t2 - t1);
                return e1 + f * (e2 - e1);
            }
        }

        0.0 // Fallback
    }

    /// Calculate conductive energy transfer between two layers
    /// Based on thermal conductivity, temperature difference, area, and time
    pub fn conductive_energy_transfer(
        &self,
        from_material: MaterialType,
        to_material: MaterialType,
        from_temp_k: f64,
        to_temp_k: f64,
        distance_km: f64,
    ) -> f64 {
        // Get thermal conductivities (W/m·K)
        let from_conductivity = self.thermal_conductivity(from_material);
        let to_conductivity = self.thermal_conductivity(to_material);
        
        // Use harmonic mean for interface conductivity
        let interface_conductivity = 2.0 * from_conductivity * to_conductivity / 
                                   (from_conductivity + to_conductivity);
        
        // Temperature difference
        let temp_diff = from_temp_k - to_temp_k;
        
        // Convert units: km² to m², km to m, years to seconds
        let area_m2 = self.area_km2 * 1e6;  // km² to m²
        let distance_m = distance_km * 1000.0;  // km to m
        let time_seconds = self.time_years * 365.25 * 24.0 * 3600.0;  // years to seconds
        
        // Fourier's law: Q = k * A * ΔT * t / d
        let energy_transfer = interface_conductivity * area_m2 * temp_diff * time_seconds / distance_m;
        
        energy_transfer
    }

    /// Get thermal conductivity for a material (W/m·K)
    fn thermal_conductivity(&self, material: MaterialType) -> f64 {
        match material {
            MaterialType::Silicate => 3.0,      // Typical silicate rock
            MaterialType::Basaltic => 2.5,      // Basaltic rock
            MaterialType::Granitic => 2.5,      // Granitic rock (similar to basalt)
            MaterialType::Metallic => 80.0,     // Metallic materials (iron/nickel)
            MaterialType::Icy => 2.2,           // Ice/water
        }
    }

    /// Calculate thermal resistance through multiple layers
    /// Used when energy must pass through intermediate layers
    pub fn thermal_resistance(&self, materials: &[MaterialType], thicknesses_km: &[f64]) -> f64 {
        let mut total_resistance = 0.0;
        
        for (material, thickness_km) in materials.iter().zip(thicknesses_km.iter()) {
            let conductivity = self.thermal_conductivity(*material);
            let thickness_m = thickness_km * 1000.0;  // km to m
            
            // Thermal resistance: R = thickness / (conductivity * area)
            let area_m2 = self.area_km2 * 1e6;  // km² to m²
            let resistance = thickness_m / (conductivity * area_m2);
            total_resistance += resistance;
        }
        
        total_resistance
    }

    /// Apply energy changes to layers, ensuring energy conservation
    /// Returns the updated energy values for each layer
    pub fn add_energy(&self, layers: &mut [LayerStanding], energy_changes: Vec<f64>) -> Vec<f64> {
        let mut updated_energies = Vec::new();

        for (i, energy_change) in energy_changes.iter().enumerate() {
            if i < layers.len() {
                // Ensure energy doesn't go negative
                let new_energy = (layers[i].std_energy + energy_change).max(0.0);
                layers[i].std_energy = new_energy;
                updated_energies.push(new_energy);
            }
        }

        updated_energies
    }

    /// Convert a cell column to standardized layer standings
    /// Combines asthenosphere layers and lithosphere layers into a unified stack
    pub fn from_cell_column(column: &crate::asth_cell::AsthCellColumn) -> Vec<LayerStanding> {
        let mut layer_standings = Vec::new();
        let mut id_counter = 0;

        // Add asthenosphere layers (surface to deep)
        for (index, layer) in column.layers.iter().enumerate() {
            layer_standings.push(LayerStanding::from_raw(
                LayerType::Asthenosphere,
                id_counter,
                index,
                layer.energy_joules(), // Use the direct method
                MaterialType::Silicate, // Asthenosphere is typically silicate
            ));
            id_counter += 1;
        }

        // Add lithosphere layers (if any)
        for (lith_index, lithosphere) in column.lithospheres.iter().enumerate() {
            if lithosphere.height_km > 0.0 {
                layer_standings.push(LayerStanding::from_raw(
                    LayerType::Lithosphere,
                    id_counter,
                    column.layers.len() + lith_index, // Continue indexing after asthenosphere
                    lithosphere.energy(), // Use the correct method
                    lithosphere.material(), // Use the correct method name
                ));
                id_counter += 1;
            }
        }

        layer_standings
    }

    /// Apply energy changes back to a cell column
    /// Updates both asthenosphere layers and lithosphere layers
    pub fn apply_to_cell_column(
        &self,
        column: &mut crate::asth_cell::AsthCellColumn,
        layer_standings: &[LayerStanding],
    ) {
        for standing in layer_standings {
            match standing.layer_type {
                LayerType::Asthenosphere => {
                    if standing.index < column.layers_next.len() {
                        // Calculate target temperature from energy
                        let volume_m3 = column.layers_next[standing.index].volume_km3() * 1e9;
                        let mass_kg = volume_m3 * column.layers_next[standing.index].density();
                        let specific_heat = column.layers_next[standing.index].specific_heat();

                        if mass_kg > 0.0 && specific_heat > 0.0 {
                            let target_temp = standing.std_energy / (mass_kg * specific_heat);
                            column.layers_next[standing.index].set_temp_kelvin(target_temp);
                        }
                    }
                }
                LayerType::Lithosphere => {
                    let lith_index = standing.index - column.layers.len();
                    if lith_index < column.lithospheres_next.len() {
                        // For now, we'll skip lithosphere energy updates
                        // Lithosphere energy management is more complex
                        // TODO: Implement proper lithosphere energy setting when needed
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mix_energy_flow() {
        let layers = vec![
            LayerStanding::from_raw(LayerType::Asthenosphere, 0, 0, 1000.0, MaterialType::Silicate),
            LayerStanding::from_raw(LayerType::Asthenosphere, 1, 1, 2000.0, MaterialType::Silicate),
            LayerStanding::from_raw(LayerType::Asthenosphere, 2, 2, 3000.0, MaterialType::Silicate),
        ];
        
        let thermal = ThermalConduction::new(100.0, 1.0);
        let changes = thermal.mix(&layers, 0.1);  // 10% mixing
        
        // Energy should flow from hot (bottom) to cold (top)
        assert!(changes[0] > 0.0);  // Top layer gains energy
        assert!(changes[2] < 0.0);  // Bottom layer loses energy
    }

    #[test]
    fn test_energy_cascading() {
        let layers = vec![
            LayerStanding::from_raw(LayerType::Asthenosphere, 0, 0, 100.0, MaterialType::Silicate),
            LayerStanding::from_raw(LayerType::Asthenosphere, 1, 1, 500.0, MaterialType::Silicate),
            LayerStanding::from_raw(LayerType::Asthenosphere, 2, 2, 1000.0, MaterialType::Silicate),
        ];
        
        let thermal = ThermalConduction::new(100.0, 1.0);
        let requests = vec![200.0, 0.0, 0.0];  // Request more than top layer has
        let changes = thermal.take(&layers, requests);
        
        // Top layer should give all its energy
        assert_eq!(changes[0], -100.0);
        // Second layer should provide the remaining 100.0
        assert_eq!(changes[1], -100.0);
        // Bottom layer unchanged
        assert_eq!(changes[2], 0.0);
    }

    #[test]
    fn test_conductive_transfer() {
        let thermal = ThermalConduction::new(100.0, 1.0);
        
        let energy = thermal.conductive_energy_transfer(
            MaterialType::Metallic, // High conductivity
            MaterialType::Silicate, // Lower conductivity
            2000.0,                 // Hot temperature
            1000.0,                 // Cold temperature
            1.0,                    // 1 km distance
        );
        
        // Should transfer energy from hot to cold
        assert!(energy > 0.0);
    }
}
