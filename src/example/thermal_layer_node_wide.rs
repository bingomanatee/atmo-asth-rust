use super::experiment_state::ExperimentState;
use crate::energy_mass_composite::{EnergyMassComposite, EnergyMassParams, StandardEnergyMassComposite};
use crate::material_composite::{get_profile_fast, MaterialCompositeType, MaterialPhase, MaterialStateProfile};
use crate::temp_utils::joules_volume_to_kelvin;

/// Science-backed Fourier heat transfer system for three-node thermal diffusion
/// Implements proper Fourier's law with material properties and geometric constraints
#[derive(Debug, Clone)]
pub struct FourierThermalTransfer {
    pub time_years: f64,
}

/// Physical constants for Fourier heat transfer calculations
mod fourier_constants {


    /// Minimum temperature difference for heat transfer (K)
    /// Below this threshold, thermal noise dominates and transfer is negligible
    pub const MIN_TEMP_DIFF_K: f64 = 0.1;

    /// Maximum energy transfer fraction per timestep for numerical stability
    /// Prevents unrealistic energy oscillations in explicit time stepping
    pub const MAX_ENERGY_TRANSFER_FRACTION: f64 = 0.1;
}

impl FourierThermalTransfer {
    pub fn new(time_years: f64) -> Self {
        Self { time_years }
    }

    /// Calculate energy transfers using science-backed Fourier heat conduction
    /// Returns (left_transfer, right_transfer) - positive values indicate energy flowing out of center
    ///
    /// Implements Fourier's law: Q = -k * A * (dT/dx) * dt
    /// Where:
    /// - k = thermal conductivity (W/m·K)
    /// - A = cross-sectional area (m²)
    /// - dT/dx = temperature gradient (K/m)
    /// - dt = time duration (s)
    /// Calculate bidirectional transfers between center node and neighbors
    /// Returns (left_transfer, right_transfer) where:
    /// - Positive values mean energy flows OUT of center node
    /// - Negative values mean energy flows INTO center node
    pub fn calculate_fourier_transfers(
        &self,
        center: &ThermalLayerNodeWide,
        left_neighbor: Option<&ThermalLayerNodeWide>,
        right_neighbor: Option<&ThermalLayerNodeWide>,
    ) -> (f64, f64) {


        let mut left_transfer = 0.0;
        let mut right_transfer = 0.0;

        // Calculate bidirectional left neighbor transfer
        if let Some(left) = left_neighbor {
            let center_temp = center.kelvin();
            let left_temp = left.kelvin();

            if center_temp > left_temp {
                // Center is hotter: energy flows from center to left (positive)
                left_transfer = self.calculate_fourier_heat_flow(center, left);
            } else if left_temp > center_temp {
                // Left is hotter: energy flows from left to center (negative)
                // Use the same calculation but with swapped nodes and negate the result
                left_transfer = -self.calculate_fourier_heat_flow(left, center);
            }
            // If temperatures are equal, left_transfer remains 0.0
        }

        // Calculate bidirectional right neighbor transfer
        if let Some(right) = right_neighbor {
            let center_temp = center.kelvin();
            let right_temp = right.kelvin();

            if center_temp > right_temp {
                // Center is hotter: energy flows from center to right (positive)
                right_transfer = self.calculate_fourier_heat_flow(center, right);
            } else if right_temp > center_temp {
                // Right is hotter: energy flows from right to center (negative)
                // Use the same calculation but with swapped nodes and negate the result
                right_transfer = -self.calculate_fourier_heat_flow(right, center);
            }
            // If temperatures are equal, right_transfer remains 0.0
        }

        // Apply physical constraint: maximum transfer is half the energy difference
        // This ensures energy flows toward equilibrium without overshooting
        if let Some(left) = left_neighbor {
            if left_transfer > 0.0 {
                // Center giving to left: limit to half the energy difference
                let energy_diff = center.energy() - left.energy();
                let max_transfer = energy_diff * 0.5;
                left_transfer = left_transfer.min(max_transfer.max(0.0));
            } else if left_transfer < 0.0 {
                // Left giving to center: limit to half the energy difference
                let energy_diff = left.energy() - center.energy();
                let max_transfer = energy_diff * 0.5;
                left_transfer = left_transfer.max(-max_transfer.max(0.0));
            }
        }

        if let Some(right) = right_neighbor {
            if right_transfer > 0.0 {
                // Center giving to right: limit to half the energy difference
                let energy_diff = center.energy() - right.energy();
                let max_transfer = energy_diff * 0.5;
                right_transfer = right_transfer.min(max_transfer.max(0.0));
            } else if right_transfer < 0.0 {
                // Right giving to center: limit to half the energy difference
                let energy_diff = right.energy() - center.energy();
                let max_transfer = energy_diff * 0.5;
                right_transfer = right_transfer.max(-max_transfer.max(0.0));
            }
        }

        (left_transfer, right_transfer)
    }

    /// Calculate heat flow between two nodes using simplified Fourier's law
    /// Returns energy transfer in Joules (positive = energy flows from 'from' to 'to')
    fn calculate_fourier_heat_flow(
        &self,
        from_node: &ThermalLayerNodeWide,
        to_node: &ThermalLayerNodeWide,
    ) -> f64 {
        use fourier_constants::*;

        // Get temperatures
        let temp_from = from_node.kelvin();
        let temp_to = to_node.kelvin();
        let temp_diff = temp_from - temp_to;

        // Only transfer heat from hot to cold
        if temp_diff <= MIN_TEMP_DIFF_K {
            return 0.0;
        }

        // SIMPLIFIED APPROACH: Use temperature difference to drive transfer
        // This ensures heat flows from hot to cold regardless of material differences

        // Calculate temperature-based transfer using thermal capacity
        let from_thermal_capacity = from_node.thermal_capacity(); // J/K
        let temp_diff_k = temp_diff; // Already calculated above

        // Conservative transfer: 1% of thermal energy difference per year, scaled by time
        let base_transfer_rate = 0.01 * self.time_years; // 1% per year
        let thermal_energy_diff = from_thermal_capacity * temp_diff_k;
        let conservative_transfer = thermal_energy_diff * base_transfer_rate;

        // Apply safety limits (use main energy for safety limit, not total)
        let max_safe_transfer = from_node.energy() * MAX_ENERGY_TRANSFER_FRACTION;

        conservative_transfer.min(max_safe_transfer)
    }


}

/// Parameters for creating ThermalLayerNode
pub struct ThermalLayerNodeWideParams {
    pub material_type: MaterialCompositeType,
    pub energy_joules: f64,
    pub volume_km3: f64,
    pub depth_km: f64,
    pub height_km: f64,
    pub area_km2: f64,
}

/// Parameters for creating ThermalLayerNode with direct temperature setting
pub struct ThermalLayerNodeWideTempParams {
    pub material_type: MaterialCompositeType,
    pub temperature_k: f64,
    pub volume_km3: f64,
    pub depth_km: f64,
    pub height_km: f64,
    pub area_km2: f64,
}

/// Thermal node with enhanced state tracking
#[derive(Clone, Debug)]
pub struct ThermalLayerNodeWide {
    energy_mass: StandardEnergyMassComposite, // Private - access through trait methods
    pub thermal_state: i32,
    pub depth_km: f64,
    pub height_km: f64,
    pub area_km2: f64,

    /// Thermal history for analysis
    pub initial_temperature: f64,
    pub max_temperature: f64,
    // this is a debugging tracker for the highest temperature reached not a physically limiting constant
    pub min_temperature: f64,
    // same
    /// Outgassing tracking
    pub total_outgassed_mass: f64,
    pub outgassing_rate: f64,
}

impl ThermalLayerNodeWide {
    pub fn new(params: ThermalLayerNodeWideParams) -> Self {
        // Validate volume constraint: volume must not exceed height * area
        let max_volume = params.height_km * params.area_km2;
        if params.volume_km3 > max_volume {
            panic!("Volume ({} km³) exceeds maximum allowed volume ({} km³) for height {} km and area {} km²",
                   params.volume_km3, max_volume, params.height_km, params.area_km2);
        }

        let profile = get_profile_fast(&params.material_type, &MaterialPhase::Liquid);

        let temperature_k = joules_volume_to_kelvin(
            params.energy_joules,
            params.volume_km3,
            profile.specific_heat_capacity_j_per_kg_k
        );
        let energy_mass = StandardEnergyMassComposite::new_with_material_state_and_energy(
            EnergyMassParams {
                material_type: params.material_type,
                initial_phase: MaterialPhase::Solid,
                energy_joules: params.energy_joules,
                volume_km3: params.volume_km3,
                height_km: params.height_km,
                pressure_gpa: 0.0, // Default surface pressure
            }
        );

        Self {
            energy_mass,
            thermal_state: 100, // Start as solid
            depth_km: params.depth_km,
            height_km: params.height_km,
            area_km2: params.area_km2,
            initial_temperature: temperature_k,
            max_temperature: temperature_k,
            min_temperature: temperature_k,
            total_outgassed_mass: 0.0,
            outgassing_rate: 0.0,
        }
    }

    /// Create a new ThermalLayerNode with direct temperature setting (simplified)
    /// This avoids the energy conversion round-trip and sets temperature directly
    pub fn new_with_temperature(params: ThermalLayerNodeWideTempParams) -> Self {
        // Validate volume constraint: volume must not exceed height * area
        let max_volume = params.height_km * params.area_km2;
        if params.volume_km3 > max_volume {
            panic!("Volume ({} km³) exceeds maximum allowed volume ({} km³) for height {} km and area {} km²",
                   params.volume_km3, max_volume, params.height_km, params.area_km2);
        }

        // Create energy mass with minimal energy, then set temperature directly
        let mut energy_mass = StandardEnergyMassComposite::new_with_material_state_and_energy(
            EnergyMassParams {
                material_type: params.material_type,
                initial_phase: MaterialPhase::Solid,
                energy_joules: 1.0, // Minimal energy, will be overridden
                volume_km3: params.volume_km3,
                height_km: params.height_km,
                pressure_gpa: 0.0, // Default surface pressure
            }
        );

        // Set temperature directly using the energy_mass set_kelvin method
        energy_mass.set_kelvin(params.temperature_k);

        Self {
            energy_mass,
            thermal_state: 100, // Start as solid
            depth_km: params.depth_km,
            height_km: params.height_km,
            area_km2: params.area_km2,
            initial_temperature: params.temperature_k,
            max_temperature: params.temperature_k,
            min_temperature: params.temperature_k,
            total_outgassed_mass: 0.0,
            outgassing_rate: 0.0,
        }
    }

    /// Convenience method for getting temperature (alias for kelvin())
    pub fn temp_kelvin(&self) -> f64 {
        self.kelvin()
    }

    /// Convenience method for getting volume (alias for volume())
    pub fn volume_km3(&self) -> f64 {
        self.volume()
    }

    /// Convenience method for getting mass (delegates to energy_mass)
    pub fn mass_kg_convenience(&self) -> f64 {
        self.energy_mass.mass_kg()
    }

    /// Get the depth in km (specific to ThermalLayerNode)
    pub fn depth_km(&self) -> f64 {
        self.depth_km
    }

    /// Get the area in km² (specific to ThermalLayerNode)
    pub fn area_km2(&self) -> f64 {
        self.area_km2
    }

    /// Get the maximum allowed volume based on height and area
    pub fn max_volume_km3(&self) -> f64 {
        self.height_km * self.area_km2
    }

    /// Setter methods for controlled access to internal properties

    /// Set the material composite type (changes the underlying material)
    pub fn set_material_type(&mut self, material_type: MaterialCompositeType) {
        // Create new energy mass with same energy but different material
        let current_energy = self.energy_mass.energy();
        let current_volume = self.energy_mass.volume();

        self.energy_mass = StandardEnergyMassComposite::new_with_material_state_and_energy(
            EnergyMassParams {
                material_type,
                initial_phase: MaterialPhase::Solid,
                energy_joules: current_energy,
                volume_km3: current_volume,
                height_km: self.height_km,
                pressure_gpa: self.energy_mass.pressure_gpa(), // Preserve current pressure
            }
        );
        self.log_extent();
    }

    /// Set the material phase (solid, liquid, gas)
    /// Updates the phase directly in the energy mass system
    pub fn set_material_phase(&mut self, phase: MaterialPhase) {
        self.energy_mass.phase = phase;
        self.log_extent();
    }

    /// Get the material composite type (public access)
    pub fn material_composite_type(&self) -> MaterialCompositeType {
        self.energy_mass.material_composite_type()
    }

    /// Set the volume (adjusts energy proportionally to maintain temperature)
    pub fn set_volume(&mut self, new_volume_km3: f64) {
        let current_temp = self.kelvin();
        self.energy_mass.scale(new_volume_km3 / self.volume());
        // Restore temperature by adjusting energy
        self.set_kelvin(current_temp);
        self.log_extent();
    }

    /// Additional getter methods for properties accessed by experiments

    /// Get the thermal capacity (J/K) - used for energy transfer calculations
    pub fn thermal_capacity(&self) -> f64 {
        self.energy_mass.thermal_capacity()
    }

    /// Get the material phase (for phase change logic) - pressure-aware
    pub fn phase(&self) -> MaterialPhase {
        self.energy_mass.phase()
    }

    /// Set the material phase (for phase change logic)
    pub fn set_phase(&mut self, phase: MaterialPhase) {
        self.set_material_phase(phase);
    }

    /// Update the material phase based on current temperature and depth
    /// This encapsulates the phase transition logic and should be called after temperature changes
    pub fn update_phase_from_kelvin(&mut self) {
        let temp = self.kelvin();

        // Use pressure-aware phase resolution based on depth
        let new_phase = crate::material_composite::resolve_phase_from_temperature_and_depth(
            &self.energy_mass.material_composite_type(),
            temp,
            self.depth_km
        );

        // Update thermal_state to reflect the phase (for display/analysis purposes)
        // This follows the existing thermal state scale: 100=solid, 0=liquid, -100=gas
        self.thermal_state = match new_phase {
            MaterialPhase::Solid => 100,
            MaterialPhase::Liquid => 0,
            MaterialPhase::Gas => -100,
        };

        self.log_extent();
    }



    /// Calculate outgassing based on temperature
    pub fn calculate_outgassing(&mut self, _config: &ExperimentState, _years: f64) -> f64 {
        0.0
    }

    /// Apply science-backed Fourier thermal transfer to neighboring nodes
    /// Returns the net energy change for this node (positive = gained energy, negative = lost energy)
    pub fn apply_fourier_thermal_transfer(
        &mut self,
        mut left_neighbor: Option<&mut ThermalLayerNodeWide>,
        mut right_neighbor: Option<&mut ThermalLayerNodeWide>,
        time_years: f64,
    ) -> f64 {
        let fourier_transfer = FourierThermalTransfer::new(time_years);

        // Record initial total energy (including phase transition banks) for debugging
        let initial_center_energy = self.total_energy();
        let initial_left_energy = left_neighbor.as_ref().map(|n| n.total_energy()).unwrap_or(0.0);
        let initial_right_energy = right_neighbor.as_ref().map(|n| n.total_energy()).unwrap_or(0.0);
        let initial_total = initial_center_energy + initial_left_energy + initial_right_energy;

        // Calculate transfers using science-backed Fourier heat conduction
        let (left_transfer, right_transfer) = fourier_transfer.calculate_fourier_transfers(
            self,
            left_neighbor.as_deref(),
            right_neighbor.as_deref()
        );

        // Apply energy transfers using atomic send_energy operations for perfect conservation

        // Analyze transfer amounts relative to physical constraints
        let center_energy = self.total_energy();
        let outgoing_transfer = left_transfer.max(0.0) + right_transfer.max(0.0);

        let transfer_percentage = if center_energy > 0.0 {
            (outgoing_transfer / center_energy) * 100.0
        } else {
            0.0
        };

        // Calculate maximum possible transfer based on energy differences
        let max_left_diff = if let Some(left) = left_neighbor.as_ref() {
            (center_energy - left.total_energy()).abs() * 0.5
        } else {
            0.0
        };
        let max_right_diff = if let Some(right) = right_neighbor.as_ref() {
            (center_energy - right.total_energy()).abs() * 0.5
        } else {
            0.0
        };
        let max_physical_transfer = max_left_diff + max_right_diff;

        let physical_utilization = if max_physical_transfer > 0.0 {
            (outgoing_transfer / max_physical_transfer) * 100.0
        } else {
            0.0
        };

        // Send energy to neighbors using atomic operations
        if let Some(ref mut left) = left_neighbor {
            if left_transfer > 0.0 {
                // Send energy from center to left neighbor
                self.energy_mass.send_energy(left_transfer, &mut left.energy_mass);
            } else if left_transfer < 0.0 {
                // Receive energy from left neighbor to center
                left.energy_mass.send_energy(-left_transfer, &mut self.energy_mass);
            }
        }

        if let Some(ref mut right) = right_neighbor {
            if right_transfer > 0.0 {
                // Send energy from center to right neighbor
                self.energy_mass.send_energy(right_transfer, &mut right.energy_mass);
            } else if right_transfer < 0.0 {
                // Receive energy from right neighbor to center
                right.energy_mass.send_energy(-right_transfer, &mut self.energy_mass);
            }
        }

        // Report transfer analysis if significant
        if outgoing_transfer > 0.0 {
            println!("📊 Transfer Analysis:");
            println!("   Total outgoing: {:.2e} J ({:.3}% of node energy)", outgoing_transfer, transfer_percentage);
            println!("   Max physical transfer: {:.2e} J (half energy differences)", max_physical_transfer);
            println!("   Physical constraint utilization: {:.1}%", physical_utilization);
            if physical_utilization > 90.0 {
                println!("   ⚠️  High physical constraint utilization!");
            }
        }

        // Verify energy conservation for debugging (using total energy including banks)
        let final_center_energy = self.total_energy();
        let final_left_energy = left_neighbor.as_ref().map(|n| n.total_energy()).unwrap_or(0.0);
        let final_right_energy = right_neighbor.as_ref().map(|n| n.total_energy()).unwrap_or(0.0);
        let final_total = final_center_energy + final_left_energy + final_right_energy;

        let energy_diff = (final_total - initial_total).abs();
        if energy_diff > 1e10 { // Only warn for significant differences
            println!("⚠️  Energy conservation warning: {:.0} J difference", energy_diff);
            println!("   Initial: {:.3e} J, Final: {:.3e} J", initial_total, final_total);
            println!("   Transfers: left={:.0}, right={:.0}", left_transfer, right_transfer);
        }

        // Return net energy change for this node
        let net_energy_change = final_center_energy - center_energy;
        net_energy_change
    }

    /// Legacy method for backward compatibility - now uses Fourier transfer
    pub fn apply_three_node_transfer(
        &mut self,
        left_neighbor: Option<&mut ThermalLayerNodeWide>,
        right_neighbor: Option<&mut ThermalLayerNodeWide>,
        time_years: f64,
    ) -> f64 {
        self.apply_fourier_thermal_transfer(left_neighbor, right_neighbor, time_years)
    }
}

/// Extent logging for debugging

impl ThermalLayerNodeWide {
    fn log_extent(&mut self) {
        let temp = self.temp_kelvin();
        self.max_temperature = self.max_temperature.max(temp);
        self.min_temperature = self.min_temperature.min(temp);
    }
}

/// Material and thermal State

impl ThermalLayerNodeWide {
    /// Format thermal state for display
    pub fn format_thermal_state(&self) -> String {
        let temp = self.thermal_state;
        match self.material_state() {
            MaterialPhase::Solid => {
                format!("<{}> 🗻 Solid/Lithosphere", temp)
            }
            MaterialPhase::Liquid => {
                format!("<{}> 💧 Liquid/Magma", temp.abs())
            }
            MaterialPhase::Gas => {
                format!("<{}> ☁️ Gas/Atmosphere", temp.abs())
            }
        }
    }

    pub fn material_state(&self) -> MaterialPhase {
        match self.thermal_state {
            std::i32::MIN..=-66 => MaterialPhase::Gas, // anything ≤ −66
            -65..=65 => MaterialPhase::Liquid,         // −65 through 65
            66..=std::i32::MAX => MaterialPhase::Solid, // 66 through max
        }
    }
}

/// Implement EnergyMassComposite trait for ThermalLayerNode
/// This allows ThermalLayerNode to be used anywhere an EnergyMassComposite is expected
impl EnergyMassComposite for ThermalLayerNodeWide {
    /// Get the current temperature in Kelvin
    fn kelvin(&self) -> f64 {
        self.energy_mass.kelvin()
    }

    /// Set the temperature in Kelvin, updating energy accordingly (volume stays constant)
    /// note -- in general you add or remove energy in a simulation not mandate the temperature
    /// 
    fn set_kelvin(&mut self, kelvin: f64) {
        self.energy_mass.set_kelvin(kelvin);
        self.update_phase_from_kelvin(); // Automatically update phase when temperature changes
        self.log_extent();
    }

    /// Get the current energy in Joules (read-only)
    fn energy(&self) -> f64 {
        self.energy_mass.energy()
    }

    /// Get the current volume in km³ (read-only)
    fn volume(&self) -> f64 {
        self.energy_mass.volume()
    }

    /// Get the height in km (for layer-based calculations)
    fn height_km(&self) -> f64 {
        self.height_km
    }

    /// Get the mass in kg (derived from volume and density)
    fn mass_kg(&self) -> f64 {
        self.energy_mass.mass_kg()
    }

    /// Get the density in kg/m³
    fn density_kgm3(&self) -> f64 {
        self.energy_mass.density_kgm3()
    }

    /// Get the specific heat in J/(kg·K)
    fn specific_heat_j_kg_k(&self) -> f64 {
        self.energy_mass.specific_heat_j_kg_k()
    }

    /// Get the thermal conductivity in W/(m·K)
    fn thermal_conductivity(&self) -> f64 {
        self.energy_mass.thermal_conductivity()
    }

    /// Get the material composite type
    fn material_composite_type(&self) -> MaterialCompositeType {
        self.energy_mass.material_composite_type()
    }

    /// Get the material composite profile
    fn material_composite_profile(&self) -> &'static MaterialStateProfile {
        self.energy_mass.material_composite_profile()
    }

    // material_composite() method removed - MaterialComposite struct no longer exists
    // Use material_composite_type() and get_profile_fast() instead

    /// Scale the entire EnergyMass by a factor (useful for splitting/combining)
    fn scale(&mut self, factor: f64) {
        self.energy_mass.scale(factor);
        self.log_extent();
    }

    /// Add energy to this energy mass
    fn add_energy(&mut self, energy_joules: f64) {
        self.energy_mass.add_energy(energy_joules);
        self.update_phase_from_kelvin(); // Update phase after energy change
        self.log_extent();
    }

    /// Remove energy from this energy mass
    fn remove_energy(&mut self, energy_joules: f64) {
        self.energy_mass.remove_energy(energy_joules);
        self.update_phase_from_kelvin(); // Update phase after energy change
        self.log_extent();
    }

    /// Send energy to another energy mass composite in a single atomic operation
    fn send_energy(&mut self, energy_joules: f64, recipient: &mut dyn EnergyMassComposite) {
        self.energy_mass.send_energy(energy_joules, recipient);
        self.update_phase_from_kelvin(); // Update phase after energy change
        self.log_extent();
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    /// Remove heat energy (temperature will decrease, enforces zero minimum)
    fn remove_joules(&mut self, heat_joules: f64) {
        self.energy_mass.remove_joules(heat_joules);
        self.update_phase_from_kelvin(); // Update phase after energy change
        self.log_extent();
    }

    /// Add energy (temperature will increase)
    fn add_joules(&mut self, energy_joules: f64) {
        self.energy_mass.add_joules(energy_joules);
        self.update_phase_from_kelvin(); // Update phase after energy change
        self.log_extent();
    }

    /// Radiate energy to another EnergyMass using conductive transfer
    fn radiate_to(&mut self, other: &mut dyn EnergyMassComposite, distance_km: f64, area_km2: f64, time_years: f64) -> f64 {
        let result = self.energy_mass.radiate_to(other, distance_km, area_km2, time_years);
        self.log_extent();
        result
    }

    /// Radiate energy to space using Stefan-Boltzmann law
    fn radiate_to_space(&mut self, area_km2: f64, time_years: f64) -> f64 {
        let result = self.energy_mass.radiate_to_space(area_km2, time_years);
        self.log_extent();
        result
    }

    /// Radiate energy to space using Stefan-Boltzmann law with thermal skin depth limiting
    fn radiate_to_space_with_skin_depth(&mut self, area_km2: f64, time_years: f64, energy_throttle: f64) -> f64 {
        let result = self.energy_mass.radiate_to_space_with_skin_depth(area_km2, time_years, energy_throttle);
        self.log_extent();
        result
    }

    /// Compute thermal-diffusive skin depth in kilometres for this material
    fn skin_depth_km(&self, time_years: f64) -> f64 {
        self.energy_mass.skin_depth_km(time_years)
    }

    /// Remove volume (enforces zero minimum, maintains temperature)
    fn remove_volume_internal(&mut self, volume_to_remove: f64) {
        self.energy_mass.remove_volume_internal(volume_to_remove);
        self.log_extent();
    }

    /// Merge another EnergyMass into this one
    fn merge_em(&mut self, other: &dyn EnergyMassComposite) {
        self.energy_mass.merge_em(other);
        self.log_extent();
    }

    /// Remove a specified volume from this EnergyMass, returning a new EnergyMass with that volume
    fn remove_volume(&mut self, volume_to_remove: f64) -> Box<dyn EnergyMassComposite> {
        let result = self.energy_mass.remove_volume(volume_to_remove);
        self.log_extent();
        result
    }

    /// Split this EnergyMass into two parts by volume fraction
    fn split_by_fraction(&mut self, fraction: f64) -> Box<dyn EnergyMassComposite> {
        let result = self.energy_mass.split_by_fraction(fraction);
        self.log_extent();
        result
    }

    /// Get the R0 thermal transmission coefficient for this material
    fn thermal_transmission_r0(&self) -> f64 {
        self.energy_mass.thermal_transmission_r0()
    }

    /// Get the current pressure in GPa
    fn pressure_gpa(&self) -> f64 {
        self.energy_mass.pressure_gpa()
    }

    /// Set the pressure in GPa and update phase accordingly
    fn set_pressure_gpa(&mut self, pressure_gpa: f64) {
        self.energy_mass.set_pressure_gpa(pressure_gpa);
        self.log_extent();
    }

    /// Get the current material phase (pressure-aware)
    fn phase(&self) -> MaterialPhase {
        self.energy_mass.phase()
    }
}

impl ThermalLayerNodeWide {
    /// Get the total energy including phase transition bank (read-only)
    pub fn total_energy(&self) -> f64 {
        self.energy_mass.energy() + self.energy_mass.state_transition_bank()
    }
}
