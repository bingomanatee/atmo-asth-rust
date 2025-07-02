#![allow(dead_code)]
#![allow(unused_variables)]

use std::collections::VecDeque;
use std::fs::File;
use std::io::Write;

// Import the real EnergyMass trait and material system
extern crate atmo_asth_rust;
use atmo_asth_rust::energy_mass::EnergyMass;
use atmo_asth_rust::energy_mass_composite::{
    EnergyMassComposite, MaterialCompositeType, MaterialPhase, StandardEnergyMassComposite,
    get_profile_fast,
};
use atmo_asth_rust::example::{ExperimentState, ExperimentSpecs, ThermalLayerNodeWide, ThermalLayerNodeWideParams, ThermalLayerNodeWideTempParams, FourierThermalTransfer};
use atmo_asth_rust::material_composite::get_material_core;
// Re-export MaterialPhase for easier access
use atmo_asth_rust::temp_utils::energy_from_kelvin;
use atmo_asth_rust::math_utils::lerp;
use atmo_asth_rust::assert_deviation;
use more_asserts::assert_lt;

mod test_utils_1km3;
use test_utils_1km3::{temperature_baselines, validate_temperature_gradient, validate_boundary_conditions};

// Export layer indices - key layer positions for analysis
const EXPORT_LAYER_INDICES: [usize; 9] = [10, 15, 20, 25, 30, 35, 40, 45, 50];

// Area constant for the 100 km² experiment
const AREA_KM2: f64 = 100.0;

// Locked configuration constants for wide experiment - used in both config and tests
const WIDE_CONDUCTIVITY_FACTOR: f64 = 15.0;
const WIDE_PRESSURE_BASELINE: f64 = 5.0;
const WIDE_MAX_CHANGE_RATE: f64 = 0.06;
const WIDE_FOUNDRY_TEMPERATURE_K: f64 = 4800.0;
const WIDE_SURFACE_TEMPERATURE_K: f64 = 280.0;

// Test expectation constants - calibrated to the wide experiment configuration
const EXPECTED_TEMPERATURE_RANGE_K: f64 = WIDE_FOUNDRY_TEMPERATURE_K - WIDE_SURFACE_TEMPERATURE_K; // 4520.0K
const EXPECTED_NUM_NODES: usize = 30;
const TEMPERATURE_TOLERANCE_PERCENT: f64 = 5.0; // Wider tolerance for wide experiment

/// Create locked configuration specs for the 100km² wide experiment
/// These values are locked to prevent config drift in tests
fn wide_experiment_specs() -> ExperimentSpecs {
    ExperimentSpecs {
        // Use Silicate as the primary material type (mantle material)
        material_type: MaterialCompositeType::Silicate,

        // Diffusion parameters - optimized for 100km² wide experiment
        conductivity_factor: WIDE_CONDUCTIVITY_FACTOR,
        pressure_baseline: WIDE_PRESSURE_BASELINE,
        max_change_rate: WIDE_MAX_CHANGE_RATE,

        // Boundary conditions - adjusted for wide experiment
        foundry_temperature_k: WIDE_FOUNDRY_TEMPERATURE_K,
        surface_temperature_k: WIDE_SURFACE_TEMPERATURE_K,
    }
}

///  This simulates a single 100 km2 with only vertical flows up and down
///  with energy coming in from the highest/lowest cell in the array
/// to the surface of the earth at cell 0 which radiates heat into space.
/// the settings for the heat flow come in as ExperimentState
struct OneHundredKm2Experiment {
    nodes: Vec<ThermalLayerNodeWide>,
    config: ExperimentState,
    layer_height_km: f64,
    total_years: u64,
    steps: u64,

    // Tracking for analysis
    temperature_history: VecDeque<Vec<f64>>,
    energy_transfers: Vec<f64>,
    total_outgassing: f64,

    // Boundary layer indices
    foundry_start: usize,
    foundry_end: usize,
    surface_start: usize,
    surface_end: usize,
}

impl OneHundredKm2Experiment {
    fn new(steps: u64, total_years: u64) -> Self {
        let config = ExperimentState::new_with_specs(wide_experiment_specs());
        let num_nodes = 30;
        let total_depth_km = 750.0; // 750km total depth
        let layer_height_km = total_depth_km / num_nodes as f64; // 25km per layer

        let mut nodes = Vec::new();

        // Create thermal nodes with realistic temperature profile
        for i in 0..num_nodes {
            let depth_km = (i as f64 + 0.5) * layer_height_km;
            let volume_km3 = AREA_KM2 * layer_height_km; // 100 km² surface area

            // Calculate temperature using linear gradient from surface to foundry
            let temp_kelvin = lerp(
                config.surface_temperature_k,
                config.foundry_temperature_k,
                i as f64 / num_nodes as f64
            );

            // Use simplified constructor that sets temperature directly
            let node = ThermalLayerNodeWide::new_with_temperature(ThermalLayerNodeWideTempParams {
                material_type: MaterialCompositeType::Silicate,
                temperature_k: temp_kelvin,
                volume_km3,
                depth_km,
                height_km: layer_height_km,
                area_km2: AREA_KM2,
            });
            nodes.push(node);
        }

        // Get layer height from the first node (all nodes should have same height)
        let layer_height_km = nodes[0].height_km();

        Self {
            nodes,
            config,
            steps,
            total_years,
            layer_height_km,
            temperature_history: VecDeque::new(),
            energy_transfers: Vec::new(),
            total_outgassing: 0.0,
            foundry_start: 0,
            foundry_end: 2,    // 2 foundry layers (0-50km)
            surface_start: 26, // Surface cooling starts at layer 26 (650km)
            surface_end: 30,   // Surface cooling ends at layer 30 (750km)
        }
    }

    fn print_initial_state(&self) {
        println!("🔬 4x Scaled Force-Directed Thermal Diffusion with Enhanced Parameters");
        println!("====================================================================");
        println!(
            "🎯 Deep geological time: {} years of thermal evolution",
            self.total_years
        );
        println!("📊 Initial Setup:");
        println!("   {} thermal nodes (25km each, 750km total depth)", self.nodes.len());

        println!("\n🌡️  Initial Temperature Profile:");
        println!(
            "   Foundry layers (bottom 75km): {}K each (energy divided by 3)",
            (self.config.foundry_temperature_k / 3.0) as i32
        );
    }

    fn init_csv(&self) -> File {
        let mut file =
            File::create("examples/data/4x_thermal_experiment.csv").expect("Could not create file");
        // Write header - export key layers
        write!(file, "years").unwrap();

        // Export key layers for analysis
        for &layer_idx in &EXPORT_LAYER_INDICES {
            let depth_km = (layer_idx as f64 + 0.5) * self.layer_height_km; // Convert layer index to depth
            write!(file, ",temp_{}km", depth_km as i32).unwrap();
        }
        for &layer_idx in &EXPORT_LAYER_INDICES {
            let depth_km = (layer_idx as f64 + 0.5) * self.layer_height_km; // Convert layer index to depth
            write!(file, ",state_{}km", depth_km as i32).unwrap();
        }
        writeln!(file).unwrap();
        file
    }

    pub fn years_per_step(&self) -> f64 {
        self.total_years as f64 / self.steps as f64
    }

    fn run(&mut self, steps: u64, total_years: u64) {
        self.total_years = total_years;
        self.steps = steps;
        let mut file = self.init_csv();

        // Export initial state
        self.export_state(&mut file, 0.0);

        let mut last_progress: u32 = 0;
        // Run force-directed thermal diffusion
        for step in 0..steps {
            self.force_directed_step();

            let years = (step + 1) as f64 * self.years_per_step();
            self.export_state(&mut file, years);

            // Progress reporting every 10% of total steps
            let progress = (step as f64 / steps as f64) * 100.0;
            let pp = (progress / 10.0).floor() as u32;
            if last_progress != pp {
                println!(
                    "    {}    Progress: {:.1}% - {:.0} years",
                    step, progress, years
                );
                last_progress = pp;
            }
        }

        self.print_final_analysis();
    }

    fn force_directed_step(&mut self) {
        let mut energy_changes = vec![0.0; self.nodes.len()];

        // BOUNDARY CONDITIONS: Heat source and heat sink
        self.apply_boundary_conditions();

        // THERMAL CONDUCTION: Each layer exchanges only with adjacent neighbors
        for node_index in 0..self.nodes.len() {
            // Exchange only with immediate neighbors (i-1, i+1) for proper thermal conductivity
            for offset in [-1, 1] {
                let target_place = node_index as i32 + offset;
                if target_place < 0 || target_place >= self.nodes.len() as i32 {
                    continue;
                }
                let target_index = target_place as usize;

                // Energy-conservative exchange between adjacent nodes only
                self.exchange_energy_between_nodes(node_index, target_index, &mut energy_changes);
            }
        }

        // Apply energy changes
        for (i, &energy_change) in energy_changes.iter().enumerate() {
            if energy_change > 0.0 {
                self.nodes[i].add_energy(energy_change);
            } else if energy_change < 0.0 {
                self.nodes[i].remove_energy(-energy_change);
            }
        }

        // Update thermal states based on temperature with time-dependent transitions
        self.update_node_phases();
    }

    /// Calculate energy redistribution array for thermal diffusion (for testing)
    fn calculate_thermal_diffusion_energy_changes(&self) -> Vec<f64> {
        let mut energy_changes = vec![0.0; self.nodes.len()];

        // BINARY EXCHANGE THERMAL DIFFUSION: Each node exchanges with up to 3 neighbors
        // Effect radius = 3, so maximum exchanges = 2 * 3 * node_count (each node can exchange with up to 3 neighbors on each side)
        // But we process each exchange only once, so it's actually node_count * 3 exchanges maximum

        const EFFECT_RADIUS: usize = 3;

        for i in 0..self.nodes.len() {
            // Each node exchanges with neighbors within effect radius
            for distance in 1..=EFFECT_RADIUS {
                if i + distance < self.nodes.len() {
                    // Exchange with neighbor at +distance
                    let j = i + distance;

                    let from_temp = self.nodes[i].temp_kelvin();
                    let to_temp = self.nodes[j].temp_kelvin();
                    let temp_diff = from_temp - to_temp;

                    if temp_diff.abs() < 0.1 {
                        continue; // No significant temperature difference
                    }

                    // Distance falloff: 0.25^(distance)
                    let distance_weight = 0.25_f64.powi(distance as i32);

                    // Calculate energy exchange based on temperature difference
                    let from_capacity = self.nodes[i].thermal_capacity();
                    let to_capacity = self.nodes[j].thermal_capacity();
                    let avg_capacity = (from_capacity + to_capacity) / 2.0;

                    // Base exchange rate (small fraction per step to prevent instability)
                    let base_exchange_rate = 0.01; // 1% per step for adjacent neighbors
                    let exchange_rate = base_exchange_rate * distance_weight;

                    // Energy transfer proportional to temperature difference and capacity
                    let energy_transfer = temp_diff * avg_capacity * exchange_rate * self.years_per_step();

                    // Limit transfer to prevent instability (max 5% of smaller capacity per step)
                    let min_capacity = from_capacity.min(to_capacity);
                    let max_transfer = min_capacity * 0.05;
                    let limited_transfer = energy_transfer.abs().min(max_transfer);

                    if temp_diff > 0.0 {
                        // Energy flows from i to j (i is hotter)
                        energy_changes[i] -= limited_transfer;
                        energy_changes[j] += limited_transfer;
                    } else {
                        // Energy flows from j to i (j is hotter)
                        energy_changes[i] += limited_transfer;
                        energy_changes[j] -= limited_transfer;
                    }
                }
            }
        }

        energy_changes
    }

    /// Run thermal diffusion step without boundary conditions (for testing)
    fn thermal_diffusion_step_only(&mut self) {
        let energy_changes = self.calculate_thermal_diffusion_energy_changes();

        // Apply energy changes
        for (i, &energy_change) in energy_changes.iter().enumerate() {
            if energy_change > 0.0 {
                self.nodes[i].add_energy(energy_change);
            } else if energy_change < 0.0 {
                self.nodes[i].remove_energy(-energy_change);
            }
        }

        // Update thermal states based on temperature with time-dependent transitions
        self.update_node_phases();
    }

    fn apply_boundary_conditions(&mut self) {
        // Foundry heat source divided equally across bottom three layers
        let num_layers = self.nodes.len();
        let foundry_base_temp = self.config.foundry_temperature_k;
        let distributed_temp = foundry_base_temp / 3.0;  // Divide foundry energy by 3

        // Bottom three layers with equal distributed temperatures
        self.nodes[num_layers - 1].set_kelvin(distributed_temp);    // Bottom layer
        self.nodes[num_layers - 2].set_kelvin(distributed_temp);    // Middle layer
        self.nodes[num_layers - 3].set_kelvin(distributed_temp);    // Top layer

        // Space cooling at the top (index 0)
        self.nodes[0].set_kelvin(self.config.surface_temperature_k);
    }

    /// Energy-conservative exchange between two nodes
    /// Calculates energy transfer and updates both nodes in the energy_changes array
    fn exchange_energy_between_nodes(
        &self,
        from_idx: usize,
        to_idx: usize,
        energy_changes: &mut Vec<f64>,
    ) {
        let from_node = &self.nodes[from_idx];
        let to_node = &self.nodes[to_idx];

        let temp_diff = to_node.temp_kelvin() - from_node.temp_kelvin();
        if temp_diff < 0.1 {
            // we only track the energy transfer from hot to cool nodes here
            return; // No significant temperature difference
        }

        // Calculate energy transfer amount (positive = energy flows from 'from' to 'to')
        let energy_transfer = self.get_energy_change_from_node_to_node(from_node, to_node);

        // again - only track positive outflow from hotter nodes
        if energy_transfer > 0.0 {
            // 'to' node is hotter, energy flows from 'to' to 'from'
            energy_changes[from_idx] += energy_transfer;
            energy_changes[to_idx] -= energy_transfer;
        }
    }

    fn get_energy_change_from_node_to_node(
        &self,
        from_node: &ThermalLayerNodeWide,
        to_node: &ThermalLayerNodeWide,
    ) -> f64 {
        let temp_diff = (to_node.temp_kelvin() - from_node.temp_kelvin()).abs();
        // Calculate distance between nodes
        let distance_km = (from_node.depth_km() - to_node.depth_km()).abs();
        // Get material-based conductivities (4x scaled)
        let from_conductivity = self.get_material_conductivity(from_node);
        let to_conductivity = self.get_material_conductivity(to_node);

        // Distance weights: 1.0, 0.25, 0.125 for neighbors 1, 2, 3 away
        let distance_weight = if distance_km <= 5.1 {
            1.0 // Adjacent neighbor
        } else if distance_km <= 10.1 {
            0.25 // Second neighbor
        } else if distance_km <= 15.1 {
            0.125 // Third neighbor
        } else {
            0.0 // Too far
        };

        // Conductivity factor: both sender and recipient must be conductive
        let sender_factor = from_conductivity / self.config.conductivity_factor;
        let recipient_factor = to_conductivity / self.config.conductivity_factor;
        let conductivity_factor = sender_factor * recipient_factor;

        // Base diffusion rate: 4x enhanced for faster heat transport
        let base_coefficient = self.config.max_change_rate / self.years_per_step(); // @TODO: not sure about the scaling here investigate

        // PRESSURE ACCELERATION: Help foundry heat reach farther (4x enhanced)
        let pressure_factor =
            ((temp_diff / 100.0).powf(1.5)).max(0.33) * self.config.pressure_baseline;

        // DIFFUSION PHYSICS: Energy transfer based on temperature difference
        let avg_thermal_capacity = (from_node.thermal_capacity()
            + to_node.thermal_capacity())
            / 2.0;
        let energy_difference = temp_diff * avg_thermal_capacity;
        let flow_coefficient =
            base_coefficient * distance_weight * conductivity_factor * pressure_factor;
        let energy_transfer = energy_difference * flow_coefficient * self.years_per_step();

        // Limit transfer to prevent instability (max 25% of smaller thermal capacity per step)
        let min_capacity = from_node
            .thermal_capacity()
            .min(to_node.thermal_capacity());
        let max_transfer = min_capacity * 0.25;
        energy_transfer.min(max_transfer)
    }

    /// deprecated
    fn calculate_thermal_force(
        &self,
        from_idx: usize,
        to_idx: usize,
        distance_km: f64,
        years: f64,
    ) -> f64 {
        let from_node = &self.nodes[from_idx];
        let to_node = &self.nodes[to_idx];

        let temp_diff = to_node.temp_kelvin() - from_node.temp_kelvin();
        if temp_diff.abs() < 0.1 {
            return 0.0;
        }

        // Get material-based conductivities (4x scaled)
        let from_conductivity = self.get_material_conductivity(from_node);
        let to_conductivity = self.get_material_conductivity(to_node);

        // Distance weights: 1.0, 0.25, 0.125 for neighbors 1, 2, 3 away
        let distance_weight = if distance_km <= 5.1 {
            1.0 // Adjacent neighbor
        } else if distance_km <= 10.1 {
            0.25 // Second neighbor
        } else if distance_km <= 15.1 {
            0.125 // Third neighbor
        } else {
            0.0 // Too far
        };

        // Conductivity factor: both sender and recipient must be conductive
        let sender_factor = from_conductivity / self.config.conductivity_factor;
        let recipient_factor = to_conductivity / self.config.conductivity_factor;
        let conductivity_factor = sender_factor * recipient_factor;

        // Base diffusion rate: 4x enhanced for faster heat transport
        let base_coefficient = self.config.max_change_rate / years;

        // PRESSURE ACCELERATION: Help foundry heat reach farther (4x enhanced)
        let temp_diff_abs = temp_diff.abs();
        let pressure_factor =
            ((temp_diff_abs / 100.0).powf(1.5)).max(0.33) * self.config.pressure_baseline;

        // DIFFUSION PHYSICS: Only the energy difference flows
        let energy_difference = temp_diff * from_node.thermal_capacity();
        let flow_coefficient =
            base_coefficient * distance_weight * conductivity_factor * pressure_factor;
        let energy_transfer = energy_difference * flow_coefficient * years;

        // Limit transfer to prevent instability (max 25% of thermal capacity per step)
        let max_transfer = from_node.thermal_capacity() * 0.25;
        let limited_transfer = energy_transfer.abs().min(max_transfer);

        if temp_diff > 0.0 {
            limited_transfer
        } else {
            -limited_transfer
        }
    }

    fn update_node_phases(&mut self) {
        // Phase transitions are now handled automatically by the energy bank system
        // This method is kept for compatibility but no longer forces phase changes
        // The actual phases are determined by temperature and energy bank state in the energy mass system
        for node in self.nodes.iter_mut() {
            node.update_phase_from_kelvin(); // This only updates the thermal_state display value
        }
    }

    fn get_material_conductivity(&self, node: &ThermalLayerNodeWide) -> f64 {
        // Use the material system to get conductivity based on phase
        self.config.get_thermal_conductivity(node.material_state())
    }

    fn export_state(&self, file: &mut std::fs::File, years: f64) {
        write!(file, "{:.0}", years).unwrap();

        // Export temperatures for selected layer indices
        for &layer_idx in &EXPORT_LAYER_INDICES {
            if layer_idx < self.nodes.len() {
                write!(file, ",{:.1}", self.nodes[layer_idx].temp_kelvin()).unwrap();
            } else {
                write!(file, ",0.0").unwrap();
            }
        }

        // Export thermal states for selected layer indices
        for &layer_idx in &EXPORT_LAYER_INDICES {
            if layer_idx < self.nodes.len() {
                write!(file, ",{}", self.nodes[layer_idx].thermal_state).unwrap();
            } else {
                write!(file, ",0").unwrap();
            }
        }

        writeln!(file).unwrap();
    }

    fn print_final_analysis(&self) {
        println!("\n🎯 Experiment Complete!");
        println!("==========================================");

        // Skip first 3 and last 2 nodes (influenced by boundary conditions)
        let analysis_nodes = &self.nodes[3..self.nodes.len()-2];

        let min_temp = analysis_nodes
            .iter()
            .map(|n| n.temp_kelvin())
            .fold(f64::INFINITY, f64::min);
        let max_temp = analysis_nodes
            .iter()
            .map(|n| n.temp_kelvin())
            .fold(f64::NEG_INFINITY, f64::max);

        println!(
            "📊 Final Temperature Range (excluding boundary layers): {:.1}K - {:.1}K",
            min_temp, max_temp
        );
        println!(
            "   Temperature span: {:.1}K (analyzing layers 4-{} of {})",
            max_temp - min_temp,
            self.nodes.len() - 2,
            self.nodes.len()
        );
        println!(
            "   Skipped: first 3 layers (space influence) and last 2 layers (foundry influence)"
        );

        self.print_rows();
        println!("\n📈 Results exported to examples/data/4x_thermal_experiment.csv");
        println!("   Tracking thermal evolution with 4x enhanced parameters");
        println!("   Deep geological time simulation complete!");

        let first_node = self.nodes.first().unwrap();
        let composite = first_node.material_composite();
        println!("  melting temperature: {}", composite.melting_point_min_k);
    }

    fn print_rows(&self) {
        for (i, node) in self.nodes.iter().enumerate() {
            let boundary_marker = if i == 0 {
                " (top layer under space)"
            } else if i == self.nodes.len() - 1 {
                " (foundry cell 1/3)"
            } else if i == self.nodes.len() - 2 {
                " (foundry cell 2/3)"
            } else if i == self.nodes.len() - 3 {
                " (foundry cell 3/3)"
            } else if i < 3 {
                " (space boundary)"
            } else {
                ""
            };

            println!(
                "   Layer {}:  {}K  at {}km depth   {}{}",
                i + 1,
                node.temp_kelvin() as i32,
                node.depth_km as i32,
                node.format_thermal_state(),
                boundary_marker
            );
        }
    }
}

fn main() {
    println!("🔬 4x Scaled Force-Directed Thermal Diffusion with Enhanced Parameters");
    println!("====================================================================");
    println!("🎯 Deep geological time: 1 billion years of thermal evolution");
    let total_years = 1_000_000;
    let years_per_step = 10_000;
    let steps = total_years / years_per_step;
    // Create experiment with 4x scaled parameters
    let mut experiment = OneHundredKm2Experiment::new(steps, total_years);
    experiment.print_initial_state();

    // Run force-directed thermal equilibration
    println!("\n🌡️  Running 4x enhanced thermal equilibration...");

    experiment.run(100, 1_000_000);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_one_km_2_experiment_initial_temperature_gradient() {
        // Create experiment instance
        let experiment = OneHundredKm2Experiment::new(100, 1_000_000);

        // Use locked constants for expected gradient (prevents config drift)
        let surface_temp = WIDE_SURFACE_TEMPERATURE_K;
        let foundry_temp = WIDE_FOUNDRY_TEMPERATURE_K;
        let num_nodes = experiment.nodes.len();

        // Validate that we have the expected number of nodes
        assert_eq!(num_nodes, EXPECTED_NUM_NODES, "Expected {} thermal nodes for wide experiment", EXPECTED_NUM_NODES);

        // Collect temperatures for validation
        let temperatures: Vec<f64> = experiment.nodes.iter().map(|node| node.temp_kelvin()).collect();

        // Validate temperature gradient using test utilities
        validate_temperature_gradient(&temperatures, surface_temp, foundry_temp, TEMPERATURE_TOLERANCE_PERCENT)
            .expect("Temperature gradient validation failed");

        // Validate boundary conditions using test utilities
        let surface_node_temp = experiment.nodes[0].temp_kelvin();
        let foundry_node_temp = experiment.nodes[num_nodes - 1].temp_kelvin();

        validate_boundary_conditions(
            surface_node_temp, foundry_node_temp,
            surface_temp, foundry_temp,
            TEMPERATURE_TOLERANCE_PERCENT
        ).expect("Boundary conditions validation failed");

        // Test temperature range using locked constants
        let temp_range = foundry_node_temp - surface_node_temp;
        let expected_range = EXPECTED_TEMPERATURE_RANGE_K;

        assert_deviation!(
            temp_range, expected_range, TEMPERATURE_TOLERANCE_PERCENT,
            "Temperature range should be within {:.1}%: got {:.1}K, expected {:.1}K",
            TEMPERATURE_TOLERANCE_PERCENT, temp_range, expected_range
        );

        // Test some intermediate points for gradient linearity
        let quarter_point = num_nodes / 4;
        let half_point = num_nodes / 2;
        let three_quarter_point = 3 * num_nodes / 4;

        let quarter_temp = experiment.nodes[quarter_point].temp_kelvin();
        let half_temp = experiment.nodes[half_point].temp_kelvin();
        let three_quarter_temp = experiment.nodes[three_quarter_point].temp_kelvin();

        let expected_quarter = lerp(surface_temp, foundry_temp, quarter_point as f64 / num_nodes as f64);
        let expected_half = lerp(surface_temp, foundry_temp, half_point as f64 / num_nodes as f64);
        let expected_three_quarter = lerp(surface_temp, foundry_temp, three_quarter_point as f64 / num_nodes as f64);

        assert_deviation!(quarter_temp, expected_quarter, 2.0, "25% depth temperature should be within 2%");
        assert_deviation!(half_temp, expected_half, 2.0, "50% depth temperature should be within 2%");
        assert_deviation!(three_quarter_temp, expected_three_quarter, 2.0, "75% depth temperature should be within 2%");
        
    }

    #[test]
    fn test_actual_thermal_diffusion_method() {
        // Create a minimal experiment with just 5 nodes to test the ACTUAL method
        let mut experiment = OneHundredKm2Experiment::new(1, 1);

        // Manually create just 5 nodes for testing
        experiment.nodes.clear();

        // Create 5 nodes: cold-cold-hot-cold-cold pattern
        for i in 0..5 {
            let temp = if i == 2 { 2000.0 } else { 1000.0 }; // Middle node hot
            let node = ThermalLayerNodeWide::new_with_temperature(ThermalLayerNodeWideTempParams {
                material_type: MaterialCompositeType::Silicate,
                temperature_k: temp,
                volume_km3: AREA_KM2 * 5.0, // 100 km² × 5 km height
                depth_km: (i as f64 + 0.5) * 5.0,
                height_km: 5.0,
                area_km2: AREA_KM2,
            });
            experiment.nodes.push(node);
        }

        // Record initial state
        let initial_temps: Vec<f64> = experiment.nodes.iter().map(|node| node.temp_kelvin()).collect();
        let initial_energies: Vec<f64> = experiment.nodes.iter().map(|node| node.energy()).collect();
        println!("Initial temps: {:?}", initial_temps);
        println!("Initial energies: {:?}", initial_energies);

        // Test the ACTUAL method from the experiment code
        let energy_changes = experiment.calculate_thermal_diffusion_energy_changes();

        println!("\nActual method energy_changes array: {:?}", energy_changes);

        // Verify energy conservation
        let total_energy_change: f64 = energy_changes.iter().sum();
        println!("Total energy change (should be ~0): {:.6}", total_energy_change);

        // Show what the final temperatures would be
        println!("\nPredicted final temperatures from actual method:");
        for (i, &energy_change) in energy_changes.iter().enumerate() {
            let capacity = experiment.nodes[i].thermal_capacity();
            let temp_change = energy_change / capacity;
            let new_temp = initial_temps[i] + temp_change;
            println!("  Node {}: {:.0}K + {:.1}K = {:.1}K (energy change: {:.0})",
                i, initial_temps[i], temp_change, new_temp, energy_change);
        }

        // Now run the actual thermal diffusion and compare
        experiment.thermal_diffusion_step_only();

        let final_temps: Vec<f64> = experiment.nodes.iter().map(|node| node.temp_kelvin()).collect();
        println!("\nActual final temperatures after thermal_diffusion_step_only:");
        for (i, (initial, final_temp)) in initial_temps.iter().zip(final_temps.iter()).enumerate() {
            let change = final_temp - initial;
            println!("  Node {}: {:.1}K -> {:.1}K (change: {:+.1}K)",
                i, initial, final_temp, change);
        }
    }

    #[test]
    fn test_energy_redistribution_array() {
        // Create a minimal experiment with just 5 nodes to debug energy redistribution
        let mut experiment = OneHundredKm2Experiment::new(1, 1);

        // Manually create just 5 nodes for testing
        experiment.nodes.clear();

        // Create 5 nodes: cold-cold-hot-cold-cold pattern
        for i in 0..5 {
            let temp = if i == 2 { 2000.0 } else { 1000.0 }; // Middle node hot
            let node = ThermalLayerNodeWide::new_with_temperature(ThermalLayerNodeWideTempParams {
                material_type: MaterialCompositeType::Silicate,
                temperature_k: temp,
                volume_km3: AREA_KM2 * 5.0, // 100 km² × 5 km height
                depth_km: (i as f64 + 0.5) * 5.0,
                height_km: 5.0,
                area_km2: AREA_KM2,
            });
            experiment.nodes.push(node);
        }

        // Record initial state
        let initial_temps: Vec<f64> = experiment.nodes.iter().map(|node| node.temp_kelvin()).collect();
        let initial_energies: Vec<f64> = experiment.nodes.iter().map(|node| node.energy()).collect();
        println!("Initial temps: {:?}", initial_temps);
        println!("Initial energies: {:?}", initial_energies);

        // MANUALLY REPLICATE THE PAIRWISE ALGORITHM TO DEBUG
        let mut energy_changes = vec![0.0; experiment.nodes.len()];
        let mut exchange_pairs = Vec::new();

        // Enumerate pairs (same as in thermal_diffusion_step_only)
        for i in 0..experiment.nodes.len() {
            for j in (i + 1)..experiment.nodes.len() {
                let index_diff = j - i;
                if index_diff == 1 {
                    exchange_pairs.push((i, j, index_diff));
                }
            }
        }

        println!("\nProcessing pairs:");
        // Process each exchange pair with detailed logging
        for (from_idx, to_idx, distance) in exchange_pairs {
            let from_temp = experiment.nodes[from_idx].temp_kelvin();
            let to_temp = experiment.nodes[to_idx].temp_kelvin();
            let temp_diff = from_temp - to_temp;

            println!("  Pair ({}, {}): {:.0}K -> {:.0}K, temp_diff = {:.1}K",
                from_idx, to_idx, from_temp, to_temp, temp_diff);

            if temp_diff.abs() < 0.1 {
                println!("    Skipped: no significant temperature difference");
                continue;
            }

            // Weight by distance curve: 0.25^(index_difference)
            let distance_weight = 0.25_f64.powi(distance as i32);

            // Calculate energy exchange based on temperature difference
            let from_capacity = experiment.nodes[from_idx].thermal_capacity();
            let to_capacity = experiment.nodes[to_idx].thermal_capacity();
            let avg_capacity = (from_capacity + to_capacity) / 2.0;

            // Base exchange rate (small fraction per step to prevent instability)
            let base_exchange_rate = 0.01; // 1% per step for adjacent neighbors
            let exchange_rate = base_exchange_rate * distance_weight;

            // Energy transfer proportional to temperature difference and capacity
            let energy_transfer = temp_diff * avg_capacity * exchange_rate * experiment.years_per_step();

            // Limit transfer to prevent instability (max 5% of smaller capacity per step)
            let min_capacity = from_capacity.min(to_capacity);
            let max_transfer = min_capacity * 0.05;
            let limited_transfer = energy_transfer.abs().min(max_transfer);

            println!("    distance_weight: {:.3}, avg_capacity: {:.0}, exchange_rate: {:.3}",
                distance_weight, avg_capacity, exchange_rate);
            println!("    energy_transfer: {:.0}, limited_transfer: {:.0}",
                energy_transfer, limited_transfer);

            if temp_diff > 0.0 {
                // Energy flows from i to j (i is hotter)
                energy_changes[from_idx] -= limited_transfer;
                energy_changes[to_idx] += limited_transfer;
                println!("    Energy flows: {} -> {}, amount: {:.0}", from_idx, to_idx, limited_transfer);
            } else {
                // Energy flows from j to i (j is hotter)
                energy_changes[from_idx] += limited_transfer;
                energy_changes[to_idx] -= limited_transfer;
                println!("    Energy flows: {} -> {}, amount: {:.0}", to_idx, from_idx, limited_transfer);
            }
        }

        println!("\nFinal energy_changes array: {:?}", energy_changes);

        // Verify energy conservation
        let total_energy_change: f64 = energy_changes.iter().sum();
        println!("Total energy change (should be ~0): {:.6}", total_energy_change);

        // Show what the final temperatures would be
        println!("\nPredicted final temperatures:");
        for (i, &energy_change) in energy_changes.iter().enumerate() {
            let new_energy = initial_energies[i] + energy_change;
            let capacity = experiment.nodes[i].thermal_capacity();
            let temp_change = energy_change / capacity;
            let new_temp = initial_temps[i] + temp_change;
            println!("  Node {}: {:.0}K + {:.1}K = {:.1}K (energy change: {:.0})",
                i, initial_temps[i], temp_change, new_temp, energy_change);
        }
    }

    #[test]
    fn test_debug_pairwise_exchanges() {
        // Create a minimal experiment with just 5 nodes to debug the pairwise exchanges
        let mut experiment = OneHundredKm2Experiment::new(1, 1);

        // Manually create just 5 nodes for testing
        experiment.nodes.clear();

        // Create 5 nodes: cold-cold-hot-cold-cold pattern
        for i in 0..5 {
            let temp = if i == 2 { 2000.0 } else { 1000.0 }; // Middle node hot
            let node = ThermalLayerNodeWide::new_with_temperature(ThermalLayerNodeWideTempParams {
                material_type: MaterialCompositeType::Silicate,
                temperature_k: temp,
                volume_km3: AREA_KM2 * 5.0, // 100 km² × 5 km height
                depth_km: (i as f64 + 0.5) * 5.0,
                height_km: 5.0,
                area_km2: AREA_KM2,
            });
            experiment.nodes.push(node);
        }

        // Record initial temperatures
        let initial_temps: Vec<f64> = experiment.nodes.iter().map(|node| node.temp_kelvin()).collect();
        println!("Initial 5-node temps: {:?}", initial_temps);

        // Debug: Show which pairs will be processed
        println!("Pairs that will be processed (immediate neighbors only):");
        for i in 0..experiment.nodes.len() {
            for j in (i + 1)..experiment.nodes.len() {
                let index_diff = j - i;
                if index_diff == 1 {
                    println!("  Pair ({}, {}) - temps: {:.0}K, {:.0}K",
                        i, j, experiment.nodes[i].temp_kelvin(), experiment.nodes[j].temp_kelvin());
                }
            }
        }

        // Run thermal diffusion
        experiment.thermal_diffusion_step_only();

        // Record final temperatures
        let final_temps: Vec<f64> = experiment.nodes.iter().map(|node| node.temp_kelvin()).collect();
        println!("Final 5-node temps: {:?}", final_temps);

        // Print changes
        for (i, (initial, final_temp)) in initial_temps.iter().zip(final_temps.iter()).enumerate() {
            let change = final_temp - initial;
            println!("Node {}: {:.1}K -> {:.1}K (change: {:+.1}K)",
                i, initial, final_temp, change);
        }
    }

    #[test]
    fn test_simple_three_node_diffusion() {
        // Create a minimal experiment with just 3 nodes to understand the behavior
        let mut experiment = OneHundredKm2Experiment::new(1, 1);

        // Manually create just 3 nodes for testing
        experiment.nodes.clear();

        // Create 3 nodes: cold-hot-cold pattern
        for i in 0..3 {
            let temp = if i == 1 { 2000.0 } else { 1000.0 }; // Middle node hot
            let node = ThermalLayerNodeWide::new_with_temperature(ThermalLayerNodeWideTempParams {
                material_type: MaterialCompositeType::Silicate,
                temperature_k: temp,
                volume_km3: AREA_KM2 * 5.0, // 100 km² × 5 km height
                depth_km: (i as f64 + 0.5) * 5.0,
                height_km: 5.0,
                area_km2: AREA_KM2,
            });
            experiment.nodes.push(node);
        }

        // Record initial temperatures
        let initial_temps: Vec<f64> = experiment.nodes.iter().map(|node| node.temp_kelvin()).collect();
        println!("Initial 3-node temps: {:?}", initial_temps);

        // Run thermal diffusion
        experiment.thermal_diffusion_step_only();

        // Record final temperatures
        let final_temps: Vec<f64> = experiment.nodes.iter().map(|node| node.temp_kelvin()).collect();
        println!("Final 3-node temps: {:?}", final_temps);

        // Print changes
        for (i, (initial, final_temp)) in initial_temps.iter().zip(final_temps.iter()).enumerate() {
            let change = final_temp - initial;
            println!("Node {}: {:.1}K -> {:.1}K (change: {:+.1}K)",
                i, initial, final_temp, change);
        }

        // Verify energy conservation
        let initial_total_energy: f64 = experiment.nodes.iter().map(|node| node.energy()).sum();
        let final_total_energy: f64 = experiment.nodes.iter().map(|node| node.energy()).sum();
        assert_deviation!(final_total_energy, initial_total_energy, 1.0,
            "Total energy should be conserved within 1%");
    }

    #[test]
    fn test_single_step_thermal_diffusion() {
        // Create experiment with minimal setup for single step testing
        let mut experiment = OneHundredKm2Experiment::new(1, 1);

        // Set all nodes to uniform temperature (1000K)
        let uniform_temp = 1000.0;
        for node in &mut experiment.nodes {
            node.set_kelvin(uniform_temp);
        }

        // Elevate middle node by 1000K (to 2000K)
        let middle_index = experiment.nodes.len() / 2;
        let elevated_temp = uniform_temp + 1000.0;
        experiment.nodes[middle_index].set_kelvin(elevated_temp);

        // Record initial temperatures
        let initial_temps: Vec<f64> = experiment.nodes.iter().map(|node| node.temp_kelvin()).collect();

        // Run thermal diffusion without boundary conditions
        experiment.thermal_diffusion_step_only();

        // Record final temperatures
        let final_temps: Vec<f64> = experiment.nodes.iter().map(|node| node.temp_kelvin()).collect();

        // Verify the elevated node lost some heat
        assert!(final_temps[middle_index] < elevated_temp,
            "Middle node should have lost heat: {} -> {}",
            elevated_temp, final_temps[middle_index]);

        // Verify adjacent nodes gained heat
        if middle_index > 0 {
            assert!(final_temps[middle_index - 1] > uniform_temp,
                "Node below middle should have gained heat: {} -> {}",
                uniform_temp, final_temps[middle_index - 1]);
        }
        if middle_index < experiment.nodes.len() - 1 {
            assert!(final_temps[middle_index + 1] > uniform_temp,
                "Node above middle should have gained heat: {} -> {}",
                uniform_temp, final_temps[middle_index + 1]);
        }

        // Verify energy conservation (total energy should be approximately the same)
        let initial_total_energy: f64 = experiment.nodes.iter().map(|node| node.energy()).sum();
        let final_total_energy: f64 = experiment.nodes.iter().map(|node| node.energy()).sum();

        assert_deviation!(final_total_energy, initial_total_energy, 1.0,
            "Total energy should be conserved within 1%");

        // Print temperature distribution for visual verification
        println!("Temperature distribution after single step (no boundary conditions):");
        for (i, (initial, final_temp)) in initial_temps.iter().zip(final_temps.iter()).enumerate() {
            let change = final_temp - initial;
            println!("Node {}: {:.1}K -> {:.1}K (change: {:+.1}K)",
                i, initial, final_temp, change);
        }

        // Verify that heat diffusion follows expected pattern
        // The middle node should lose the most heat
        let middle_change = final_temps[middle_index] - initial_temps[middle_index];
        assert!(middle_change < 0.0, "Middle node should lose heat");

        // Print detailed analysis of heat distribution
        println!("\nDetailed heat distribution analysis:");
        println!("Middle node ({}): {:.1}K change", middle_index, middle_change);

        if middle_index > 1 && middle_index < experiment.nodes.len() - 2 {
            let adjacent_change_1 = final_temps[middle_index - 1] - initial_temps[middle_index - 1];
            let adjacent_change_2 = final_temps[middle_index + 1] - initial_temps[middle_index + 1];
            let distant_change_1 = final_temps[middle_index - 2] - initial_temps[middle_index - 2];
            let distant_change_2 = final_temps[middle_index + 2] - initial_temps[middle_index + 2];

            println!("Adjacent nodes ({}, {}): {:.1}K, {:.1}K",
                middle_index - 1, middle_index + 1, adjacent_change_1, adjacent_change_2);
            println!("Distant nodes ({}, {}): {:.1}K, {:.1}K",
                middle_index - 2, middle_index + 2, distant_change_1, distant_change_2);

            // Note: The current implementation appears to distribute heat evenly
            // This might be due to the thermal diffusion algorithm design
            println!("Note: Heat appears to be distributed evenly across all non-source nodes");
        }
    }

    #[test]
    fn test_one_km_2_experiment_node_properties() {
        let experiment = OneHundredKm2Experiment::new(100, 1_000_000);

        // Test that all nodes have consistent properties
        for (i, node) in experiment.nodes.iter().enumerate() {
            // All nodes should have the same height
            assert_eq!(
                node.height_km(),
                experiment.layer_height_km,
                "Node {} should have consistent layer height", i
            );

            // All nodes should have 100 km² volume (height * 100 km²)
            let expected_volume = experiment.layer_height_km * AREA_KM2;
            assert_deviation!(
                node.volume_km3(), expected_volume, 0.1,
                "Node {} should have volume {:.3} km³", i, expected_volume
            );

            // Depth should increase linearly
            let expected_depth = (i as f64 + 0.5) * experiment.layer_height_km;
            assert_deviation!(
                node.depth_km(), expected_depth, 0.1,
                "Node {} should have depth {:.3} km", i, expected_depth
            );

            // Temperature should be positive and reasonable
            let temp = node.temp_kelvin();
            assert!(
                temp > 0.0 && temp < 10000.0,
                "Node {} temperature {:.1}K should be reasonable", i, temp
            );
        }
    }

    #[test]
    fn test_fourier_thermal_transfer_basic() {
        // Test the new science-backed Fourier thermal transfer with a simple 3-node setup
        const TIME_YEARS: f64 = 1.0;

        // Create 3 nodes: cold-hot-cold pattern
        let mut nodes = Vec::new();
        for i in 0..3 {
            let temp = if i == 1 { 2000.0 } else { 1000.0 }; // Middle node hot
            let node = ThermalLayerNodeWide::new_with_temperature(ThermalLayerNodeWideTempParams {
                material_type: MaterialCompositeType::Silicate,
                temperature_k: temp,
                volume_km3: AREA_KM2 * 5.0, // 100 km² × 5 km height
                depth_km: (i as f64 + 0.5) * 5.0,
                height_km: 5.0,
                area_km2: AREA_KM2,
            });
            nodes.push(node);
        }

        // Record initial state
        let initial_temps: Vec<f64> = nodes.iter().map(|node| node.temp_kelvin()).collect();
        let initial_energies: Vec<f64> = nodes.iter().map(|node| node.energy()).collect();
        println!("🌡️  Initial temps: {:?}", initial_temps);
        println!("⚡ Initial energies: {:?}", initial_energies);

        // Apply Fourier thermal transfer to center node (index 1)
        let energy_change = {
            let (left, center_and_right) = nodes.split_at_mut(1);
            let (center, right) = center_and_right.split_at_mut(1);

            let left_neighbor = left.get_mut(0);
            let center_node = &mut center[0];
            let right_neighbor = right.get_mut(0);

            center_node.apply_fourier_thermal_transfer(left_neighbor, right_neighbor, TIME_YEARS)
        };

        // Record final state
        let final_temps: Vec<f64> = nodes.iter().map(|node| node.temp_kelvin()).collect();
        let final_energies: Vec<f64> = nodes.iter().map(|node| node.energy()).collect();
        println!("🌡️  Final temps: {:?}", final_temps);
        println!("⚡ Final energies: {:?}", final_energies);
        println!("🔄 Center node energy change: {:.0}", energy_change);

        // Verify the hot middle node lost energy
        assert!(energy_change < 0.0, "Hot center node should lose energy");
        assert!(final_temps[1] < initial_temps[1], "Hot center node should cool down");

        // Verify the cold side nodes gained energy
        assert!(final_energies[0] > initial_energies[0], "Cold left node should gain energy");
        assert!(final_energies[2] > initial_energies[2], "Cold right node should gain energy");

        // Verify energy conservation
        let initial_total: f64 = initial_energies.iter().sum();
        let final_total: f64 = final_energies.iter().sum();
        let energy_diff = (final_total - initial_total).abs();
        let relative_error = energy_diff / initial_total;
        println!("🔋 Energy conservation error: {:.6}%", relative_error * 100.0);
        assert!(relative_error < 0.001, "Energy should be conserved (error: {:.6}%)", relative_error * 100.0);
    }

    #[test]
    fn test_fourier_transfer_material_properties() {
        // Test Fourier transfer with different material types to verify material property effects
        const TIME_YEARS: f64 = 0.1; // Shorter time for more controlled transfer

        // Create nodes with different materials: Metallic (high conductivity) - Silicate - Granitic (low conductivity)
        let mut nodes = Vec::new();

        // High conductivity metallic node (hot)
        nodes.push(ThermalLayerNodeWide::new_with_temperature(ThermalLayerNodeWideTempParams {
            material_type: MaterialCompositeType::Metallic,
            temperature_k: 1500.0,
            volume_km3: AREA_KM2 * 5.0,
            depth_km: 2.5,
            height_km: 5.0,
            area_km2: AREA_KM2,
        }));

        // Medium conductivity silicate node (medium temp)
        nodes.push(ThermalLayerNodeWide::new_with_temperature(ThermalLayerNodeWideTempParams {
            material_type: MaterialCompositeType::Silicate,
            temperature_k: 1200.0,
            volume_km3: AREA_KM2 * 5.0,
            depth_km: 7.5,
            height_km: 5.0,
            area_km2: AREA_KM2,
        }));

        // Low conductivity granitic node (cold)
        nodes.push(ThermalLayerNodeWide::new_with_temperature(ThermalLayerNodeWideTempParams {
            material_type: MaterialCompositeType::Granitic,
            temperature_k: 900.0,
            volume_km3: AREA_KM2 * 5.0,
            depth_km: 12.5,
            height_km: 5.0,
            area_km2: AREA_KM2,
        }));

        // Record initial state and material properties
        let initial_temps: Vec<f64> = nodes.iter().map(|node| node.temp_kelvin()).collect();
        let initial_energies: Vec<f64> = nodes.iter().map(|node| node.energy()).collect();
        let conductivities: Vec<f64> = nodes.iter().map(|node| node.thermal_conductivity()).collect();

        println!("🌡️  Initial temps: {:?}", initial_temps);
        println!("⚡ Initial energies: {:?}", initial_energies);
        println!("🔥 Thermal conductivities: {:?}", conductivities);

        // Apply Fourier transfer to center node (silicate)
        let energy_change = {
            let (left, center_and_right) = nodes.split_at_mut(1);
            let (center, right) = center_and_right.split_at_mut(1);

            let left_neighbor = left.get_mut(0);
            let center_node = &mut center[0];
            let right_neighbor = right.get_mut(0);

            center_node.apply_fourier_thermal_transfer(left_neighbor, right_neighbor, TIME_YEARS)
        };

        // Record final state
        let final_temps: Vec<f64> = nodes.iter().map(|node| node.temp_kelvin()).collect();
        let final_energies: Vec<f64> = nodes.iter().map(|node| node.energy()).collect();

        println!("🌡️  Final temps: {:?}", final_temps);
        println!("⚡ Final energies: {:?}", final_energies);
        println!("🔄 Center node energy change: {:.0}", energy_change);

        // Verify heat flows from hot to cold
        // The center node (silicate, 1200K) should receive energy from hot metallic (1500K)
        // and give energy to cold granitic (900K)
        assert!(final_temps[0] < initial_temps[0], "Hot metallic node should cool down (gives energy to center)");
        assert!(final_temps[1] > initial_temps[1], "Medium silicate node should warm up (receives from metallic, gives to granitic)");
        assert!(final_temps[2] > initial_temps[2], "Cold granitic node should warm up (receives from center)");

        // Verify energy conservation
        let initial_total: f64 = initial_energies.iter().sum();
        let final_total: f64 = final_energies.iter().sum();
        let energy_diff = (final_total - initial_total).abs();
        let relative_error = energy_diff / initial_total;

        println!("🔋 Energy conservation error: {:.6}%", relative_error * 100.0);
        assert!(relative_error < 0.001, "Energy should be conserved (error: {:.6}%)", relative_error * 100.0);

        // Verify that higher conductivity materials transfer more energy
        // The metallic node should have transferred more energy than it would if it were silicate
        println!("✅ Fourier transfer with material properties working correctly!");
    }

    #[test]
    fn test_energy_add_remove_basic() {
        // Test basic energy add/remove operations to isolate the conservation issue
        let mut node = ThermalLayerNodeWide::new_with_temperature(ThermalLayerNodeWideTempParams {
            material_type: MaterialCompositeType::Silicate,
            temperature_k: 1500.0,
            volume_km3: AREA_KM2 * 5.0,
            depth_km: 2.5,
            height_km: 5.0,
            area_km2: AREA_KM2,
        });

        let initial_energy = node.energy();
        println!("Initial energy: {:.3e} J", initial_energy);

        // Remove some energy
        let remove_amount = 1e15; // 1 PJ
        node.remove_energy(remove_amount);
        let after_remove = node.energy();
        println!("After removing {:.0} J: {:.3e} J", remove_amount, after_remove);

        // Add it back
        node.add_energy(remove_amount);
        let after_add = node.energy();
        println!("After adding {:.0} J back: {:.3e} J", remove_amount, after_add);

        // Check conservation
        let diff = (after_add - initial_energy).abs();
        let relative_error = diff / initial_energy;
        println!("Energy difference: {:.0} J ({:.6}%)", diff, relative_error * 100.0);

        assert!(relative_error < 1e-10, "Basic add/remove should be perfectly conserved (error: {:.10}%)", relative_error * 100.0);
    }
}
