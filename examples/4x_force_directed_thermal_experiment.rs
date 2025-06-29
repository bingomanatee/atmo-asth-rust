#![allow(dead_code)]
#![allow(unused_variables)]

use std::collections::VecDeque;
use std::fs::File;
use std::io::Write;

// Import the real EnergyMass trait and material system
extern crate atmo_asth_rust;
use atmo_asth_rust::energy_mass::EnergyMass;
use atmo_asth_rust::example::{ExperimentState, ThermalLayerNode};
use atmo_asth_rust::example::thermal_layer_node::ThermalLayerNodeParams;
use atmo_asth_rust::material_composite::{MATERIAL_COMPOSITES};
use atmo_asth_rust::energy_mass_composite::{StandardEnergyMassComposite, 
                                            get_profile_fast, 
                                            MaterialCompositeType, 
                                            MaterialPhase,
                                            EnergyMassComposite};
// Re-export MaterialPhase for easier access
use atmo_asth_rust::temp_utils::energy_from_kelvin;

// Export layer indices - key layer positions for analysis
const EXPORT_LAYER_INDICES: [usize; 9] = [10, 15, 20, 25, 30, 35, 40, 45, 50];

/// 4x Scaled Force-Directed Thermal Diffusion Experiment
struct ScaledThermalExperiment {
    nodes: Vec<ThermalLayerNode>,
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

impl ScaledThermalExperiment {
    fn new(steps: u64, total_years: u64) -> Self {
        let config = ExperimentState::basic_experiment_state();
        let num_nodes = 60;
        let total_depth_km = 300.0; // 300km total depth
        let layer_height_km = total_depth_km / num_nodes as f64; // 5km per layer

        let mut nodes = Vec::new();

        // Create thermal nodes with realistic temperature profile
        for i in 0..num_nodes {
            let depth_km = (i as f64 + 0.5) * layer_height_km;
            let volume_km3 = 1.0 * layer_height_km; // 1 km² surface area

            // Surface layers
            let temp_kelvin = {
                let surface_temperature = config.surface_temperature_k;
                let foundry_temperature = config.foundry_temperature_k;
                let diff = foundry_temperature - surface_temperature;
                surface_temperature + diff * (i as f64 / num_nodes as f64)
            };
            
            let material_profile = get_profile_fast(&MaterialCompositeType::Silicate,
            &MaterialPhase::Liquid);
            
            let energy_joules = energy_from_kelvin(
                temp_kelvin, volume_km3, 
                material_profile.specific_heat_capacity_j_per_kg_k
            );

            let node = ThermalLayerNode::new(ThermalLayerNodeParams {
                material_type: MaterialCompositeType::Silicate,
                energy_joules,
                volume_km3,
                depth_km,
                height_km: layer_height_km,
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
            foundry_end: 8,    // 8 foundry layers (0-40km)
            surface_start: 52, // Surface cooling starts at layer 52 (260km)
            surface_end: 60,   // Surface cooling ends at layer 60 (300km)
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
        println!("   {} thermal nodes", self.nodes.len());

        println!("\n🌡️  Initial Temperature Profile:");
        println!(
            "   Foundry layers (0-40km): {}K",
            self.config.foundry_temperature_k
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
    
    pub fn years_per_step(&self)  -> f64{
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

        // EXPONENTIAL THERMAL DIFFUSION: Each layer exchanges with 2 neighbors on each side
        for node_index in 1..self.nodes.len() {
            // Exchange with 2 layers on each side (i-2, i-1, i+1, i+2) with exponential falloff
            for offset in [-2, -1, 1, 2] {
                let target_place = node_index as i32 + offset;
                if target_place < 0 || target_place >= self.nodes.len() as i32 {
                    continue;
                }
                let target_index = target_place as usize;

                // Energy-conservative exchange between nodes (distance computed internally)
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
        self.update_thermal_states();
    }

    fn apply_boundary_conditions(&mut self) {
        self.nodes[0]
            .energy_mass
            .set_kelvin(self.config.foundry_temperature_k);
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
        from_node: &ThermalLayerNode,
        to_node: &ThermalLayerNode,
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
        let avg_thermal_capacity = (from_node.energy_mass.thermal_capacity() + to_node.energy_mass.thermal_capacity()) / 2.0;
        let energy_difference = temp_diff * avg_thermal_capacity;
        let flow_coefficient =
            base_coefficient * distance_weight * conductivity_factor * pressure_factor;
        let energy_transfer = energy_difference * flow_coefficient * self.years_per_step();

        // Limit transfer to prevent instability (max 25% of smaller thermal capacity per step)
        let min_capacity = from_node.energy_mass.thermal_capacity().min(to_node.energy_mass.thermal_capacity());
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
        let energy_difference = temp_diff * from_node.energy_mass.thermal_capacity();
        let flow_coefficient =
            base_coefficient * distance_weight * conductivity_factor * pressure_factor;
        let energy_transfer = energy_difference * flow_coefficient * years;

        // Limit transfer to prevent instability (max 25% of thermal capacity per step)
        let max_transfer = from_node.energy_mass.thermal_capacity() * 0.25;
        let limited_transfer = energy_transfer.abs().min(max_transfer);

        if temp_diff > 0.0 {
            limited_transfer
        } else {
            -limited_transfer
        }
    }

    fn update_thermal_states(&mut self) {
        for (i, node) in self.nodes.iter_mut().enumerate() {
            let material_type = node.energy_mass.material_composite_type();
            let profile = node.energy_mass.material_composite_profile();
            
            let composite = MATERIAL_COMPOSITES.get(&node.energy_mass.material_composite_type()).unwrap();
            
            // @TODO: make melting and cooling take a nonzero time
            let temp = node.temp_kelvin();

                if temp > composite.melting_point_min_k {
                    node.thermal_state = 0; // Liquid/magma
                } else {
                    node.thermal_state = 100; // solid
                }
            
        }
    }

    fn get_material_conductivity(&self, node: &ThermalLayerNode) -> f64 {
        match node.material_state() {
            MaterialPhase::Solid => self.config.solid_conductivity,
            MaterialPhase::Liquid => self.config.liquid_conductivity,
            MaterialPhase::Gas => self.config.transition_conductivity,
        }
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

        let min_temp = self
            .nodes
            .iter()
            .map(|n| n.temp_kelvin())
            .fold(f64::INFINITY, f64::min);
        let max_temp = self
            .nodes
            .iter()
            .map(|n| n.temp_kelvin())
            .fold(f64::NEG_INFINITY, f64::max);

        println!(
            "📊 Final Temperature Range: {:.1}K - {:.1}K",
            min_temp, max_temp
        );
        println!(
            "   Temperature span: {:.1}K ({} ...{})",
            max_temp - min_temp,
            min_temp,
            max_temp
        );

        self.print_rows();
        println!("\n📈 Results exported to examples/data/4x_thermal_experiment.csv");
        println!("   Tracking thermal evolution with 4x enhanced parameters");
        println!("   Deep geological time simulation complete!");

        let first_node = self.nodes.first().unwrap();
        let composite = first_node.energy_mass.material_composite();
        println!("  melting temperature: {}", composite.melting_point_min_k);
    }

    fn print_rows(&self) {
        for (i, node) in self.nodes.iter().enumerate() {
            println!(
                "   Layer {}:  {}K  at {}km depth   {}",
                i + 1,
                node.temp_kelvin() as i32,
                node.depth_km as i32,
                node.format_thermal_state()
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
    let steps  = total_years/years_per_step;
    // Create experiment with 4x scaled parameters
    let mut experiment = ScaledThermalExperiment::new(steps, total_years);
    experiment.print_initial_state();

    // Run force-directed thermal equilibration
    println!("\n🌡️  Running 4x enhanced thermal equilibration...");

    experiment.run(100, 1_000_000);
}
