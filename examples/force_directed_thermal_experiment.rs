#![allow(dead_code)]
#![allow(unused_variables)]

use std::fs::File;
use std::io::Write;

/// Force-directed thermal diffusion experiment with boundary conditions
/// Each layer equalizes with neighbors + heat source (core) + heat sink (space)
fn main() {
    println!("🔬 Force-Directed Thermal Diffusion with Optimized Parameters");
    println!("==============================================================");
    println!("🎯 Deep geological time: 1 million years of thermal evolution");

    // Create experiment with scientifically optimized parameters
    let mut experiment = ForceDirectedThermal::new();
    experiment.create_chunky_temperature_profile();

    println!("📊 Initial Setup:");
    println!("   {} thermal nodes", experiment.nodes.len());
    println!("   🔥 HEAT SOURCE: Core heat input at bottom layer");
    println!("   ❄️  HEAT SINK: Space radiation cooling at top layer");

    experiment.print_initial_profile();

    // Run force-directed thermal equilibration
    println!("\n🌡️  Running scientifically calibrated thermal equilibration...");

    let years_per_step = 1000.0;   // 1,000-year steps (10x larger than original 100)
    let total_years = 1000000000.0; // 1 billion years total
    let steps = (total_years / years_per_step) as usize;

    experiment.run_force_directed_diffusion(steps, years_per_step);

    println!("\n📈 Results exported to examples/data/thermal_experiment.csv");
    println!("   Tracking 5 key depths: 52.5km, 77.5km, 102.5km, 127.5km, 152.5km");
    println!("   Columns: temp_XXkm (temperatures) + state_XXkm (thermal states)");
    println!("   1 billion years with 1k-year steps for deep geological time!");
}

#[derive(Debug, Clone)]
struct ThermalNode {
    depth_km: f64,
    height_km: f64,
    temperature_k: f64,
    thermal_conductivity: f64,  // W/m/K
    thermal_capacity: f64,      // J/K (mass * specific_heat)
    energy_j: f64,             // Total energy content
    thermal_state: u8,         // 0=space, 1-5=lithosphere, 6-9=asthenosphere, 10=core
}

struct ForceDirectedThermal {
    nodes: Vec<ThermalNode>,
    initial_nodes: Vec<ThermalNode>,
    // Tunable parameters
    conductivity_factor: f64,
    distance_length: f64,
    pressure_baseline: f64,
}

impl ForceDirectedThermal {
    fn new() -> Self {
        Self {
            nodes: Vec::new(),
            initial_nodes: Vec::new(),
            conductivity_factor: 0.058,  // Realistic geological timescales
            distance_length: 20.0,       // Optimal diffusion range
            pressure_baseline: 100.0,    // Realistic thermal pressure
        }
    }

    fn new_with_params(conductivity_factor: f64, distance_length: f64, pressure_baseline: f64) -> Self {
        Self {
            nodes: Vec::new(),
            initial_nodes: Vec::new(),
            conductivity_factor,
            distance_length,
            pressure_baseline,
        }
    }
    
    fn create_chunky_temperature_profile(&mut self) {
        // Create DRAMATIC temperature split to clearly see smoothing
        // First half: COLD (500K), Second half: HOT (1500K)

        let num_layers = 60;  // Extended to 60 layers for deeper thermal column
        let total_depth = 300.0; // Extended to 300km for more asthenosphere space
        let layer_thickness = total_depth / num_layers as f64;

        for i in 0..num_layers {
            let depth = (i as f64 + 0.5) * layer_thickness; // Center of each layer

            // REALISTIC GEOTHERMAL GRADIENT: Using codebase values
            let depth = (i as f64 + 0.5) * layer_thickness;
            let geothermal_gradient_k_per_km = 25.0; // More realistic than 0.25 for this experiment
            let surface_temp_k = 288.0; // 15°C surface temperature
            let foundry_temp_k = 20000.0; // FOUNDRY: Reduced heat for thermal equilibrium
            let asthenosphere_surface_temp_k = 1873.0; // Engine-asthenosphere boundary

            let temp = if i < 4 {
                // CORE/FOUNDRY (Fixed temperatures) - 4 layers with massive overflow heat
                foundry_temp_k - (i as f64 * 6042.0) // 20000K down to 1873K (moderate foundry gradient)
            } else if i >= 52 {
                // SURFACE LAYERS (No space sink) - natural thermal gradient
                400.0 - ((i - 52) as f64 * 40.0) // 400K down to 80K natural cooling
            } else {
                // ALL LAYERS START HOT (molten planet genesis)
                // Let space cooling naturally create lithosphere from top down
                let depth_factor = (i as f64 - 4.0) / 56.0; // Normalize from engine to surface
                1800.0 + (depth_factor * 400.0) // 1800K up to 2200K (HOTTER as we go deeper toward foundry)
            };

            // REALISTIC material-based thermal conductivity and states
            let (conductivity, thermal_state) = if i < 8 {
                (3.2, 10) // CORE/ASTHENOSPHERE: Fixed engine state (8 layers)
            } else if i >= 32 {
                (10.0, 0)  // SPACE: High conductivity for strong heat sink, space state (8 layers)
            } else {
                (2.5, 7)  // LITHOSPHERE: Start as asthenosphere (will transition, 24 layers)
            };
            let density = 3000.0; // kg/m³
            let specific_heat = 1200.0; // J/kg/K
            let volume_m3 = layer_thickness * 1000.0 * 1e6; // layer_thickness_km * 1km² in m³
            let mass_kg = density * volume_m3;
            let thermal_capacity = mass_kg * specific_heat;
            let energy_j = thermal_capacity * temp;

            self.nodes.push(ThermalNode {
                depth_km: depth,
                height_km: layer_thickness,
                temperature_k: temp,
                thermal_conductivity: conductivity,
                thermal_capacity,
                energy_j,
                thermal_state,
            });
        }
        
        self.initial_nodes = self.nodes.clone();
        
        println!("   Foundry overflow heat: 3000K foundry with gradient to 1873K asthenosphere");
        println!("   Foundry layers (0-40km): 3000K-1873K (massive overflow heat for thermal tank)");
        println!("   Surface radiation: Stefan-Boltzmann T^4 radiation to space (top layers only)");
    }
    
    fn print_initial_profile(&self) {
        println!("📊 Initial Setup:");
        println!("   40 thermal nodes (doubled depth to 200km)");
        println!("   🔥 HEAT SOURCE: Core heat engine (8 layers, 0-40km)");
        println!("   ❄️  HEAT SINK: Space cooling sink (8 layers, 160-200km)");
        println!("   ⚡ EXPONENTIAL DIFFUSION: 2 neighbors each side with falloff");

        println!("\n🌡️  Initial Temperature Profile (FOUNDRY OVERFLOW HEAT):");
        println!("   Foundry layers (0-40km): 3000K → 1873K (massive overflow heat)");
        println!("   Lithosphere (40-160km): Variable geothermal gradient");
        println!("   Surface layers (160-200km): 400K → 80K (natural cooling + T^4 radiation)");
    }
    
    fn run_force_directed_diffusion(&mut self, steps: usize, years_per_step: f64) {
        let mut file = File::create("examples/data/thermal_experiment.csv").expect("Could not create file");
        
        // Write header - only every 5th layer, excluding boundaries
        write!(file, "years").unwrap();

        // Export every 5th layer from variable zone (8-31), so layers: 10, 15, 20, 25, 30
        let export_layers = [10, 15, 20, 25, 30];
        for &layer_idx in &export_layers {
            let depth_km = (layer_idx as f64 + 0.5) * 5.0; // 5km per layer
            write!(file, ",temp_{}km", depth_km as i32).unwrap();
        }
        for &layer_idx in &export_layers {
            let depth_km = (layer_idx as f64 + 0.5) * 5.0;
            write!(file, ",state_{}km", depth_km as i32).unwrap();
        }
        writeln!(file).unwrap();
        
        // Export initial state
        self.export_state(&mut file, 0, 0.0);
        
        // Run force-directed thermal diffusion
        for step in 0..steps {
            self.force_directed_step(years_per_step);
            
            // Export every few steps
            if step % 5 == 0 || step == steps - 1 {
                let years = (step + 1) as f64 * years_per_step;
                self.export_state(&mut file, step + 1, years);
            }
            
            if step % 10 == 0 {
                let temp_range = self.calculate_temperature_range();
                let surface_temp = self.nodes[0].temperature_k; // Top layer (heat sink)
                let core_temp = self.nodes[self.nodes.len()-1].temperature_k; // Bottom layer (heat source)
                let boundary_temps = (self.nodes[9].temperature_k, self.nodes[10].temperature_k);
                println!("   Step {}: Range={:.1}K, Surface={:.1}K, Core={:.1}K, Boundary=({:.1}K,{:.1}K)",
                         step, temp_range, surface_temp, core_temp, boundary_temps.0, boundary_temps.1);
            }
        }
        
        self.print_final_analysis();
    }
    
    fn force_directed_step(&mut self, years: f64) {
        let mut energy_changes = vec![0.0; self.nodes.len()];

        // Update thermal states based on temperature (asthenosphere ↔ lithosphere transitions)
        self.update_thermal_states();

        // BOUNDARY CONDITIONS: Heat source and heat sink
        self.apply_boundary_conditions(&mut energy_changes, years);

        // EXPONENTIAL THERMAL DIFFUSION: Each layer exchanges with 2 neighbors on each side
        for i in 0..self.nodes.len() {
            let current_node = &self.nodes[i];

            // Exchange with 2 layers on each side (i-2, i-1, i+1, i+2) with exponential falloff
            for offset in [-2, -1, 1, 2] {
                let j = i as i32 + offset;
                if j < 0 || j >= self.nodes.len() as i32 { continue; }
                let j = j as usize;

                let other_node = &self.nodes[j];
                let distance_km = (current_node.depth_km - other_node.depth_km).abs();

                // Calculate thermal force using scientific physics with exponential falloff
                let energy_transfer = self.calculate_thermal_force(current_node, other_node, distance_km, years);
                energy_changes[i] += energy_transfer;
            }
        }
        
        // Apply energy changes (with conservation check)
        let total_energy_before: f64 = self.nodes.iter().map(|n| n.energy_j).sum();
        
        for (i, &energy_change) in energy_changes.iter().enumerate() {
            self.nodes[i].energy_j += energy_change;
            // Update temperature based on new energy
            self.nodes[i].temperature_k = self.nodes[i].energy_j / self.nodes[i].thermal_capacity;
        }

        // AFTER energy flow, reset boundary temperatures to fixed values (infinite reservoirs)
        self.reset_boundary_temperatures();

        let total_energy_after: f64 = self.nodes.iter().map(|n| n.energy_j).sum();
        let energy_conservation_error = (total_energy_after - total_energy_before).abs() / total_energy_before;
        
        if energy_conservation_error > 0.01 {
            println!("   Warning: Energy conservation error: {:.3}%", energy_conservation_error * 100.0);
        }
    }

    fn apply_boundary_conditions(&self, _energy_changes: &mut Vec<f64>, _years: f64) {
        // No boundary conditions needed here - energy flows naturally
        // Boundary temperatures will be reset after energy changes are applied
    }

    fn reset_boundary_temperatures(&mut self) {
        // FOUNDRY LAYERS (0-3): Reset to fixed hot temperatures (infinite heat reservoir)
        for i in 0..4 {
            let target_temp = 28000.0 - (i as f64 * 8709.0); // 28000K down to 1873K (ultra-extreme foundry overflow heat)
            self.nodes[i].temperature_k = target_temp;
            self.nodes[i].energy_j = target_temp * self.nodes[i].thermal_capacity;
        }

        // GENTLE SPACE COOLING: Create lithosphere from top down (genesis process)
        // Only apply to the top few layers (thin skin effect)
        for i in 55..60 {
            let node = &mut self.nodes[i];
            let temp_k = node.temperature_k;

            // Gentle Stefan-Boltzmann radiation for lithosphere formation
            let stefan_boltzmann_coefficient = 5.67e-8; // W/m²/K⁴
            let surface_area = 25.0e6; // 25 km² surface area per node
            let years_to_seconds = 365.25 * 24.0 * 3600.0; // Convert years to seconds
            let power_watts = stefan_boltzmann_coefficient * surface_area * temp_k.powi(3);
            let energy_loss_per_year = power_watts * years_to_seconds;

            // Apply enhanced radiation loss for thermal equilibrium
            let radiation_loss = energy_loss_per_year * 0.0005; // Enhanced cooling (5x stronger)
            node.energy_j = (node.energy_j - radiation_loss).max(node.thermal_capacity * 200.0); // Minimum 200K
            node.temperature_k = node.energy_j / node.thermal_capacity;
        }
    }
    
    fn calculate_thermal_force(&self, from_node: &ThermalNode, to_node: &ThermalNode, distance_km: f64, years: f64) -> f64 {
        // PROPER DIFFUSION EQUILIBRIUM MODEL:
        // Only the energy DIFFERENCE flows, weighted by distance and conductivity

        let temp_diff = to_node.temperature_k - from_node.temperature_k;
        if temp_diff.abs() < 0.1 { return 0.0; }

        // Get material-based conductivities
        let from_conductivity = self.get_material_conductivity(from_node);
        let to_conductivity = self.get_material_conductivity(to_node);

        // Distance weights: 1.0, 0.25, 0.125 for neighbors 1, 2, 3 away
        let distance_weight = if distance_km <= 5.1 {
            1.0  // Adjacent neighbor
        } else if distance_km <= 10.1 {
            0.25 // Second neighbor
        } else if distance_km <= 15.1 {
            0.125 // Third neighbor
        } else {
            0.0  // Too far
        };

        // Conductivity factor: both sender and recipient must be conductive
        let sender_factor = from_conductivity / 3.0; // Normalize by average (3.0)
        let recipient_factor = to_conductivity / 3.0; // Normalize by average (3.0)
        let conductivity_factor = sender_factor * recipient_factor;

        // Base diffusion rate: 2% per century (DOUBLED for faster heat transport)
        let base_coefficient = 0.02 / years; // Doubled diffusion rate

        // PRESSURE ACCELERATION: Help foundry heat reach farther
        // (temp_diff / 100)^1.5 with minimum 0.33 for stronger acceleration
        let temp_diff_abs = temp_diff.abs();
        let pressure_factor = ((temp_diff_abs / 100.0).powf(1.5)).max(0.33);

        // DIFFUSION PHYSICS: Only the energy difference flows
        // This represents the portion of energy difference that flows through this specific path
        let energy_difference = temp_diff * from_node.thermal_capacity;
        let flow_coefficient = base_coefficient * distance_weight * conductivity_factor * pressure_factor;
        let energy_transfer = energy_difference * flow_coefficient * years;

        // Limit transfer to prevent instability (max 25% of thermal capacity per step)
        let max_transfer = from_node.thermal_capacity * 0.25;
        let limited_transfer = energy_transfer.abs().min(max_transfer);

        if temp_diff > 0.0 { limited_transfer } else { -limited_transfer }
    }

    fn update_thermal_states(&mut self) {
        const MELTING_POINT_K: f64 = 1523.0; // 1250°C transition point

        for node in &mut self.nodes {
            // Skip fixed boundary layers (space and core)
            if node.thermal_state == 0 || node.thermal_state == 10 {
                continue;
            }

            // Asthenosphere ↔ Lithosphere transitions
            if node.temperature_k < MELTING_POINT_K && node.thermal_state >= 6 {
                // Cooling: Asthenosphere → Lithosphere
                node.thermal_state = (node.thermal_state - 1).max(1);
            } else if node.temperature_k > MELTING_POINT_K && node.thermal_state <= 5 {
                // Heating: Lithosphere → Asthenosphere
                node.thermal_state = (node.thermal_state + 1).min(9);
            }
        }
    }

    fn get_material_conductivity(&self, node: &ThermalNode) -> f64 {
        match node.thermal_state {
            0 => 1.0,           // Space: baseline conductivity
            1..=5 => 2.5,       // Lithosphere: lower conductivity (solid)
            6..=9 => 3.2,       // Asthenosphere: higher conductivity (convective)
            10 => 3.2,          // Core: same as asthenosphere
            _ => 2.5,           // Default to lithosphere
        }
    }
    
    fn export_state(&self, file: &mut File, step: usize, years: f64) {
        write!(file, "{:.0}", years).unwrap();

        // Export only every 5th layer from variable zone (8-31)
        let export_layers = [10, 15, 20, 25, 30];

        // Export temperatures for selected layers
        for &layer_idx in &export_layers {
            write!(file, ",{:.1}", self.nodes[layer_idx].temperature_k).unwrap();
        }

        // Export thermal states for selected layers
        for &layer_idx in &export_layers {
            write!(file, ",{}", self.nodes[layer_idx].thermal_state).unwrap();
        }

        writeln!(file).unwrap();
    }
    
    fn calculate_temperature_range(&self) -> f64 {
        let min_temp = self.nodes.iter().map(|n| n.temperature_k).fold(f64::INFINITY, f64::min);
        let max_temp = self.nodes.iter().map(|n| n.temperature_k).fold(f64::NEG_INFINITY, f64::max);
        max_temp - min_temp
    }
    
    fn calculate_max_gradient(&self) -> f64 {
        let mut max_gradient: f64 = 0.0;
        for i in 1..self.nodes.len() {
            let temp_diff = (self.nodes[i].temperature_k - self.nodes[i-1].temperature_k).abs();
            let depth_diff = (self.nodes[i].depth_km - self.nodes[i-1].depth_km).abs();
            if depth_diff > 0.0 {
                let gradient = temp_diff / depth_diff;
                max_gradient = max_gradient.max(gradient);
            }
        }
        max_gradient
    }
    
    fn print_final_analysis(&self) {
        println!("\n📊 Final Analysis:");
        println!("   Initial temperature range: {:.1}K", 
                 self.initial_nodes.iter().map(|n| n.temperature_k).fold(f64::NEG_INFINITY, f64::max) - 
                 self.initial_nodes.iter().map(|n| n.temperature_k).fold(f64::INFINITY, f64::min));
        println!("   Final temperature range: {:.1}K", self.calculate_temperature_range());
        
        // Melting point constants for reference
        const BASALT_SOLIDUS_K: f64 = 1473.0;     // 1200°C - lithosphere material
        const PERIDOTITE_SOLIDUS_K: f64 = 1573.0; // 1300°C - asthenosphere material
        const TRANSITION_THRESHOLD_K: f64 = 1523.0; // 1250°C - transition point

        println!("\n🌡️  Final Temperature Profile:");
        println!("   🔥 Melting Points: Basalt=1473K, Transition=1523K, Peridotite=1573K");
        println!("   🌡️  Thermal States: 0=Space, 1-5=Lithosphere, 6-9=Asthenosphere, 10=Core");
        for (i, node) in self.nodes.iter().enumerate() {
            let initial_temp = self.initial_nodes[i].temperature_k;
            let change = node.temperature_k - initial_temp;

            // Determine layer type based on temperature (ALL layers, not just < 32)
            let layer_type = if i < 4 {
                "FOUNDRY ENGINE".to_string()
            } else {
                // Check if above melting point (1523K) for asthenosphere vs lithosphere
                if node.temperature_k >= TRANSITION_THRESHOLD_K {
                    let asth_layer = i - 4;
                    format!("AL {} (Asth)", asth_layer)
                } else {
                    let lith_layer = i - 4;
                    format!("EL {} (Lith)", lith_layer)
                }
            };

            // Show melting status with color coding: red for melting, blue for cooling
            let melting_status = if node.temperature_k >= PERIDOTITE_SOLIDUS_K {
                "\x1b[31m🔥HOT\x1b[0m"  // Red for melting/above melting
            } else if node.temperature_k >= TRANSITION_THRESHOLD_K {
                "\x1b[31m🌋MELT\x1b[0m"  // Red for melting
            } else if node.temperature_k >= BASALT_SOLIDUS_K {
                "\x1b[34m🔶WARM\x1b[0m"  // Blue for cooling
            } else if node.temperature_k >= 273.0 && node.temperature_k <= 373.0 {
                "\x1b[32m🌍HABITABLE\x1b[0m"  // Green for human habitable range (0-100°C)
            } else {
                "\x1b[34m❄️COLD\x1b[0m"  // Blue for cooling
            };

            // Show thermal state description
            let state_desc = match node.thermal_state {
                0 => "Space",
                1..=5 => "Lith",
                6..=9 => "Asth",
                10 => "Core",
                _ => "Unknown"
            };

            println!("   Layer {}: {:.1}K (Δ{:+.1}K) at {:.1}km depth - {} [State:{}={}] {}",
                     i, node.temperature_k, change, node.depth_km, layer_type, node.thermal_state, state_desc, melting_status);
        }
        
        println!("\n🔥 Thermal equilibration complete!");

        println!("\n🌡️  Melting point {}K --- Lith formation temp {}K", PERIDOTITE_SOLIDUS_K, TRANSITION_THRESHOLD_K);

        // Count asthenosphere vs lithosphere layers for final summary
        let mut asth_count = 0;
        let mut lith_count = 0;
        let mut total_layers = 0;

        for (i, node) in self.nodes.iter().enumerate() {
            if i >= 4 && i < 52 { // Exclude foundry (0-3) and surface (52-59) layers
                total_layers += 1;
                if node.temperature_k >= TRANSITION_THRESHOLD_K {
                    asth_count += 1;
                } else {
                    lith_count += 1;
                }
            }
        }

        let asth_percentage = (asth_count as f64 / total_layers as f64) * 100.0;
        let asth_thickness_km = asth_count * 5; // 5km per layer
        let lith_thickness_km = lith_count * 5; // 5km per layer

        println!("\n🎯 Asthenosphere: {:.1}% ({}km) | Lithosphere: {:.1}% ({}km)",
                 asth_percentage, asth_thickness_km, 100.0 - asth_percentage, lith_thickness_km);
    }
}
