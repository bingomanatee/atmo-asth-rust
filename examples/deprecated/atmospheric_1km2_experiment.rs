#![allow(dead_code)]
#![allow(unused_variables)]

use std::fs::File;
use std::io::Write;

// Import the atmospheric mass functions and EnergyMass trait
extern crate atmo_asth_rust;
use atmo_asth_rust::energy_mass::{
    EnergyMass, 
    StandardEnergyMass, 
    create_atmospheric_layer_simple_decay,
    create_atmospheric_column_simple_decay,
    create_atmospheric_layer_for_cell
};
use atmo_asth_rust::material::MaterialType;

/// Thermal node with atmospheric integration
#[derive(Clone, Debug)]
struct ThermalNode {
    energy_mass: StandardEnergyMass,
    height_km: f64,
    is_atmospheric: bool,
    atmospheric_layer_index: Option<usize>,
}

impl ThermalNode {
    fn new_solid(material_type: MaterialType, temperature_k: f64, volume_km3: f64, height_km: f64) -> Self {
        Self {
            energy_mass: StandardEnergyMass::new_with_material(material_type, temperature_k, volume_km3),
            height_km,
            is_atmospheric: false,
            atmospheric_layer_index: None,
        }
    }

    fn new_atmospheric(layer_index: usize, height_km: f64, temperature_k: f64) -> Self {
        let atmospheric_layer = create_atmospheric_layer_simple_decay(layer_index, height_km, temperature_k);
        
        // Convert to StandardEnergyMass for compatibility
        let energy_mass = StandardEnergyMass::new_with_material(
            MaterialType::Air,
            temperature_k,
            atmospheric_layer.volume(),
        );

        Self {
            energy_mass,
            height_km,
            is_atmospheric: true,
            atmospheric_layer_index: Some(layer_index),
        }
    }

    fn temperature_k(&self) -> f64 {
        self.energy_mass.kelvin()
    }

    fn volume_km3(&self) -> f64 {
        self.energy_mass.volume()
    }

    fn mass_kg(&self) -> f64 {
        self.energy_mass.mass_kg()
    }

    fn add_energy(&mut self, energy_j: f64) {
        self.energy_mass.add_energy(energy_j);
    }

    fn remove_energy(&mut self, energy_j: f64) {
        if energy_j > 0.0 && energy_j <= self.energy_mass.energy() {
            self.energy_mass.remove_energy(energy_j);
        }
    }
}

/// 1km² thermal experiment with atmospheric formation
struct Atmospheric1km2Experiment {
    solid_nodes: Vec<ThermalNode>,
    atmospheric_nodes: Vec<ThermalNode>,
    layer_height_km: f64,
    surface_area_km2: f64,
    outgassing_threshold_k: f64,
    max_atmospheric_layers: usize,
}

impl Atmospheric1km2Experiment {
    fn new() -> Self {
        let layer_height_km = 5.0; // 5km per layer
        let surface_area_km2 = 1.0; // 1 km²
        let num_solid_layers = 40; // 200km depth total
        
        // Create solid thermal layers (asthenosphere/lithosphere)
        let mut solid_nodes = Vec::new();
        for i in 0..num_solid_layers {
            let depth_km = (i as f64 + 0.5) * layer_height_km;
            let temperature_k = 300.0 + depth_km * 25.0; // Geothermal gradient
            let volume_km3 = surface_area_km2 * layer_height_km;
            
            let node = ThermalNode::new_solid(
                MaterialType::Silicate,
                temperature_k,
                volume_km3,
                layer_height_km,
            );
            solid_nodes.push(node);
        }

        Self {
            solid_nodes,
            atmospheric_nodes: Vec::new(),
            layer_height_km,
            surface_area_km2,
            outgassing_threshold_k: 1400.0, // Temperature for outgassing
            max_atmospheric_layers: 20, // Up to 100km atmosphere (20 × 5km)
        }
    }

    fn print_initial_state(&self) {
        println!("🌍 Initial Thermal Profile:");
        println!("   Solid layers: {}", self.solid_nodes.len());
        println!("   Atmospheric layers: {}", self.atmospheric_nodes.len());
        println!("   Layer height: {:.1} km", self.layer_height_km);
        println!("   Surface area: {:.1} km²", self.surface_area_km2);
        
        // Show temperature profile for first few and last few layers
        println!("\n📊 Temperature Profile (solid layers):");
        for (i, node) in self.solid_nodes.iter().take(5).enumerate() {
            let depth_km = (i as f64 + 0.5) * self.layer_height_km;
            println!("   Layer {}: {:.1}km depth, {:.1}K", i, depth_km, node.temperature_k());
        }
        println!("   ...");
        let total_layers = self.solid_nodes.len();
        for (i, node) in self.solid_nodes.iter().skip(total_layers - 3).enumerate() {
            let actual_i = total_layers - 3 + i;
            let depth_km = (actual_i as f64 + 0.5) * self.layer_height_km;
            println!("   Layer {}: {:.1}km depth, {:.1}K", actual_i, depth_km, node.temperature_k());
        }
    }

    fn check_atmospheric_formation(&mut self, years_per_step: f64) -> f64 {
        let mut total_outgassing_kg = 0.0;

        // Check each solid layer for outgassing
        for (i, node) in self.solid_nodes.iter().enumerate() {
            if node.temperature_k() > self.outgassing_threshold_k {
                // Calculate outgassing rate based on temperature excess
                let temp_excess = node.temperature_k() - self.outgassing_threshold_k;
                let outgassing_rate_kg_per_year = temp_excess * node.volume_km3() * 1e6; // Simplified model
                let outgassed_mass = outgassing_rate_kg_per_year * years_per_step;
                total_outgassing_kg += outgassed_mass;
            }
        }

        // Convert outgassed mass to atmospheric layers using 0.88 decay model
        if total_outgassing_kg > 0.0 && self.atmospheric_nodes.len() < self.max_atmospheric_layers {
            self.add_atmospheric_layer(total_outgassing_kg);
        }

        total_outgassing_kg
    }

    fn add_atmospheric_layer(&mut self, outgassed_mass_kg: f64) {
        let layer_index = self.atmospheric_nodes.len();
        let surface_temp = if let Some(surface_node) = self.solid_nodes.first() {
            surface_node.temperature_k()
        } else {
            300.0 // Default surface temperature
        };

        // Calculate atmospheric temperature (cooler than surface)
        let atmospheric_temp = surface_temp - (layer_index as f64 * 6.5); // Lapse rate

        let atmospheric_node = ThermalNode::new_atmospheric(
            layer_index,
            self.layer_height_km,
            atmospheric_temp,
        );

        self.atmospheric_nodes.push(atmospheric_node);
    }

    fn apply_core_heat(&mut self, years_per_step: f64) {
        // Apply 4x core heat to bottom layer (270 mW/m²)
        let core_heat_flux_j_per_km2_per_year = 10.08e12; // 4x Earth's core heat
        let core_energy = core_heat_flux_j_per_km2_per_year * self.surface_area_km2 * years_per_step;

        if let Some(bottom_node) = self.solid_nodes.last_mut() {
            bottom_node.add_energy(core_energy);
        }
    }

    fn apply_surface_cooling(&mut self, years_per_step: f64) {
        // Apply Stefan-Boltzmann cooling to surface
        if let Some(surface_node) = self.solid_nodes.first_mut() {
            let surface_temp = surface_node.temperature_k();
            let stefan_boltzmann_constant = 5.67e-8; // W/(m²·K⁴)
            let cooling_power = stefan_boltzmann_constant * surface_temp.powi(4); // W/m²
            let cooling_energy = cooling_power * 1e6 * 365.25 * 24.0 * 3600.0 * years_per_step; // Convert to J/km²/year
            
            surface_node.remove_energy(cooling_energy);
        }
    }

    fn thermal_diffusion_step(&mut self, years_per_step: f64) {
        // Simple thermal diffusion between adjacent solid layers
        let diffusion_rate = 0.01; // 1% energy transfer per step
        
        for i in 0..self.solid_nodes.len() - 1 {
            let temp_diff = self.solid_nodes[i + 1].temperature_k() - self.solid_nodes[i].temperature_k();
            let energy_transfer = temp_diff * diffusion_rate * self.solid_nodes[i].energy_mass.energy();
            
            if energy_transfer > 0.0 {
                self.solid_nodes[i + 1].remove_energy(energy_transfer);
                self.solid_nodes[i].add_energy(energy_transfer);
            } else if energy_transfer < 0.0 {
                self.solid_nodes[i].remove_energy(-energy_transfer);
                self.solid_nodes[i + 1].add_energy(-energy_transfer);
            }
        }
    }

    fn run_experiment(&mut self, total_years: f64, years_per_step: f64) {
        let steps = (total_years / years_per_step) as usize;
        
        // Create CSV file for results
        let mut file = File::create("../data/atmospheric_1km2_experiment.csv")
            .expect("Could not create CSV file");
        
        // Write CSV header
        write!(file, "years,surface_temp_k,bottom_temp_k,atmospheric_layers,total_atm_mass_kg").unwrap();
        for i in 0..5 {
            write!(file, ",layer_{}_temp_k", i).unwrap();
        }
        writeln!(file).unwrap();

        println!("\n🚀 Running atmospheric formation experiment...");
        println!("   Total duration: {:.0} years", total_years);
        println!("   Time step: {:.0} years", years_per_step);
        println!("   Total steps: {}", steps);

        for step in 0..steps {
            let current_years = step as f64 * years_per_step;

            // Apply thermal processes
            self.apply_core_heat(years_per_step);
            self.thermal_diffusion_step(years_per_step);
            self.apply_surface_cooling(years_per_step);

            // Check for atmospheric formation
            let outgassing_kg = self.check_atmospheric_formation(years_per_step);

            // Export data every 1000 steps
            if step % 1000 == 0 {
                self.export_step(&mut file, current_years);
                
                if step % 10000 == 0 {
                    println!("   Step {}: {:.0} years, Surface: {:.1}K, Atmosphere: {} layers",
                             step, current_years, 
                             self.solid_nodes.first().map_or(0.0, |n| n.temperature_k()),
                             self.atmospheric_nodes.len());
                }
            }
        }

        println!("\n✅ Experiment complete!");
        println!("   Final atmospheric layers: {}", self.atmospheric_nodes.len());
        println!("   Results saved to: examples/data/atmospheric_1km2_experiment.csv");

        // Final detailed row-by-row summary
        self.print_final_summary();
    }

    fn export_step(&self, file: &mut File, years: f64) {
        let surface_temp = self.solid_nodes.first().map_or(0.0, |n| n.temperature_k());
        let bottom_temp = self.solid_nodes.last().map_or(0.0, |n| n.temperature_k());
        let total_atm_mass: f64 = self.atmospheric_nodes.iter().map(|n| n.mass_kg()).sum();

        write!(file, "{:.0},{:.1},{:.1},{},{:.2e}",
               years, surface_temp, bottom_temp, self.atmospheric_nodes.len(), total_atm_mass).unwrap();

        // Export first 5 layer temperatures
        for i in 0..5 {
            if i < self.solid_nodes.len() {
                write!(file, ",{:.1}", self.solid_nodes[i].temperature_k()).unwrap();
            } else {
                write!(file, ",0.0").unwrap();
            }
        }
        writeln!(file).unwrap();
    }

    fn print_final_summary(&self) {
        println!("\n🌡️  Final Thermal Profile Summary");
        println!("=====================================");

        // Atmospheric layers summary
        if !self.atmospheric_nodes.is_empty() {
            println!("\n🌫️  Atmospheric Layers (0.88 Decay Model):");
            let total_atm_mass: f64 = self.atmospheric_nodes.iter().map(|n| n.mass_kg()).sum();
            println!("   Total atmospheric mass: {:.2e} kg", total_atm_mass);
            println!("   Total atmospheric height: {:.1} km", self.atmospheric_nodes.len() as f64 * self.layer_height_km);

            for (i, node) in self.atmospheric_nodes.iter().enumerate() {
                let height_km = (i as f64 + 0.5) * self.layer_height_km;
                println!("   Layer {}: {:.1}km height, {:.1}K, {:.2e} kg, {:.3} kg/m³",
                         i, height_km, node.temperature_k(), node.mass_kg(), node.energy_mass.density_kgm3());
            }
        } else {
            println!("\n🌫️  No atmospheric layers formed");
        }

        // Solid layers summary
        println!("\n🪨 Solid Layers (Thermal Profile):");
        println!("   Total solid layers: {}", self.solid_nodes.len());

        // Count layers by temperature ranges
        let mut hot_layers = 0;  // >1400K (outgassing)
        let mut warm_layers = 0; // 800-1400K
        let mut cool_layers = 0; // <800K

        for node in &self.solid_nodes {
            let temp = node.temperature_k();
            if temp > self.outgassing_threshold_k {
                hot_layers += 1;
            } else if temp > 800.0 {
                warm_layers += 1;
            } else {
                cool_layers += 1;
            }
        }

        println!("   Hot layers (>{}K): {} ({:.1}%)",
                 self.outgassing_threshold_k, hot_layers,
                 (hot_layers as f64 / self.solid_nodes.len() as f64) * 100.0);
        println!("   Warm layers (800-{}K): {} ({:.1}%)",
                 self.outgassing_threshold_k, warm_layers,
                 (warm_layers as f64 / self.solid_nodes.len() as f64) * 100.0);
        println!("   Cool layers (<800K): {} ({:.1}%)",
                 cool_layers, (cool_layers as f64 / self.solid_nodes.len() as f64) * 100.0);

        // Show detailed breakdown of first few and last few layers
        println!("\n📊 Detailed Layer Breakdown:");
        println!("   Surface layers:");
        for (i, node) in self.solid_nodes.iter().take(3).enumerate() {
            let depth_km = (i as f64 + 0.5) * self.layer_height_km;
            println!("     Layer {}: {:.1}km depth, {:.1}K", i, depth_km, node.temperature_k());
        }

        println!("   ...");

        println!("   Deep layers:");
        let total_layers = self.solid_nodes.len();
        for (i, node) in self.solid_nodes.iter().skip(total_layers - 3).enumerate() {
            let actual_i = total_layers - 3 + i;
            let depth_km = (actual_i as f64 + 0.5) * self.layer_height_km;
            println!("     Layer {}: {:.1}km depth, {:.1}K", actual_i, depth_km, node.temperature_k());
        }

        // Summary statistics
        let surface_temp = self.solid_nodes.first().map_or(0.0, |n| n.temperature_k());
        let bottom_temp = self.solid_nodes.last().map_or(0.0, |n| n.temperature_k());
        let avg_temp: f64 = self.solid_nodes.iter().map(|n| n.temperature_k()).sum::<f64>() / self.solid_nodes.len() as f64;

        println!("\n🎯 Final Statistics:");
        println!("   Surface temperature: {:.1}K", surface_temp);
        println!("   Bottom temperature: {:.1}K", bottom_temp);
        println!("   Average temperature: {:.1}K", avg_temp);
        println!("   Temperature gradient: {:.2}K/km", (bottom_temp - surface_temp) / (self.solid_nodes.len() as f64 * self.layer_height_km));

        if !self.atmospheric_nodes.is_empty() {
            let total_atm_mass: f64 = self.atmospheric_nodes.iter().map(|n| n.mass_kg()).sum();
            println!("   Atmospheric mass: {:.2e} kg ({:.1}% of max)",
                     total_atm_mass, (total_atm_mass / 1.0e10) * 100.0);
        }
    }
}

fn main() {
    println!("🌍 Atmospheric Formation Experiment (1km²)");
    println!("==========================================");
    println!("🎯 4x Core Heat + 0.88 Decay Atmospheric Model");

    let mut experiment = Atmospheric1km2Experiment::new();
    experiment.print_initial_state();

    // Run 1 billion year simulation with 1000-year steps
    experiment.run_experiment(1_000_000_000.0, 1000.0);
}
