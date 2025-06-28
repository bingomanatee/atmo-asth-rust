use atmo_asth_rust::constants::EARTH_RADIUS_KM;
use atmo_asth_rust::planet::Planet;
use atmo_asth_rust::sim::sim_op::{
    CoreRadianceOp, CsvWriterOp, LithosphereUnifiedOp,
    ThermalDiffusionOp,
};
use atmo_asth_rust::material::MaterialType;
use h3o::Resolution;
use std::fs;
use atmo_asth_rust::sim::simulation::{SimProps, Simulation};

/// Demonstration of how production rate modifier reduces chaos in lithosphere formation
/// 
/// This example runs two simulations side-by-side:
/// 1. High production rate (1.0) - shows chaotic behavior
/// 2. Low production rate (0.1) - shows stable behavior
/// 
/// The results are exported to CSV files for comparison.
fn main() {
    println!("🌍 Production Rate Modifier Demonstration");
    println!("=========================================");
    println!();
    
    // Ensure output directory exists
    fs::create_dir_all("examples/data").expect("Failed to create output directory");
    
    let csv_file_chaotic = "examples/data/production_rate_chaotic.csv".to_string();
    let csv_file_stable = "examples/data/production_rate_stable.csv".to_string();
    
    // Remove existing files
    let _ = fs::remove_file(&csv_file_chaotic);
    let _ = fs::remove_file(&csv_file_stable);
    
    println!("🔥 Running CHAOTIC simulation (production rate = 1.0)...");
    run_simulation("chaotic", csv_file_chaotic.clone(), 1.0);
    
    println!("\n🧊 Running STABLE simulation (production rate = 0.1)...");
    run_simulation("stable", csv_file_stable.clone(), 0.1);
    
    println!("\n📊 Comparison Results:");
    println!("======================");
    analyze_chaos_comparison(&csv_file_chaotic, &csv_file_stable);
}

fn run_simulation(name: &'static str, csv_file: String, production_rate: f64) {
    // Create operators with different production rates
    let operators = vec![
        CoreRadianceOp::handle_earth(), // Core heat input
        ThermalDiffusionOp::handle(1.0, 25.0), // Thermal diffusion
        LithosphereUnifiedOp::handle(
            vec![(MaterialType::Silicate, 1.0)], // Simple silicate lithosphere
            42,    // Random seed (same for both)
            0.1,   // Scale factor (same for both)
            production_rate, // THIS IS THE KEY DIFFERENCE
        ),
        CsvWriterOp::handle_with_layer_temps(csv_file.clone(), 3, 3), // 3 asth, 3 lith layers
    ];
    
    // Create simulation with high starting temperature to trigger chaos
    let sim_props = SimProps {
        name,
        planet: Planet {
            radius_km: EARTH_RADIUS_KM as f64,
            resolution: Resolution::Zero,
        },
        ops: operators,
        res: Resolution::Two, // Small grid for faster computation
        layer_count: 3,      // 3 asthenosphere layers
        asth_layer_height_km: 50.0,
        lith_layer_height_km: 25.0,
        sim_steps: 100,      // Run for 100 steps to see behavior
        years_per_step: 10000, // 10,000 years per step = 1M years total
        debug: false,
        alert_freq: 25,      // Report every 25 steps
        starting_surface_temp_k: 2000.0, // High starting temperature (above formation threshold)
    };
    
    println!("   📊 Parameters: {} steps × {} years = {:.1}M years",
             sim_props.sim_steps, sim_props.years_per_step,
             (sim_props.sim_steps as u32 * sim_props.years_per_step) / 1_000_000);
    println!("   🌡️  Starting temperature: {:.0}K (above formation threshold)", sim_props.starting_surface_temp_k);
    println!("   ⚙️  Production rate modifier: {:.1}", production_rate);
    
    let mut sim = Simulation::new(sim_props);
    sim.simulate();
    
    println!("   ✅ Simulation complete, data exported to: {}", csv_file);
}

fn analyze_chaos_comparison(chaotic_file: &str, stable_file: &str) {
    let chaotic_data = read_csv_temperatures(chaotic_file);
    let stable_data = read_csv_temperatures(stable_file);
    
    if chaotic_data.is_empty() || stable_data.is_empty() {
        println!("   ❌ Could not read CSV data for comparison");
        return;
    }
    
    // Calculate temperature variance (measure of chaos)
    let chaotic_variance = calculate_temperature_variance(&chaotic_data);
    let stable_variance = calculate_temperature_variance(&stable_data);
    
    // Calculate average temperatures
    let chaotic_avg = chaotic_data.iter().sum::<f64>() / chaotic_data.len() as f64;
    let stable_avg = stable_data.iter().sum::<f64>() / stable_data.len() as f64;
    
    println!("   🔥 CHAOTIC (rate=1.0):");
    println!("      - Average temperature: {:.1}K", chaotic_avg);
    println!("      - Temperature variance: {:.1}K² (chaos measure)", chaotic_variance);
    println!("      - Data points: {}", chaotic_data.len());
    
    println!("   🧊 STABLE (rate=0.1):");
    println!("      - Average temperature: {:.1}K", stable_avg);
    println!("      - Temperature variance: {:.1}K² (chaos measure)", stable_variance);
    println!("      - Data points: {}", stable_data.len());
    
    let chaos_reduction = ((chaotic_variance - stable_variance) / chaotic_variance) * 100.0;
    
    println!("\n   📈 CHAOS REDUCTION:");
    if chaos_reduction > 0.0 {
        println!("      ✅ {:.1}% reduction in temperature variance", chaos_reduction);
        println!("      ✅ Lower production rate successfully reduces chaos!");
    } else {
        println!("      ⚠️  No significant chaos reduction detected");
    }
    
    println!("\n   📁 Open the CSV files in a spreadsheet to visualize:");
    println!("      - {}", chaotic_file);
    println!("      - {}", stable_file);
    println!("   📊 Plot 'avg_asth_layer_0_temp_k' to see the difference in stability");
}

fn read_csv_temperatures(file_path: &str) -> Vec<f64> {
    let mut temperatures = Vec::new();
    
    if let Ok(content) = fs::read_to_string(file_path) {
        let lines: Vec<&str> = content.lines().collect();
        if lines.len() < 2 {
            return temperatures;
        }
        
        // Find the column index for surface temperature
        let header = lines[0];
        let columns: Vec<&str> = header.split(',').collect();
        let temp_col_index = columns.iter().position(|&col| col == "avg_asth_layer_0_temp_k");
        
        if let Some(col_idx) = temp_col_index {
            for line in lines.iter().skip(1) {
                let values: Vec<&str> = line.split(',').collect();
                if let Some(temp_str) = values.get(col_idx) {
                    if let Ok(temp) = temp_str.parse::<f64>() {
                        temperatures.push(temp);
                    }
                }
            }
        }
    }
    
    temperatures
}

fn calculate_temperature_variance(temperatures: &[f64]) -> f64 {
    if temperatures.len() < 2 {
        return 0.0;
    }
    
    let mean = temperatures.iter().sum::<f64>() / temperatures.len() as f64;
    let variance = temperatures.iter()
        .map(|temp| (temp - mean).powi(2))
        .sum::<f64>() / temperatures.len() as f64;
    
    variance
}
