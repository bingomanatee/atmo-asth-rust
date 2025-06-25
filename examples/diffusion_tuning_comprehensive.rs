// Comprehensive diffusion tuning test
// Tests different thermal diffusion percentages to find optimal equilibrium

use atmo_asth_rust::sim::{Simulation, SimProps};
use atmo_asth_rust::sim::sim_op::{CoolingOp, RadianceOp, LithosphereUnifiedOp, CsvWriterOp};
use atmo_asth_rust::planet::Planet;
use atmo_asth_rust::constants::EARTH_RADIUS_KM;
use atmo_asth_rust::material::MaterialType;
use h3o::Resolution;

const SIM_STEPS: i32 = 2000;

fn main() {
    println!("🔬 Comprehensive Thermal Diffusion Tuning");
    println!("==========================================");
    println!("Testing different mixing percentages for 1000 steps each:");
    println!("Finding optimal thermal equilibrium through energy redistribution");

    // Create diffusion directory for CSV files
    std::fs::create_dir_all("diffusion").expect("Failed to create diffusion directory");
    println!("mixing rates from 1%..15%");
    
    // Test different (mixing_rate, cooling_rate) combinations
    let parameter_combinations = vec![
        (0.10, 1.0),    // 10% mixing, 1.0x cooling
        (0.15, 1.0),   // 15% mixing, 1.0x cooling
        (0.20, 1.0),   // 25% mixing, 1.0x cooling
        (0.025, 0.5),    // 30%I adjusted  mixing, 1.0x cooling
        (0.10, 0.5),   // 33% mixing, 1.0x cooling
        (0.15, 0.5),   // 15% mixing, 1.5x cooling
        (0.20, 0.5),   // 25% mixing, 1.5x cooling
    ];

    println!("\n🎯 Testing (mixing%, cooling_scale) combinations:");
    for &(mixing, cooling) in &parameter_combinations {
        println!("   ({:.1}%, {:.1}x) = {:.3} mixing, {:.1}x cooling",
                mixing * 100.0, cooling, mixing, cooling);
    }
    
    let mut results = Vec::new();

    for &(mixing_rate, cooling_scale) in &parameter_combinations {
        println!("\n{}", "=".repeat(80));
        println!("🌡️  Testing: {:.1}% mixing, {:.1}x cooling", mixing_rate * 100.0, cooling_scale);
        println!("Running {} simulation steps...", SIM_STEPS);

        let result = run_parameter_test(mixing_rate, cooling_scale);
        results.push(((mixing_rate, cooling_scale), result.clone()));
        
        println!("📊 Results after {} steps:", SIM_STEPS);
        println!("   Surface Temp: {:.0}K → {:.0}K (Δ{:+.0}K)", 
                result.initial_surface_temp, result.final_surface_temp, 
                result.final_surface_temp - result.initial_surface_temp);
        println!("   Deep Temp: {:.0}K → {:.0}K (Δ{:+.0}K)", 
                result.initial_deep_temp, result.final_deep_temp,
                result.final_deep_temp - result.initial_deep_temp);
        println!("   Temp Gradient: {:.0}K → {:.0}K (deep - surface)", 
                result.initial_gradient, result.final_gradient);
        println!("   Total Energy: {:.2e}J → {:.2e}J", 
                result.initial_energy, result.final_energy);
        println!("   Lithosphere: {:.1}km total ({:.1}km avg)", 
                result.final_lithosphere, result.final_lithosphere / 5882.0);
        
        // Evaluate the result
        evaluate_result(&result, mixing_rate, cooling_scale);
    }
    
    println!("\n{}", "=".repeat(80));
    println!("🏆 COMPREHENSIVE ANALYSIS");
    println!("{}", "=".repeat(80));
    
    println!("\n📊 Summary Table:");
    println!("Mix%/Cool | Surface T | Deep T | Gradient | Lithosphere | Evaluation");
    println!("----------|-----------|--------|----------|-------------|------------");

    for ((mixing_rate, cooling_scale), result) in &results {
        let eval = get_evaluation_summary(&result);
        println!("{:4.1}%/{:.1}x | {:7.0}K | {:6.0}K | {:6.0}K | {:9.1}km | {}",
                mixing_rate * 100.0,
                cooling_scale,
                result.final_surface_temp,
                result.final_deep_temp,
                result.final_gradient,
                result.final_lithosphere / 5882.0,
                eval);
    }
    
    println!("\n🎯 Optimal Diffusion Recommendations:");
    
    // Find best results
    let best_temp_stability = results.iter()
        .min_by(|a, b| {
            let a_temp_change = (a.1.final_surface_temp - 1873.15).abs();
            let b_temp_change = (b.1.final_surface_temp - 1873.15).abs();
            a_temp_change.partial_cmp(&b_temp_change).unwrap()
        });
    
    let best_gradient = results.iter()
        .filter(|(_, r)| r.final_gradient > 0.0)
        .max_by(|a, b| a.1.final_gradient.partial_cmp(&b.1.final_gradient).unwrap());
    
    let best_energy_conservation = results.iter()
        .min_by(|a, b| {
            let a_change = (a.1.final_energy / a.1.initial_energy - 1.0).abs();
            let b_change = (b.1.final_energy / b.1.initial_energy - 1.0).abs();
            a_change.partial_cmp(&b_change).unwrap()
        });
    
    if let Some(((mixing, cooling), _)) = best_temp_stability {
        println!("   🌡️  Best Temperature Stability: {:.1}% mixing, {:.1}x cooling (closest to 1873K formation threshold)", mixing * 100.0, cooling);
    }

    if let Some(((mixing, cooling), _)) = best_gradient {
        println!("   📈 Best Temperature Gradient: {:.1}% mixing, {:.1}x cooling (maintains deep > surface)", mixing * 100.0, cooling);
    }

    if let Some(((mixing, cooling), _)) = best_energy_conservation {
        println!("   ⚡ Best Energy Conservation: {:.1}% mixing, {:.1}x cooling (minimal energy change)", mixing * 100.0, cooling);
    }
    
    println!("\n💡 Key Insights:");
    println!("   • Lower diffusion = better temperature gradient maintenance");
    println!("   • Higher diffusion = faster equilibration but potential instability");
    println!("   • Optimal range likely between gradients that maintain physics");
    println!("   • Look for: stable temps near 1873K + positive gradient + lithosphere cycling");

    println!("\n📊 CSV Files Generated for Graphing:");
    for &(mixing_rate, cooling_scale) in &parameter_combinations {
        println!("   • diffusion/mix{:.1}_cool{:.1}.csv - Surface temp & lithosphere data over time",
                mixing_rate * 100.0, cooling_scale);
    }

    println!("\n📈 Graphing Recommendations:");
    println!("   • Plot surface temperature vs time for each diffusion %");
    println!("   • Plot average lithosphere thickness vs time for each diffusion %");
    println!("   • Compare equilibrium values across different diffusion rates");
    println!("   • Look for oscillations around 1873K formation threshold");
    println!("   • Identify which diffusion % gives stable lithosphere cycling");
}

#[derive(Debug, Clone)]
struct DiffusionResult {
    initial_surface_temp: f64,
    final_surface_temp: f64,
    initial_deep_temp: f64,
    final_deep_temp: f64,
    initial_gradient: f64,
    final_gradient: f64,
    initial_energy: f64,
    final_energy: f64,
    final_lithosphere: f64,
}

fn run_parameter_test(mixing_rate: f64, cooling_scale: f64) -> DiffusionResult {
    let mut sim = Simulation::new(SimProps {
        name: "diffusion_tuning",
        planet: Planet {
            radius_km: EARTH_RADIUS_KM as f64,
            resolution: Resolution::One,
        },
        ops: vec![
            // Add CSV writer for each parameter test - save to diffusion subfolder
            CsvWriterOp::handle(format!("diffusion/mix{:.1}_cool{:.1}.csv", mixing_rate * 100.0, cooling_scale)),
            CoolingOp::handle(cooling_scale),
            RadianceOp::handle_with_mixing(mixing_rate),
            LithosphereUnifiedOp::handle(
                vec![(MaterialType::Silicate, 1.0)],
                42,
                0.1,
            ),
        ],
        res: Resolution::Two,
        layer_count: 4,
        layer_height_km: 10.0,
        sim_steps: SIM_STEPS,  // Long simulation for equilibrium
        years_per_step: 10_000, // 10,000 years per step
        debug: false,
        alert_freq: 200,
        starting_surface_temp_k: 2000.0,
    });

    // Record initial state
    let initial_surface_temp = sim.cells.values()
        .filter_map(|column| column.layers.first())
        .map(|layer| layer.kelvin())
        .sum::<f64>() / sim.cells.len() as f64;
    
    let initial_deep_temp = sim.cells.values()
        .filter_map(|column| column.layers.last())
        .map(|layer| layer.kelvin())
        .sum::<f64>() / sim.cells.len() as f64;
    
    let initial_gradient = initial_deep_temp - initial_surface_temp;
    let initial_energy = sim.energy_at_layer(0);
    
    // Run simulation
    sim.simulate();
    
    // Record final state
    let final_surface_temp = sim.cells.values()
        .filter_map(|column| column.layers.first())
        .map(|layer| layer.kelvin())
        .sum::<f64>() / sim.cells.len() as f64;
    
    let final_deep_temp = sim.cells.values()
        .filter_map(|column| column.layers.last())
        .map(|layer| layer.kelvin())
        .sum::<f64>() / sim.cells.len() as f64;
    
    let final_gradient = final_deep_temp - final_surface_temp;
    let final_energy = sim.energy_at_layer(0);
    
    let final_lithosphere = sim.cells.values()
        .map(|column| column.total_lithosphere_height())
        .sum::<f64>();
    
    DiffusionResult {
        initial_surface_temp,
        final_surface_temp,
        initial_deep_temp,
        final_deep_temp,
        initial_gradient,
        final_gradient,
        initial_energy,
        final_energy,
        final_lithosphere,
    }
}

fn evaluate_result(result: &DiffusionResult, mixing_rate: f64, cooling_scale: f64) {
    println!("\n🎯 Evaluation for {:.1}% mixing, {:.1}x cooling:", mixing_rate * 100.0, cooling_scale);
    
    // Temperature stability (near formation threshold)
    let formation_temp = 1873.15;
    let temp_diff = (result.final_surface_temp - formation_temp).abs();
    if temp_diff < 100.0 {
        println!("   ✅ EXCELLENT: Surface temperature very close to formation threshold!");
    } else if temp_diff < 500.0 {
        println!("   ✓ GOOD: Surface temperature reasonably close to formation threshold");
    } else if result.final_surface_temp > 5000.0 {
        println!("   ❌ TOO HOT: Surface temperature extremely high");
    } else if result.final_surface_temp < 1000.0 {
        println!("   ❌ TOO COLD: Surface temperature very low");
    } else {
        println!("   ⚠️  MARGINAL: Surface temperature somewhat extreme");
    }
    
    // Temperature gradient
    if result.final_gradient > 50.0 {
        println!("   ✅ EXCELLENT: Strong positive temperature gradient maintained");
    } else if result.final_gradient > 0.0 {
        println!("   ✓ GOOD: Positive temperature gradient maintained");
    } else {
        println!("   ❌ GRADIENT ISSUE: Negative or zero temperature gradient");
    }
    
    // Energy conservation
    let energy_change_ratio = result.final_energy / result.initial_energy;
    if energy_change_ratio > 0.8 && energy_change_ratio < 1.2 {
        println!("   ✅ EXCELLENT: Good energy conservation");
    } else if energy_change_ratio > 0.5 && energy_change_ratio < 2.0 {
        println!("   ✓ GOOD: Reasonable energy conservation");
    } else {
        println!("   ⚠️  Energy conservation concern (ratio: {:.2})", energy_change_ratio);
    }
    
    // Lithosphere formation
    let avg_lithosphere = result.final_lithosphere / 5882.0;
    if avg_lithosphere > 0.1 && avg_lithosphere < 90.0 {
        println!("   ✓ Lithosphere formation active");
    } else if avg_lithosphere >= 90.0 {
        println!("   ⚠️  Lithosphere at material limit");
    } else {
        println!("   ⚠️  No significant lithosphere formation");
    }
}

fn get_evaluation_summary(result: &DiffusionResult) -> &'static str {
    let formation_temp = 1873.15;
    let temp_diff = (result.final_surface_temp - formation_temp).abs();
    let has_good_gradient = result.final_gradient > 0.0;
    let energy_ratio = result.final_energy / result.initial_energy;
    let good_energy = energy_ratio > 0.5 && energy_ratio < 2.0;
    
    if temp_diff < 100.0 && has_good_gradient && good_energy {
        "EXCELLENT"
    } else if temp_diff < 500.0 && has_good_gradient {
        "GOOD"
    } else if has_good_gradient {
        "MARGINAL"
    } else {
        "POOR"
    }
}
