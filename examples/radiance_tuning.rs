// Radiance tuning tool to find optimal thermal balance
// Tests different radiance values to achieve sensible equilibrium

use atmo_asth_rust::sim::{Simulation, SimProps};
use atmo_asth_rust::sim::sim_op::{CoolingOp, RadianceOp, LithosphereUnifiedOp};
use atmo_asth_rust::planet::Planet;
use atmo_asth_rust::constants::EARTH_RADIUS_KM;
use atmo_asth_rust::material::MaterialType;
use h3o::Resolution;

fn main() {
    println!("🔧 Radiance Tuning Tool");
    println!("=======================");
    println!("Finding optimal radiance for sensible thermal equilibrium");
    println!("Target: System oscillates around 1873K formation threshold");
    
    // Test different radiance multipliers
    let radiance_multipliers = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    
    println!("\n🎯 Testing radiance multipliers:");
    for &multiplier in &radiance_multipliers {
        println!("   {:.1}x = {:.2e} J/year global", multiplier, 1.3e21 * multiplier);
    }
    
    for &multiplier in &radiance_multipliers {
        println!("\n{}", "=".repeat(60));
        println!("🔥 Testing Radiance Multiplier: {:.1}x", multiplier);
        println!("   Global Radiance: {:.2e} J/year", 1.3e21 * multiplier);
        
        let result = test_radiance_balance(multiplier);
        
        println!("📊 Results:");
        println!("   Initial Temp: {:.0}K → Final Temp: {:.0}K", result.initial_temp, result.final_temp);
        println!("   Initial Energy: {:.2e}J → Final Energy: {:.2e}J", result.initial_energy, result.final_energy);
        println!("   Lithosphere: {:.1}km total ({:.3}km avg)", result.final_lithosphere, result.final_lithosphere / 5882.0);
        
        // Evaluate the result
        let temp_change = result.final_temp - result.initial_temp;
        let near_formation_temp = (result.final_temp - 1873.15).abs() < 200.0;
        let reasonable_temp = result.final_temp > 1000.0 && result.final_temp < 5000.0;
        let has_lithosphere = result.final_lithosphere > 0.0;
        
        println!("🎯 Evaluation:");
        if reasonable_temp && near_formation_temp {
            println!("   ✅ EXCELLENT: Temperature near formation threshold!");
        } else if reasonable_temp {
            println!("   ✓ GOOD: Reasonable temperature range");
        } else if result.final_temp > 10000.0 {
            println!("   ❌ TOO HOT: Radiance too high");
        } else if result.final_temp < 500.0 {
            println!("   ❌ TOO COLD: Radiance too low");
        } else {
            println!("   ⚠️  MARGINAL: Temperature somewhat extreme");
        }
        
        if has_lithosphere {
            println!("   ✓ Lithosphere present");
        } else {
            println!("   ⚠️  No lithosphere formed");
        }
        
        println!("   Temperature change: {:+.0}K", temp_change);
    }
    
    println!("\n{}", "=".repeat(60));
    println!("🎯 Tuning Recommendations:");
    println!("Look for multipliers that give:");
    println!("   • Final temperature 1500-2500K (near formation threshold)");
    println!("   • Some lithosphere formation");
    println!("   • Stable energy balance (not runaway heating/cooling)");
    println!("   • Temperature oscillation around 1873K");
}

#[derive(Debug)]
struct TuningResult {
    initial_temp: f64,
    final_temp: f64,
    initial_energy: f64,
    final_energy: f64,
    final_lithosphere: f64,
}

fn test_radiance_balance(radiance_multiplier: f64) -> TuningResult {
    // Temporarily modify the radiance constant by creating a custom simulation
    // We'll run a short simulation to see the thermal balance
    
    let mut sim = Simulation::new(SimProps {
        name: "radiance_test",
        planet: Planet {
            radius_km: EARTH_RADIUS_KM as f64,
            resolution: Resolution::One,
        },
        ops: vec![
            CoolingOp::handle(1.0),
            // Use custom radiance with the multiplier
            RadianceOp::handle_with_custom_heat(1.3e21 * radiance_multiplier),
            LithosphereUnifiedOp::handle(
                vec![(MaterialType::Silicate, 1.0)],
                42,
                0.1,
            ),
        ],
        res: Resolution::Two,
        layer_count: 4,
        layer_height: 10.0,
        layer_height_km: 10.0,
        sim_steps: 50,  // Short test
        years_per_step: 50_000,
        debug: false,
        alert_freq: 50,
        starting_surface_temp_k: 2000.0,
    });

    // Record initial state
    let initial_temp = sim.cells.values()
        .filter_map(|column| column.layers.first())
        .map(|layer| layer.kelvin())
        .sum::<f64>() / sim.cells.len() as f64;
    
    let initial_energy = sim.energy_at_layer(0);
    
    // Run simulation
    sim.simulate();
    
    // Record final state
    let final_temp = sim.cells.values()
        .filter_map(|column| column.layers.first())
        .map(|layer| layer.kelvin())
        .sum::<f64>() / sim.cells.len() as f64;
    
    let final_energy = sim.energy_at_layer(0);
    
    let final_lithosphere = sim.cells.values()
        .map(|column| column.total_lithosphere_height())
        .sum::<f64>();
    
    TuningResult {
        initial_temp,
        final_temp,
        initial_energy,
        final_energy,
        final_lithosphere,
    }
}

// This example now actually tests different radiance values using the
// configurable RadianceOp::handle_with_custom_heat() method!
