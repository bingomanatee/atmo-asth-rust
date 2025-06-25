// Complete equilibrium test using the unified lithosphere operator
// Tests the full thermal cycle: cooling, radiance, lithosphere formation/melting

use atmo_asth_rust::sim::{Simulation, SimProps};
use atmo_asth_rust::sim::sim_op::{CoolingOp, RadianceOp, LithosphereUnifiedOp, CsvWriterOp};
use atmo_asth_rust::planet::Planet;
use atmo_asth_rust::constants::EARTH_RADIUS_KM;
use atmo_asth_rust::material::MaterialType;
use h3o::Resolution;

fn main() {
    println!("🌍 Complete Thermal Equilibrium Test");
    println!("====================================");
    println!("Testing the complete thermal cycle with unified lithosphere operator:");
    println!("1. 🧊 Cooling removes heat (varies with lithosphere insulation)");
    println!("2. 🔥 Radiance adds heat from planetary interior");
    println!("3. 🏔️  Unified Lithosphere: Forms when cool, melts when hot");
    println!("4. ⚖️  Natural equilibrium through formation/melting balance");
    
    // Create a simulation with the three core thermal operators
    let mut sim = Simulation::new(SimProps {
        name: "complete_equilibrium_test",
        planet: Planet {
            radius_km: EARTH_RADIUS_KM as f64,
            resolution: Resolution::One,
        },
        ops: vec![
            // 1. CSV Writer - records data for analysis
            CsvWriterOp::handle("complete_equilibrium.csv".to_string()),
            
            // 2. Cooling operator - removes heat based on lithosphere thickness
            CoolingOp::handle(1.0),
            
            // 3. Radiance operator - adds heat from planetary interior
            RadianceOp::handle(),
            
            // 4. Unified Lithosphere operator - handles both formation AND melting
            LithosphereUnifiedOp::handle(
                vec![(MaterialType::Silicate, 1.0)], // 100% silicate
                42,    // random seed
                0.1,   // noise scale
            ),
        ],
        res: Resolution::Two,
        layer_count: 4,
        layer_height: 10.0,
        layer_height_km: 10.0,
        sim_steps: 150,  // Longer simulation to see full cycle
        years_per_step: 50_000, // 50,000 years per step
        debug: true,
        alert_freq: 25,
        starting_surface_temp_k: 2000.0 // Start near lithosphere formation temperature
    });

    println!("\n🔥 Initial Conditions:");
    print_simulation_stats(&sim);
    
    println!("\n🚀 Starting complete thermal cycle test...");
    println!("Expected behavior:");
    println!("1. 🌡️  Initial cooling from 2000K");
    println!("2. 🏔️  Lithosphere forms when temp drops below ~1873K");
    println!("3. 🧊 Lithosphere insulation reduces cooling rate");
    println!("4. 🔥 Radiance continues adding heat from below");
    println!("5. 🌡️  Temperature rises due to insulation + radiance");
    println!("6. 🔥 When temp exceeds formation threshold → lithosphere melts");
    println!("7. 🧊 Less insulation → more cooling → temperature drops");
    println!("8. ⚖️  System reaches dynamic equilibrium");
    
    sim.simulate();
    
    println!("\n✅ Final Results:");
    print_simulation_stats(&sim);
    
    println!("\n📊 Complete Thermal System Analysis:");
    println!("🧊 CoolingOp: Heat removal varies with lithosphere insulation");
    println!("   - No lithosphere = maximum cooling");
    println!("   - Thick lithosphere = minimal cooling");
    
    println!("🔥 RadianceOp: Constant heat input from planetary interior");
    println!("   - Heat attenuated by lithosphere thickness");
    println!("   - Provides baseline energy to prevent total cooling");
    
    println!("🏔️  LithosphereUnifiedOp: Complete formation/melting cycle");
    println!("   - Forms when T < 1873K (silicate formation threshold)");
    println!("   - Melts when T > 1873K using melt_from_below_km_per_year");
    println!("   - Respects 100km maximum height limit");
    println!("   - Adds melted energy back to asthenosphere");
    
    println!("\n🔄 Complete Thermal Cycle:");
    println!("   Cool → Lithosphere Forms → Insulation → Less Cooling → Heat Builds Up");
    println!("   Hot → Lithosphere Melts → Less Insulation → More Cooling → Temperature Drops");
    println!("   ⚖️  Dynamic equilibrium between formation and melting");
    
    println!("\n📈 Data Analysis:");
    println!("   Check complete_equilibrium.csv to see:");
    println!("   - Temperature oscillations around formation threshold");
    println!("   - Lithosphere formation/melting cycles");
    println!("   - Energy balance dynamics");
    println!("   - Natural equilibrium behavior");
    
    println!("\n🎯 Key Success Indicators:");
    println!("   ✓ Temperature stabilizes near 1873K formation threshold");
    println!("   ✓ Lithosphere thickness oscillates (formation/melting cycles)");
    println!("   ✓ Energy balance reaches steady state");
    println!("   ✓ No runaway heating or cooling");
}

fn print_simulation_stats(sim: &Simulation) {
    let total_energy = sim.energy_at_layer(0);
    let cell_count = sim.cells.len();
    
    // Calculate total lithosphere thickness
    let total_lithosphere: f64 = sim.cells.values()
        .map(|column| column.total_lithosphere_height())
        .sum();
    let avg_lithosphere = total_lithosphere / cell_count as f64;
    
    // Find min/max lithosphere thickness
    let lithosphere_thicknesses: Vec<f64> = sim.cells.values()
        .map(|column| column.total_lithosphere_height())
        .collect();
    let min_lithosphere = lithosphere_thicknesses.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_lithosphere = lithosphere_thicknesses.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    
    // Calculate average temperature from first layer
    let avg_temp: f64 = sim.cells.values()
        .filter_map(|column| column.layers.first())
        .map(|layer| layer.kelvin())
        .sum::<f64>() / cell_count as f64;
    
    println!("  🌡️  Avg Temperature: {:.0} K", avg_temp);
    println!("  ⚡ Total Energy: {:.2e} J", total_energy);
    println!("  📱 Cells: {}", cell_count);
    println!("  🏔️  Total Lithosphere: {:.1} km", total_lithosphere);
    println!("  📏 Avg Lithosphere: {:.3} km/cell", avg_lithosphere);
    println!("  📊 Lithosphere Range: {:.3} - {:.3} km", min_lithosphere, max_lithosphere);
    
    // Check formation threshold
    let formation_temp = 1873.15; // Silicate formation temperature
    if (avg_temp - formation_temp).abs() < 50.0 {
        println!("  🎯 Temperature near formation threshold ({:.0}K)!", formation_temp);
    }
    
    // Check if we're at the material limit
    if max_lithosphere >= 99.0 { // Close to 100 km limit
        println!("  🚨 Lithosphere approaching material limit (100 km)!");
    }
    
    // Check for dynamic behavior
    if total_lithosphere > 0.0 && avg_temp > formation_temp {
        println!("  🔥 Hot temperature with lithosphere present - melting should occur");
    } else if total_lithosphere == 0.0 && avg_temp < formation_temp {
        println!("  🧊 Cool temperature with no lithosphere - formation should occur");
    }
}
