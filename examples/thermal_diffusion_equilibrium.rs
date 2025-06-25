// Thermal diffusion equilibrium test using the new mixing-based radiance operator
// Tests realistic thermal equilibrium through energy redistribution rather than energy addition

use atmo_asth_rust::sim::{Simulation, SimProps};
use atmo_asth_rust::sim::sim_op::{CoolingOp, RadianceOp, LithosphereUnifiedOp, CsvWriterOp};
use atmo_asth_rust::planet::Planet;
use atmo_asth_rust::constants::EARTH_RADIUS_KM;
use atmo_asth_rust::material::MaterialType;
use h3o::Resolution;

fn main() {
    println!("🌡️  Thermal Diffusion Equilibrium Test");
    println!("=======================================");
    println!("Testing realistic thermal equilibrium through energy mixing:");
    println!("1. 🧊 Cooling removes heat from surface (varies with lithosphere)");
    println!("2. 🌡️  Thermal diffusion mixes 10% energy between adjacent layers");
    println!("3. 🏔️  Unified Lithosphere: Forms when cool, melts when hot");
    println!("4. ⚖️  Natural equilibrium through heat redistribution (no artificial energy)");
    
    // Create a simulation with thermal diffusion instead of artificial radiance
    let mut sim = Simulation::new(SimProps {
        name: "thermal_diffusion_equilibrium",
        planet: Planet {
            radius_km: EARTH_RADIUS_KM as f64,
            resolution: Resolution::One,
        },
        ops: vec![
            // 1. CSV Writer - records data for analysis
            CsvWriterOp::handle("thermal_diffusion.csv".to_string()),
            
            // 2. Cooling operator - removes heat based on lithosphere thickness
            CoolingOp::handle(0.5),
            
            // 3. Thermal diffusion - mixes energy between layers (no artificial energy addition)
            RadianceOp::handle_with_mixing(0.1), // 10% mixing between layers
            
            // 4. Unified Lithosphere operator - handles both formation AND melting
            LithosphereUnifiedOp::handle(
                vec![(MaterialType::Silicate, 1.0)], // 100% silicate
                42,    // random seed
                0.1,   // noise scale
            ),
        ],
        res: Resolution::Two,
        layer_count: 4,
        layer_height_km: 10.0,
        sim_steps: 100,  // Medium simulation to see equilibrium
        years_per_step: 50_000, // 50,000 years per step
        debug: true,
        alert_freq: 20,
        starting_surface_temp_k: 2000.0 // Start near lithosphere formation temperature
    });

    println!("\n🔥 Initial Conditions:");
    print_simulation_stats(&sim);
    
    println!("\n🚀 Starting thermal diffusion equilibrium test...");
    println!("Expected behavior:");
    println!("1. 🌡️  Initial cooling from 2000K");
    println!("2. 🌡️  Heat flows from deep layers to surface through diffusion");
    println!("3. 🏔️  Lithosphere forms when surface temp drops below ~1873K");
    println!("4. 🧊 Lithosphere insulation reduces cooling rate");
    println!("5. 🌡️  Thermal diffusion continues bringing heat from depth");
    println!("6. 🔥 When surface temp exceeds formation threshold → lithosphere melts");
    println!("7. 🧊 Less insulation → more cooling → temperature drops");
    println!("8. ⚖️  System reaches natural equilibrium through heat redistribution");
    
    sim.simulate();
    
    println!("\n✅ Final Results:");
    print_simulation_stats(&sim);
    
    println!("\n📊 Thermal Diffusion System Analysis:");
    println!("🧊 CoolingOp: Heat removal varies with lithosphere insulation");
    println!("   - No lithosphere = maximum cooling");
    println!("   - Thick lithosphere = minimal cooling");
    
    println!("🌡️  Thermal Diffusion: Energy redistribution between layers");
    println!("   - 10% energy mixing between adjacent layers");
    println!("   - Heat flows from hot (deep) to cold (surface) layers");
    println!("   - Maintains natural temperature gradient");
    println!("   - NO artificial energy addition - just redistribution");
    
    println!("🏔️  LithosphereUnifiedOp: Complete formation/melting cycle");
    println!("   - Forms when T < 1873K (silicate formation threshold)");
    println!("   - Melts when T > 1873K using melt_from_below_km_per_year");
    println!("   - Respects 100km maximum height limit");
    println!("   - Adds melted energy back to asthenosphere");
    
    println!("\n🔄 Natural Thermal Cycle:");
    println!("   Cool → Lithosphere Forms → Insulation → Less Cooling");
    println!("   Diffusion → Heat from Depth → Temperature Rises");
    println!("   Hot → Lithosphere Melts → Less Insulation → More Cooling");
    println!("   ⚖️  Natural equilibrium through heat redistribution");
    
    println!("\n📈 Data Analysis:");
    println!("   Check thermal_diffusion.csv to see:");
    println!("   - Temperature stabilization through natural processes");
    println!("   - Lithosphere formation/melting cycles");
    println!("   - Energy conservation (no artificial energy addition)");
    println!("   - Realistic thermal gradient maintenance");
    
    println!("\n🎯 Key Success Indicators:");
    println!("   ✓ Temperature stabilizes through natural heat flow");
    println!("   ✓ Temperature gradient maintained (deep >= surface)");
    println!("   ✓ Lithosphere thickness oscillates naturally");
    println!("   ✓ Energy conservation (total energy decreases only through cooling)");
    println!("   ✓ No runaway heating (unlike artificial radiance)");
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
    
    // Calculate temperature gradient (difference between deepest and surface layers)
    let temp_gradients: Vec<f64> = sim.cells.values()
        .filter_map(|column| {
            if column.layers.len() >= 2 {
                let surface_temp = column.layers.first()?.kelvin();
                let deep_temp = column.layers.last()?.kelvin();
                Some(deep_temp - surface_temp)
            } else {
                None
            }
        })
        .collect();
    
    let avg_gradient = if !temp_gradients.is_empty() {
        temp_gradients.iter().sum::<f64>() / temp_gradients.len() as f64
    } else {
        0.0
    };
    
    println!("  🌡️  Avg Surface Temperature: {:.0} K", avg_temp);
    println!("  📈 Avg Temperature Gradient: {:.0} K (deep - surface)", avg_gradient);
    println!("  ⚡ Total Energy: {:.2e} J", total_energy);
    println!("  📱 Cells: {}", cell_count);
    println!("  🏔️  Total Lithosphere: {:.1} km", total_lithosphere);
    println!("  📏 Avg Lithosphere: {:.3} km/cell", avg_lithosphere);
    println!("  📊 Lithosphere Range: {:.3} - {:.3} km", min_lithosphere, max_lithosphere);
    
    // Check formation threshold
    let formation_temp = 1873.15; // Silicate formation temperature
    if (avg_temp - formation_temp).abs() < 100.0 {
        println!("  🎯 Temperature near formation threshold ({:.0}K)!", formation_temp);
    }
    
    // Check temperature gradient
    if avg_gradient > 0.0 {
        println!("  ✅ Positive temperature gradient maintained (deep > surface)");
    } else {
        println!("  ⚠️  Temperature gradient issue (deep <= surface)");
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
