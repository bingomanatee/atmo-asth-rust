// Example testing natural equilibrium between cooling, radiance, and lithosphere formation
// This tests whether the three core operations naturally balance without feedback

use atmo_asth_rust::sim::{Simulation, SimProps};
use atmo_asth_rust::sim::sim_op::{CoolingOp, RadianceOp, LithosphereOp, CsvWriterOp};
use atmo_asth_rust::planet::Planet;
use atmo_asth_rust::constants::EARTH_RADIUS_KM;
use atmo_asth_rust::material::MaterialType;
use h3o::Resolution;

fn main() {
    println!("🌍 Natural Equilibrium Test");
    println!("==========================");
    println!("Testing if cooling, radiance, and lithosphere formation reach natural equilibrium");
    println!("WITHOUT feedback operator - lithosphere should max out at material limit");
    
    // Create a simulation with only the three core thermal operators
    let mut sim = Simulation::new(SimProps {
        name: "equilibrium_test",
        planet: Planet {
            radius_km: EARTH_RADIUS_KM as f64,
            resolution: Resolution::One,
        },
        ops: vec![
            // 1. CSV Writer - records data for analysis
            CsvWriterOp::handle("equilibrium_test.csv".to_string()),
            
            // 2. Cooling operator - removes heat based on lithosphere thickness
            CoolingOp::handle(1.0),
            
            // 3. Radiance operator - adds heat from planetary interior
            RadianceOp::handle(),
            
            // 4. Lithosphere formation operator - grows lithosphere when cool enough
            // NOW RESPECTS max_lith_height_km limit (100 km for silicate)
            LithosphereOp::handle(
                vec![(MaterialType::Silicate, 1.0)], // 100% silicate
                42,    // random seed
                0.1,   // noise scale
                0.001, // growth rate (unused in current implementation)
                1e20,  // formation threshold (unused in current implementation)
            ),
            
            // NO LithosphereFeedbackOp - testing natural equilibrium
        ],
        res: Resolution::Two,
        layer_count: 4,
        layer_height: 10.0,
        layer_height_km: 10.0,
        sim_steps: 30,  // Very short test to see corrected CSV data
        years_per_step: 50_000, // 50,000 years per step
        debug: true,
        alert_freq: 20,
        starting_surface_temp_k: 2000.0 // Start near lithosphere formation temperature
    });

    println!("\n🔥 Initial Conditions:");
    print_simulation_stats(&sim);
    
    println!("\n🚀 Starting equilibrium test...");
    println!("Expected behavior:");
    println!("1. 🌡️  Temperature should cool from radiant heat loss");
    println!("2. 🏔️  Lithosphere should form when temp drops below ~1873K");
    println!("3. 🧊 More lithosphere → more insulation → less cooling");
    println!("4. 🔥 Radiance continues adding heat from below");
    println!("5. ⚖️  System should reach equilibrium when:");
    println!("   - Lithosphere hits max height (100 km for silicate)");
    println!("   - OR cooling/heating balance naturally");
    
    sim.simulate();
    
    println!("\n✅ Final Results:");
    print_simulation_stats(&sim);
    
    println!("\n📊 Analysis:");
    println!("Check equilibrium_test.csv to see if the system:");
    println!("- Reaches stable temperature");
    println!("- Lithosphere growth stops at material limit");
    println!("- Energy balance stabilizes");
    
    println!("\n🔬 Key Questions:");
    println!("1. Does lithosphere reach the 100 km silicate limit?");
    println!("2. Does temperature stabilize without feedback?");
    println!("3. What is the equilibrium temperature?");
    println!("4. How long does it take to reach equilibrium?");
}

fn print_simulation_stats(sim: &Simulation) {
    let total_energy = sim.energy_at_layer(0);
    let cell_count = sim.cells.len();
    let avg_energy_per_cell = total_energy / cell_count as f64;
    
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
    
    // Check if we're at the material limit
    if max_lithosphere >= 99.0 { // Close to 100 km limit
        println!("  🚨 Lithosphere approaching material limit (100 km)!");
    }
}
