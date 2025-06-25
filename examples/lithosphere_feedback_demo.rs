// Example demonstrating the lithosphere feedback system
// This shows how cooling, lithosphere formation, and feedback heating work together

use atmo_asth_rust::sim::{Simulation, SimProps};
use atmo_asth_rust::sim::sim_op::{CoolingOp, RadianceOp, LithosphereOp, LithosphereFeedbackOp, CsvWriterOp};
use atmo_asth_rust::planet::Planet;
use atmo_asth_rust::constants::{ASTHENOSPHERE_SURFACE_START_TEMP_K, EARTH_RADIUS_KM};
use atmo_asth_rust::material::MaterialType;
use h3o::Resolution;

fn main() {
    println!("🌍 Lithosphere Feedback System Demo");
    println!("===================================");
    
    // Create a simulation with all the thermal operators
    let mut sim = Simulation::new(SimProps {
        name: "lithosphere_feedback_demo",
        planet: Planet {
            radius_km: EARTH_RADIUS_KM as f64,
            resolution: Resolution::One,
        },
        ops: vec![
            // 1. CSV Writer - records simulation data for graphing
            CsvWriterOp::handle("lithosphere_feedback_simulation.csv".to_string()),

            // 2. Cooling operator - removes heat based on lithosphere thickness
            CoolingOp::handle(1.0),

            // 3. Radiance operator - adds heat from below, attenuated by lithosphere
            RadianceOp::handle(),

            // 4. Lithosphere formation operator - grows lithosphere when cool enough
            LithosphereOp::handle(
                vec![(MaterialType::Silicate, 1.0)], // 100% silicate
                42,    // random seed
                0.1,   // noise scale
                0.001, // growth rate (unused in current implementation)
                1e20,  // formation threshold (unused in current implementation)
            ),

            // 5. Lithosphere feedback operator - adds heat when too much lithosphere forms
            LithosphereFeedbackOp::handle(
                10.0,  // global threshold: 10 km average lithosphere triggers feedback
                15.0,  // local threshold: 15 km per cell triggers feedback (unused in global mode)
                1e22,  // heat factor: 1e22 J per km excess per year
                true,  // use global threshold mode
            ),
        ],
        res: Resolution::Two,
        layer_count: 4,
        layer_height: 10.0,
        layer_height_km: 10.0,
        sim_steps: 200,
        years_per_step: 100_000, // 100,000 years per step for faster cooling
        debug: true,
        alert_freq: 20,
        starting_surface_temp_k: 2000.0 // Start closer to lithosphere formation temperature
    });

    println!("\n🔥 Initial Conditions:");
    print_simulation_stats(&sim);
    
    println!("\n🚀 Starting simulation...");
    sim.simulate();
    
    println!("\n✅ Final Results:");
    print_simulation_stats(&sim);
    
    println!("\n📊 System Behavior Summary:");
    println!("1. 📊 CsvWriterOp: Records simulation data to CSV file");
    println!("   - Tracks temperature, energy, volume, lithosphere statistics");
    println!("   - Data saved to: lithosphere_feedback_simulation.csv");

    println!("2. 🧊 CoolingOp: Removes heat based on lithosphere insulation");
    println!("   - No lithosphere = maximum cooling");
    println!("   - Thick lithosphere = minimal cooling");

    println!("3. 🔥 RadianceOp: Adds heat from planetary interior");
    println!("   - Heat attenuated by lithosphere thickness");
    println!("   - Provides baseline energy input");

    println!("4. 🏔️  LithosphereOp: Forms lithosphere when temperature is low enough");
    println!("   - Growth rate depends on material and temperature");
    println!("   - Stops growing when too hot");

    println!("5. ⚖️  LithosphereFeedbackOp: Prevents runaway lithosphere formation");
    println!("   - Monitors global/local lithosphere thickness");
    println!("   - Adds heat when threshold exceeded");
    println!("   - Creates negative feedback loop");

    println!("\n🔄 Feedback Loop:");
    println!("   Cool → More Lithosphere → More Insulation → Less Cooling");
    println!("   BUT: Too Much Lithosphere → Feedback Heat → Less Lithosphere");

    println!("\n📈 Data Analysis:");
    println!("   You can now graph the CSV data to visualize:");
    println!("   - Temperature evolution over time");
    println!("   - Energy balance dynamics");
    println!("   - Lithosphere formation and feedback cycles");
    println!("   - System equilibrium behavior");
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
    
    println!("  🌡️  Total Energy: {:.2e} J", total_energy);
    println!("  📱 Cells: {}", cell_count);
    println!("  ⚡ Avg Energy/Cell: {:.2e} J", avg_energy_per_cell);
    println!("  🏔️  Total Lithosphere: {:.2} km", total_lithosphere);
    println!("  📏 Avg Lithosphere: {:.3} km/cell", avg_lithosphere);
    println!("  📊 Lithosphere Range: {:.3} - {:.3} km", min_lithosphere, max_lithosphere);
    
    // Estimate average temperature (very rough approximation)
    // This is just for demonstration - actual temperature calculation is more complex
    let rough_avg_temp = (avg_energy_per_cell / 1e20).powf(0.25) * 1000.0; // Very rough estimate
    println!("  🌡️  Rough Avg Temp: ~{:.0} K", rough_avg_temp);
}
