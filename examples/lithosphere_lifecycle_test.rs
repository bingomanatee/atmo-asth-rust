// Test to see lithosphere formation and melting lifecycle
// Tracks lithosphere changes over time to see if it forms then melts away

use atmo_asth_rust::sim::{Simulation, SimProps};
use atmo_asth_rust::sim::sim_op::{CoolingOp, RadianceOp, LithosphereUnifiedOp, CsvWriterOp};
use atmo_asth_rust::planet::Planet;
use atmo_asth_rust::constants::EARTH_RADIUS_KM;
use atmo_asth_rust::material::MaterialType;
use h3o::Resolution;

fn main() {
    println!("🔬 Lithosphere Lifecycle Analysis");
    println!("=================================");
    println!("Tracking lithosphere formation and melting over time");
    println!("Using 15% diffusion (optimal from comprehensive test)");
    
    let mut sim = Simulation::new(SimProps {
        name: "lithosphere_lifecycle",
        planet: Planet {
            radius_km: EARTH_RADIUS_KM as f64,
            resolution: Resolution::One,
        },
        ops: vec![
            CsvWriterOp::handle("lithosphere_lifecycle.csv".to_string()),
            CoolingOp::handle(1.0),
            RadianceOp::handle_with_mixing(0.15), // 15% diffusion (optimal)
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
        sim_steps: 100,  // Medium length to see lifecycle
        years_per_step: 10_000,
        debug: false,
        alert_freq: 10,
        starting_surface_temp_k: 2000.0,
    });

    println!("\n📊 Tracking every 10 steps:");
    println!("Step | Surface T | Deep T | Gradient | Lithosphere | Status");
    println!("-----|-----------|--------|----------|-------------|--------");
    
    // Track initial state
    print_step_status(&sim, 0);

    // Run the simulation
    sim.simulate();
    
    println!("\n🔍 Final Analysis:");
    let final_surface_temp = sim.cells.values()
        .filter_map(|column| column.layers.first())
        .map(|layer| layer.kelvin())
        .sum::<f64>() / sim.cells.len() as f64;
    
    let final_lithosphere = sim.cells.values()
        .map(|column| column.total_lithosphere_height())
        .sum::<f64>() / sim.cells.len() as f64;
    
    let formation_temp = 1873.15;
    
    println!("Final surface temperature: {:.0}K", final_surface_temp);
    println!("Formation threshold: {:.0}K", formation_temp);
    println!("Temperature difference: {:+.0}K", final_surface_temp - formation_temp);
    println!("Final lithosphere: {:.3}km average", final_lithosphere);
    
    if final_surface_temp > formation_temp {
        println!("\n🔥 DIAGNOSIS: Temperature above formation threshold");
        println!("   • Lithosphere cannot form at {:.0}K (needs < {:.0}K)", final_surface_temp, formation_temp);
        println!("   • Any existing lithosphere melts via melt_from_below_km_per_year");
        println!("   • System equilibrates with zero lithosphere");
        println!("   • Need more cooling or less diffusion to reach formation temperature");
    } else {
        println!("\n🧊 DIAGNOSIS: Temperature below formation threshold");
        println!("   • Lithosphere should form at {:.0}K", final_surface_temp);
        if final_lithosphere == 0.0 {
            println!("   • No lithosphere present - possible bug in formation logic");
        } else {
            println!("   • Lithosphere present: {:.3}km average", final_lithosphere);
        }
    }
    
    println!("\n📈 Check lithosphere_lifecycle.csv for detailed time series data");
    
    println!("\n🎯 Tuning Recommendations:");
    if final_surface_temp > formation_temp + 50.0 {
        println!("   • Increase diffusion to 20-25% for more cooling");
        println!("   • Or reduce starting temperature");
        println!("   • Or increase cooling rates");
    } else if final_surface_temp > formation_temp {
        println!("   • Fine-tune diffusion to 16-18% for slight cooling");
        println!("   • Very close to formation threshold!");
    } else {
        println!("   • Current settings should allow lithosphere formation");
        println!("   • Check unified lithosphere operator logic");
    }
}

fn print_step_status(sim: &Simulation, step: usize) {
    let surface_temp = sim.cells.values()
        .filter_map(|column| column.layers.first())
        .map(|layer| layer.kelvin())
        .sum::<f64>() / sim.cells.len() as f64;
    
    let deep_temp = sim.cells.values()
        .filter_map(|column| column.layers.last())
        .map(|layer| layer.kelvin())
        .sum::<f64>() / sim.cells.len() as f64;
    
    let gradient = deep_temp - surface_temp;
    
    let total_lithosphere = sim.cells.values()
        .map(|column| column.total_lithosphere_height())
        .sum::<f64>() / sim.cells.len() as f64;
    
    let formation_temp = 1873.15;
    let status = if surface_temp > formation_temp {
        "MELTING"
    } else {
        "FORMING"
    };
    
    println!("{:4} | {:7.0}K | {:6.0}K | {:6.0}K | {:9.3}km | {}", 
             step, surface_temp, deep_temp, gradient, total_lithosphere, status);
}
