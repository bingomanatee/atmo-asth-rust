// Comprehensive operator validation tests
// Tests each operator individually to ensure values change in expected ways

use atmo_asth_rust::sim::sim_op::{AtmosphereOp, CoreRadianceOp};
use atmo_asth_rust::sim::{Simulation, SimProps};
use atmo_asth_rust::planet::Planet;
use atmo_asth_rust::constants::EARTH_RADIUS_KM;
use atmo_asth_rust::temp_utils;
use h3o::Resolution;

#[test]
fn test_core_radiance_op_adds_energy() {
    println!("🔥 Testing CoreRadianceOp - Should ADD energy to bottom layer");
    
    let mut sim = create_test_simulation();
    
    // Record initial state
    let cell = sim.cells.values().next().unwrap();
    let initial_bottom_energy = cell.asth_layers[cell.asth_layers.len() - 1].energy_joules();
    let initial_bottom_temp = cell.asth_layers[cell.asth_layers.len() - 1].kelvin();
    
    println!("   Initial bottom energy: {:.2e} J", initial_bottom_energy);
    println!("   Initial bottom temp: {:.1}K", initial_bottom_temp);
    
    // Apply CoreRadianceOp
    let mut core_op = CoreRadianceOp {
        name: "TestCore".to_string(),
        core_radiance_j_per_km2_per_year: 2.52e12, // Earth's core heat
    };
    
    sim.step_with_ops(&mut [&mut core_op]);
    
    // Check results
    let cell = sim.cells.values().next().unwrap();
    let final_bottom_energy = cell.asth_layers[cell.asth_layers.len() - 1].energy_joules();
    let final_bottom_temp = cell.asth_layers[cell.asth_layers.len() - 1].kelvin();
    
    println!("   Final bottom energy: {:.2e} J", final_bottom_energy);
    println!("   Final bottom temp: {:.1}K", final_bottom_temp);
    
    let energy_added = final_bottom_energy - initial_bottom_energy;
    let temp_increase = final_bottom_temp - initial_bottom_temp;
    
    println!("   Energy added: {:.2e} J", energy_added);
    println!("   Temperature increase: {:.1}K", temp_increase);
    
    assert!(energy_added > 0.0, "CoreRadianceOp should add energy!");
    assert!(temp_increase > 0.0, "CoreRadianceOp should increase temperature!");
    
    println!("   ✅ CoreRadianceOp correctly adds energy and increases temperature");
}

#[test]
fn test_atmosphere_op_removes_energy() {
    println!("❄️ Testing AtmosphereOp - Should REMOVE energy from surface");
    
    let mut sim = create_hot_surface_simulation();
    
    // Record initial state
    let cell = sim.cells.values().next().unwrap();
    let initial_surface_energy = cell.asth_layers[0].energy_joules();
    let initial_surface_temp = cell.asth_layers[0].kelvin();
    
    println!("   Initial surface energy: {:.2e} J", initial_surface_energy);
    println!("   Initial surface temp: {:.1}K", initial_surface_temp);
    
    // Apply AtmosphereOp
    let mut atmosphere_op = AtmosphereOp {
        name: "TestAtmo".to_string(),
        atmospheric_mass_kg_per_m2: 0.0,
        outgassing_threshold_k: 1000.0,
        outgassing_rate_multiplier: 1e-12,
        atmospheric_efficiency: 0.8,
    };
    
    sim.step_with_ops(&mut [&mut atmosphere_op]);
    
    // Check results
    let cell = sim.cells.values().next().unwrap();
    let final_surface_energy = cell.asth_layers[0].energy_joules();
    let final_surface_temp = cell.asth_layers[0].kelvin();
    
    println!("   Final surface energy: {:.2e} J", final_surface_energy);
    println!("   Final surface temp: {:.1}K", final_surface_temp);
    
    let energy_removed = initial_surface_energy - final_surface_energy;
    let temp_decrease = initial_surface_temp - final_surface_temp;
    
    println!("   Energy removed: {:.2e} J", energy_removed);
    println!("   Temperature decrease: {:.1}K", temp_decrease);
    
    assert!(energy_removed > 0.0, "AtmosphereOp should remove energy!");
    assert!(temp_decrease > 0.0, "AtmosphereOp should decrease temperature!");
    
    println!("   ✅ AtmosphereOp correctly removes energy and decreases temperature");
}

#[test]
fn test_radiance_op_mixes_energy() {
    println!("🌡️ Testing RadianceOp - Should MIX energy between layers");
    
    let mut sim = create_gradient_simulation();
    
    // Record initial state
    let cell = sim.cells.values().next().unwrap();
    let initial_temps: Vec<f64> = cell.asth_layers.iter().map(|l| l.kelvin()).collect();
    let initial_gradient = initial_temps[0] - initial_temps[initial_temps.len() - 1];
    
    println!("   Initial temperatures: {:?}", initial_temps);
    println!("   Initial gradient: {:.1}K", initial_gradient);
    
    // Apply RadianceOp
    let mut radiance_op = RadianceOp::new();
    
    sim.step_with_ops(&mut [&mut radiance_op]);
    
    // Check results
    let cell = sim.cells.values().next().unwrap();
    let final_temps: Vec<f64> = cell.asth_layers.iter().map(|l| l.kelvin()).collect();
    let final_gradient = final_temps[0] - final_temps[final_temps.len() - 1];
    
    println!("   Final temperatures: {:?}", final_temps);
    println!("   Final gradient: {:.1}K", final_gradient);
    
    let gradient_reduction = initial_gradient - final_gradient;
    
    println!("   Gradient reduction: {:.1}K", gradient_reduction);
    
    assert!(gradient_reduction.abs() > 0.1, "RadianceOp should change temperature gradient!");
    
    println!("   ✅ RadianceOp correctly mixes energy between layers");
}

#[test]
fn test_combined_operators() {
    println!("🔄 Testing Combined Operators - All working together");
    
    let mut sim = create_test_simulation();
    
    // Record initial state
    let cell = sim.cells.values().next().unwrap();
    let initial_temps: Vec<f64> = cell.asth_layers.iter().map(|l| l.kelvin()).collect();
    let initial_total_energy: f64 = cell.asth_layers.iter().map(|l| l.energy_joules()).sum();

    println!("   Initial temperatures: {:?}", initial_temps);
    println!("   Initial total energy: {:.2e} J", initial_total_energy);
    
    // Create all operators
    let mut core_op = CoreRadianceOp {
        name: "TestCore".to_string(),
        core_radiance_j_per_km2_per_year: 2.52e12,
    };
    let mut atmosphere_op = AtmosphereOp {
        name: "TestAtmo".to_string(),
        atmospheric_mass_kg_per_m2: 0.0,
        outgassing_threshold_k: 1400.0,
        outgassing_rate_multiplier: 1e-12,
        atmospheric_efficiency: 0.8,
    };
    let mut radiance_op = RadianceOp::new();
    
    // Run multiple steps
    for step in 0..5 {
        sim.step_with_ops(&mut [&mut core_op, &mut atmosphere_op, &mut radiance_op]);
        
        let cell = sim.cells.values().next().unwrap();
        let temps: Vec<f64> = cell.asth_layers.iter().map(|l| l.kelvin()).collect();
        let total_energy: f64 = cell.asth_layers.iter().map(|l| l.energy_joules()).sum();

        println!("   Step {}: temps={:?}, energy={:.2e}", step + 1, temps, total_energy);
    }
    
    // Check final state
    let cell = sim.cells.values().next().unwrap();
    let final_temps: Vec<f64> = cell.asth_layers.iter().map(|l| l.kelvin()).collect();
    let final_total_energy: f64 = cell.asth_layers.iter().map(|l| l.energy_joules()).sum();

    println!("   Final temperatures: {:?}", final_temps);
    println!("   Final total energy: {:.2e} J", final_total_energy);
    
    // Verify changes occurred
    let temps_changed = initial_temps != final_temps;
    let energy_changed = (final_total_energy - initial_total_energy).abs() > 1e15;
    
    assert!(temps_changed, "Combined operators should change temperatures!");
    assert!(energy_changed, "Combined operators should change total energy!");
    
    println!("   ✅ Combined operators work together and change system state");
}

// Helper functions to create test simulations
fn create_test_simulation() -> Simulation {
    Simulation::new(SimProps {
        name: "test_sim",
        planet: Planet {
            radius_km: EARTH_RADIUS_KM as f64,
            resolution: Resolution::Zero,
        },
        ops: vec![],
        res: Resolution::Two,
        layer_count: 3,
        layer_height_km: 50.0,
        sim_steps: 1,
        years_per_step: 1000,
        debug: false,
        alert_freq: 1,
        starting_surface_temp_k: 1500.0,
    })
}

fn create_hot_surface_simulation() -> Simulation {
    let mut sim = create_test_simulation();
    // Set very hot surface for significant cooling (realistic hot temperature ~2000K)
    let cell = sim.cells.values_mut().next().unwrap();
    let volume = cell.asth_layers[0].volume_km3();
    let hot_energy = temp_utils::volume_kelvin_to_joules(volume, 2000.0, 1000.0); // Using 1000 J/(kg·K) specific heat
    cell.asth_layers[0].set_energy_joules(hot_energy);
    sim
}

fn create_gradient_simulation() -> Simulation {
    let mut sim = create_test_simulation();
    // Create temperature gradient: cool surface, hot bottom
    let cell = sim.cells.values_mut().next().unwrap();
    cell.asth_layers[0].set_energy_joules(1.0e20); // Cool surface
    cell.asth_layers[1].set_energy_joules(2.0e20); // Medium
    cell.asth_layers[2].set_energy_joules(3.0e20); // Hot bottom
    sim
}
