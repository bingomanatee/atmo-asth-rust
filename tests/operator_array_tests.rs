// Integration tests to verify operators modify the correct arrays
// This is the critical test that should have caught the "temperatures never change" bug

use atmo_asth_rust::sim::sim_op::{AtmosphereOp, CoreRadianceOp, SimOp};
use atmo_asth_rust::sim::{Simulation, SimProps};
use atmo_asth_rust::planet::Planet;
use atmo_asth_rust::constants::EARTH_RADIUS_KM;
use h3o::Resolution;

#[test]
fn test_all_operators_modify_next_arrays() {
    println!("🧪 Testing that all operators modify NEXT arrays, not current arrays");
    println!("This test should catch the 'temperatures never change' bug!");
    
    // Create minimal simulation
    let mut sim = Simulation::new(SimProps {
        name: "array_test",
        planet: Planet {
            radius_km: EARTH_RADIUS_KM as f64,
            resolution: Resolution::Zero,
        },
        ops: vec![], // No operators in SimProps
        res: Resolution::Two,
        layer_count: 2,
        layer_height_km: 50.0,
        sim_steps: 1,
        years_per_step: 1000,
        debug: false,
        alert_freq: 1,
        starting_surface_temp_k: 1800.0,
    });

    // Set up initial state
    let cell = sim.cells.values_mut().next().unwrap();
    cell.asth_layers[0].set_energy_joules(1.0e20); // Surface
    cell.asth_layers[1].set_energy_joules(2.0e20); // Bottom
    
    // Copy current to next (simulate step start)
    for i in 0..cell.asth_layers.len() {
        cell.asth_layers_next[i] = cell.asth_layers[i].clone();
    }

    // Record initial states
    let initial_current_temps: Vec<f64> = cell.asth_layers.iter().map(|l| l.kelvin()).collect();
    let initial_next_temps: Vec<f64> = cell.asth_layers_next.iter().map(|l| l.kelvin()).collect();
    
    println!("Initial current temps: {:?}", initial_current_temps);
    println!("Initial next temps: {:?}", initial_next_temps);

    // Test 1: RadianceOp
    println!("\n🔬 Testing RadianceOp...");
    let mut radiance_op = RadianceOp::new();
    radiance_op.update_sim(&mut sim);
    
    let cell = sim.cells.values().next().unwrap();
    let current_temps_after_radiance: Vec<f64> = cell.asth_layers.iter().map(|l| l.kelvin()).collect();
    let next_temps_after_radiance: Vec<f64> = cell.asth_layers_next.iter().map(|l| l.kelvin()).collect();
    
    // Check if RadianceOp modified next arrays
    let radiance_modified_next = next_temps_after_radiance != initial_next_temps;
    let radiance_modified_current = current_temps_after_radiance != initial_current_temps;
    
    println!("   Current temps after RadianceOp: {:?}", current_temps_after_radiance);
    println!("   Next temps after RadianceOp: {:?}", next_temps_after_radiance);
    println!("   RadianceOp modified current arrays: {}", radiance_modified_current);
    println!("   RadianceOp modified next arrays: {}", radiance_modified_next);
    
    if radiance_modified_next {
        println!("   ✅ RadianceOp correctly modifies NEXT arrays");
    } else {
        println!("   ❌ RadianceOp does NOT modify next arrays!");
    }

    // Test 2: CoreRadianceOp
    println!("\n🔬 Testing CoreRadianceOp...");
    let mut core_op = CoreRadianceOp {
        name: "TestCore".to_string(),
        core_radiance_j_per_km2_per_year: 2.52e12,
    };
    
    // Reset arrays for clean test
    let cell = sim.cells.values_mut().next().unwrap();
    for i in 0..cell.asth_layers.len() {
        cell.asth_layers_next[i] = cell.asth_layers[i].clone();
    }
    let initial_bottom_energy = cell.asth_layers_next[1].energy_joules();
    
    core_op.update_sim(&mut sim);
    
    let cell = sim.cells.values().next().unwrap();
    let final_bottom_energy_current = cell.asth_layers[1].energy_joules();
    let final_bottom_energy_next = cell.asth_layers_next[1].energy_joules();
    
    let core_modified_current = final_bottom_energy_current != initial_bottom_energy;
    let core_modified_next = final_bottom_energy_next != initial_bottom_energy;
    
    println!("   Initial bottom energy: {:.2e}", initial_bottom_energy);
    println!("   Final bottom energy (current): {:.2e}", final_bottom_energy_current);
    println!("   Final bottom energy (next): {:.2e}", final_bottom_energy_next);
    println!("   CoreRadianceOp modified current arrays: {}", core_modified_current);
    println!("   CoreRadianceOp modified next arrays: {}", core_modified_next);
    
    if core_modified_next {
        println!("   ✅ CoreRadianceOp correctly modifies NEXT arrays");
    } else {
        println!("   ❌ CoreRadianceOp does NOT modify next arrays!");
    }

    // Test 3: AtmosphereOp
    println!("\n🔬 Testing AtmosphereOp...");
    let mut atmosphere_op = AtmosphereOp {
        name: "TestAtmo".to_string(),
        atmospheric_mass_kg_per_m2: 0.0,
        outgassing_threshold_k: 1400.0,
        outgassing_rate_multiplier: 1e-12,
        atmospheric_efficiency: 0.8,
    };
    
    // Set up hot surface for cooling
    let cell = sim.cells.values_mut().next().unwrap();
    cell.asth_layers[0].set_energy_joules(5.0e20); // Very hot for significant cooling
    for i in 0..cell.asth_layers.len() {
        cell.asth_layers_next[i] = cell.asth_layers[i].clone();
    }
    let initial_surface_energy = cell.asth_layers_next[0].energy_joules();
    
    atmosphere_op.update_sim(&mut sim);
    
    let cell = sim.cells.values().next().unwrap();
    let final_surface_energy_current = cell.asth_layers[0].energy_joules();
    let final_surface_energy_next = cell.asth_layers_next[0].energy_joules();
    
    let atmo_modified_current = final_surface_energy_current != initial_surface_energy;
    let atmo_modified_next = final_surface_energy_next != initial_surface_energy;
    
    println!("   Initial surface energy: {:.2e}", initial_surface_energy);
    println!("   Final surface energy (current): {:.2e}", final_surface_energy_current);
    println!("   Final surface energy (next): {:.2e}", final_surface_energy_next);
    println!("   AtmosphereOp modified current arrays: {}", atmo_modified_current);
    println!("   AtmosphereOp modified next arrays: {}", atmo_modified_next);
    
    if atmo_modified_next {
        println!("   ✅ AtmosphereOp correctly modifies NEXT arrays");
    } else {
        println!("   ❌ AtmosphereOp does NOT modify next arrays!");
    }

    // Summary
    println!("\n🎯 SUMMARY - Which operators modify the correct arrays:");
    println!("   RadianceOp modifies NEXT arrays: {}", radiance_modified_next);
    println!("   CoreRadianceOp modifies NEXT arrays: {}", core_modified_next);
    println!("   AtmosphereOp modifies NEXT arrays: {}", atmo_modified_next);
    
    let all_correct = radiance_modified_next && core_modified_next && atmo_modified_next;
    
    if all_correct {
        println!("\n✅ ALL OPERATORS CORRECTLY MODIFY NEXT ARRAYS!");
    } else {
        println!("\n❌ SOME OPERATORS MODIFY WRONG ARRAYS - THIS CAUSES 'TEMPERATURES NEVER CHANGE' BUG!");
        
        // Fail the test if operators modify wrong arrays
        assert!(radiance_modified_next, "RadianceOp must modify next arrays, not current arrays");
        assert!(core_modified_next, "CoreRadianceOp must modify next arrays, not current arrays");
        assert!(atmo_modified_next, "AtmosphereOp must modify next arrays, not current arrays");
    }
}
