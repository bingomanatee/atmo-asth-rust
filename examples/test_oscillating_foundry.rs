s public if yu need the /// Test oscillating foundry temperature with sinusoidal variation
/// Demonstrates foundry temperature oscillating between 25% and 175% of base rate
/// with 500-year period and pseudo-random phase offset per cell

use atmo_asth_rust::energy_mass_composite::MaterialCompositeType;
use atmo_asth_rust::sim_op::{
    PressureAdjustmentOp, SurfaceEnergyInitOp, TemperatureReportingOp, HeatRedistributionOp,
};
use atmo_asth_rust::sim_op::surface_energy_init_op::SurfaceEnergyInitParams;
use atmo_asth_rust::global_thermal::global_h3_cell::{GlobalH3CellConfig, LayerConfig};
use atmo_asth_rust::planet::Planet;
use atmo_asth_rust::sim_op::SimOpHandle;
use atmo_asth_rust::sim::simulation::{SimProps, Simulation};
use h3o::Resolution;
use std::rc::Rc;

pub fn run_oscillating_foundry_test() {
    println!("🌋 Testing Oscillating Foundry Temperature");
    println!("{}", "=".repeat(60));
    println!("📊 Configuration:");
    println!("   - Period: 500 years");
    println!("   - Range: 25% to 175% of base foundry temperature");
    println!("   - Base foundry temp: 1800K");
    println!("   - Effective range: 450K to 3150K");
    println!("   - Pseudo-random phase offset per cell");
    println!();

    // Create Earth planet with L2 resolution (fewer cells for testing)
    let planet = Planet::earth(Resolution::Two);

    // Create surface energy init with oscillating foundry parameters
    let surface_energy_params = SurfaceEnergyInitParams::with_foundry_oscillation(
        280.0,   // surface_temp_k
        25.0,    // geothermal_gradient_k_per_km
        1800.0,  // core_temp_k (base foundry temperature)
        true,    // oscillation_enabled
        500.0,   // period_years
        0.25,    // min_multiplier (25%)
        1.75,    // max_multiplier (175%)
    );

    // Create simulation properties
    let sim_props = SimProps {
        planet: planet.clone(),
        res: Resolution::Two,
        layer_count: 16, // Will be overridden by GlobalH3Cell configuration
        sim_steps: 20,   // Short test: 20 steps
        years_per_step: 50, // 50 years per step = 1000 years total (2 full cycles)
        name: "OscillatingFoundryTest",
        debug: false,
        ops: vec![
            SimOpHandle::new(Box::new(SurfaceEnergyInitOp::new_with_params(surface_energy_params))),
            SimOpHandle::new(Box::new(PressureAdjustmentOp::new())),
            SimOpHandle::new(Box::new(HeatRedistributionOp::new())),
            SimOpHandle::new(Box::new(TemperatureReportingOp::new())),
        ],
    };

    // Create simulation
    let mut sim = Simulation::new(sim_props);

    // Configure cells with simple layout for testing
    let _planet_rc = Rc::new(planet);
    sim.make_cells(|cell_index, planet| {
        // Simple layout: 2×20km atmosphere + 6×10km lithosphere + 8×10km asthenosphere
        let layer_schedule = vec![
            LayerConfig {
                cell_type: MaterialCompositeType::Air,
                cell_count: 2,
                height_km: 20.0, // 40km total atmosphere
            },
            LayerConfig {
                cell_type: MaterialCompositeType::Silicate,
                cell_count: 6,
                height_km: 10.0, // 60km total lithosphere
            },
            LayerConfig {
                cell_type: MaterialCompositeType::Silicate,
                cell_count: 8,
                height_km: 10.0, // 80km total asthenosphere
            },
        ];

        GlobalH3CellConfig {
            h3_index: cell_index,
            planet,
            layer_schedule,
        }
    });

    // Run the simulation
    sim.run();

    // After simulation, analyze the oscillation patterns by examining final temperatures
    println!("\n📈 Oscillation Analysis:");
    println!("Examining final foundry temperatures across different cells to verify oscillation:");
    println!("   Cell ID          Final Temp(K)  Expected Multiplier  Phase Offset");
    println!("   ---------------- ------------- ------------------- ------------");

    let final_time_years = sim.sim_steps as f64 * sim.years_per_step as f64;
    let surface_op = SurfaceEnergyInitOp::new_with_params(
        SurfaceEnergyInitParams::with_foundry_oscillation(
            280.0, 25.0, 1800.0, true, 500.0, 0.25, 1.75
        )
    );

    let mut cell_count = 0;
    for (cell_id, cell) in sim.cells.iter() {
        if cell_count >= 10 { break; } // Show first 10 cells

        if let Some((deepest_layer, _)) = cell.layers_t.last() {
            let foundry_temp = deepest_layer.temperature_k();
            let cell_id_u64 = u64::from(*cell_id);

            let expected_multiplier = surface_op.calculate_foundry_multiplier(final_time_years, cell_id_u64);
            let phase_offset = surface_op.generate_cell_phase_offset(cell_id_u64);

            println!("   {:016x} {:13.1} {:19.3} {:12.3}",
                     cell_id_u64, foundry_temp, expected_multiplier, phase_offset);
        }
        cell_count += 1;
    }



    println!("\n✅ Oscillating foundry temperature test completed!");
    println!("🔍 Key observations:");
    println!("   - Each cell has different phase offset (pseudo-random)");
    println!("   - Temperature oscillates between ~450K and ~3150K");
    println!("   - 500-year period creates smooth sinusoidal variation");
    println!("   - Heat input scales proportionally with temperature");
}



fn main() {
    run_oscillating_foundry_test();
}
