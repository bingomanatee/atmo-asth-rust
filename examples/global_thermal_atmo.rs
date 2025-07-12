use atmo_asth_rust::energy_mass_composite::MaterialCompositeType;
use atmo_asth_rust::sim_op::{
    AtmosphericGenerationOp, PressureAdjustmentOp, SurfaceEnergyInitOp, TemperatureReportingOp,
};
use atmo_asth_rust::sim_op::atmospheric_generation_op::CrystallizationParams;
use atmo_asth_rust::sim_op::surface_energy_init_op::SurfaceEnergyInitParams;
use atmo_asth_rust::global_thermal::sim_cell::{GlobalH3CellConfig, LayerConfig};
use atmo_asth_rust::planet::Planet;
use atmo_asth_rust::sim_op::SimOpHandle;
/// Global thermal simulation with atmospheric generation
/// Creates a planet with L2 resolution, standard cell layout, surface energy distribution,
/// pressure adjustment, atmospheric generation from melting lithosphere, and temperature reporting
use atmo_asth_rust::sim::simulation::{SimProps, Simulation};
use h3o::Resolution;
use std::rc::Rc;

pub fn run_global_thermal_atmo_example() {
    println!("🌋 Global Thermal Atmosphere Simulation with Oscillating Foundry");
    println!("{}", "=".repeat(70));
    println!("📊 Foundry Configuration:");
    println!("   - Base temperature: 1800K");
    println!("   - Oscillation range: 25% to 175% (450K to 3150K)");
    println!("   - Period: 500 years");
    println!("   - Each cell has unique phase offset");
    println!();

    // Create Earth planet with L2 resolution
    let planet = Planet::earth(Resolution::Two);

    // Create oscillating foundry parameters
    let surface_energy_params = SurfaceEnergyInitParams::with_foundry_oscillation(
        280.0,   // surface_temp_k
        25.0,    // geothermal_gradient_k_per_km
        1800.0,  // core_temp_k (base foundry temperature)
        true,    // oscillation_enabled
        500.0,   // period_years
        0.25,    // min_multiplier (25%)
        1.75,    // max_multiplier (175%)
    );

    // Create simulation properties with atmospheric generation
    let sim_props = SimProps {
        planet: planet.clone(),
        res: Resolution::Two,
        layer_count: 24, // Will be overridden by GlobalH3Cell configuration
        sim_steps: 1000,
        years_per_step: 1000, // 1000 years per step = 1 million years total
        name: "GlobalThermalAtmo",
        debug: false,
        ops: vec![
            SimOpHandle::new(Box::new(SurfaceEnergyInitOp::new_with_params(surface_energy_params))),
            SimOpHandle::new(Box::new(PressureAdjustmentOp::new())),
            SimOpHandle::new(Box::new(AtmosphericGenerationOp::with_crystallization_params(
                CrystallizationParams {
                    outgassing_rate: 0.01,  // 1% outgassing rate
                    volume_decay: 0.7,      // 70% volume decay per layer
                    density_decay: 0.12,    // 12% density per layer (88% reduction)
                    depth_attenuation: 0.8, // 80% condotribution from deeper layers (20% reduction per depth)
                    crystallization_rate: 0.1, // 10% crystallization loss per atmospheric layer
                    debug: false,           // debug output
                }
            ))),
            SimOpHandle::new(Box::new(TemperatureReportingOp::new())),
        ],
    };

    // Create simulation
    let mut sim = Simulation::new(sim_props);

    // Configure cells with standard Earth-like layout
    let _planet_rc = Rc::new(planet);
    sim.make_cells(|cell_index, planet| {
        // Standard cell layout: 4×20km atmosphere + 10×4km lithosphere + 10×8km asthenosphere
        let layer_schedule = vec![
            LayerConfig {
                cell_type: MaterialCompositeType::Air,
                cell_count: 4,
                height_km: 20.0, // 80km total atmosphere
                is_foundry: false,
            },
            LayerConfig {
                cell_type: MaterialCompositeType::Silicate,
                cell_count: 4,
                height_km: 10.0, // 40km total lithosphere
                is_foundry: false,
            },
            LayerConfig {
                cell_type: MaterialCompositeType::Silicate,
                cell_count: 8,
                height_km: 20.0, // 80km total asthenosphere
                is_foundry: false,
            },
        ];

        GlobalH3CellConfig {
            h3_index: cell_index,
            planet,
            layer_schedule,
        }
    });

    // Run simulation
    sim.run();
}

fn main() {
    run_global_thermal_atmo_example();
}
