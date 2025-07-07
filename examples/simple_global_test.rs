/// Enhanced global test with atmospheric generation and oscillating foundry
/// Tests the most up-to-date configuration with atmospheric dynamics

use atmo_asth_rust::sim::simulation::{Simulation, SimProps};
use atmo_asth_rust::planet::Planet;
use atmo_asth_rust::global_thermal::global_h3_cell::{GlobalH3CellConfig, LayerConfig};
use atmo_asth_rust::energy_mass_composite::MaterialCompositeType;
use atmo_asth_rust::sim_op::{
    SurfaceEnergyInitOp,
    SurfaceEnergyInitParams,
    PressureAdjustmentOp,
    HeatRedistributionOp,
    TemperatureReportingOp,
    SpaceRadiationOp,
    SpaceRadiationOpParams,
    AtmosphericGenerationOp,
};
use atmo_asth_rust::sim_op::atmospheric_generation_op::CrystallizationParams;
use atmo_asth_rust::sim_op::SimOpHandle;
use h3o::Resolution;
use std::rc::Rc;

pub fn run_simple_global_test() {
    // Create Earth planet with L2 resolution
    let planet = Planet::earth(Resolution::Two);

    // Create simulation properties with all four operations
    let sim_props = SimProps {
        planet: planet.clone(),
        res: Resolution::Two,
        layer_count: 24, // Will be overridden by GlobalH3Cell configuration
        sim_steps: 10, // Just 10 steps for testing
        years_per_step: 1000,
        name: "SimpleGlobalTest",
        debug: false,
        ops: vec![
            SimOpHandle::new(Box::new(SurfaceEnergyInitOp::new_with_params(
                SurfaceEnergyInitParams::with_temperatures(1500.0, 25.0, 3000.0)
            ))), // Initialize with 1500K surface temperature
            SimOpHandle::new(Box::new(PressureAdjustmentOp::new())), // Apply pressure compaction during init
            SimOpHandle::new(Box::new(SpaceRadiationOp::new(
                SpaceRadiationOpParams::with_reporting()
            ))), // Radiate heat from surface layers to space FIRST
            SimOpHandle::new(Box::new(HeatRedistributionOp::new())), // Heat redistribution between layers AFTER radiation
            SimOpHandle::new(Box::new(TemperatureReportingOp::with_frequency(20.0))), // Report at start, middle, and end
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
            },
            LayerConfig {
                cell_type: MaterialCompositeType::Silicate,
                cell_count: 10,
                height_km: 4.0, // 40km total lithosphere
            },
            LayerConfig {
                cell_type: MaterialCompositeType::Silicate,
                cell_count: 10,
                height_km: 8.0, // 80km total asthenosphere
            },
        ];
        
        GlobalH3CellConfig {
            h3_index: cell_index,
            planet,
            layer_schedule,
        }
    });

    // Run the simulation to see the layer reporting in action
    sim.run();
}

fn main() {
    run_simple_global_test();
}
