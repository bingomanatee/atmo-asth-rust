use atmo_asth_rust::energy_mass_composite::MaterialCompositeType;
use atmo_asth_rust::sim::sim_op::{
    AtmosphericGenerationOp, PressureAdjustmentOp, SurfaceEnergyInitOp, TemperatureReportingOp,
};
use atmo_asth_rust::global_thermal::global_h3_cell::{GlobalH3CellConfig, LayerConfig};
use atmo_asth_rust::planet::Planet;
use atmo_asth_rust::sim::sim_op::SimOpHandle;
/// Global thermal simulation with atmospheric generation
/// Creates a planet with L2 resolution, standard cell layout, surface energy distribution,
/// pressure adjustment, atmospheric generation from melting lithosphere, and temperature reporting
use atmo_asth_rust::sim::simulation::{SimProps, Simulation};
use h3o::Resolution;
use std::rc::Rc;

pub fn run_global_thermal_atmo_example() {
    // Create Earth planet with L2 resolution
    let planet = Planet::earth(Resolution::Two);

    // Create simulation properties with atmospheric generation
    let sim_props = SimProps {
        planet: planet.clone(),
        res: Resolution::Two,
        layer_count: 24, // Will be overridden by GlobalH3Cell configuration
        sim_steps: 1000,
        years_per_step: 1000, // 1000 years per step = 1 million years total
        name: "GlobalThermalAtmo",
        debug: true,
        ops: vec![
            SimOpHandle::new(Box::new(SurfaceEnergyInitOp::new())),
            SimOpHandle::new(Box::new(PressureAdjustmentOp::new())),
            SimOpHandle::new(Box::new(AtmosphericGenerationOp::with_crystallization_params(
                0.01,  // 1% outgassing rate
                0.7,   // 70% volume decay per layer
                0.12,  // 12% density per layer (88% reduction)
                0.8,   // 80% contribution from deeper layers (20% reduction per depth)
                0.1,   // 10% crystallization loss per atmospheric layer
                true   // debug output
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
            },
            LayerConfig {
                cell_type: MaterialCompositeType::Silicate,
                cell_count: 4,
                height_km: 10.0, // 40km total lithosphere
            },
            LayerConfig {
                cell_type: MaterialCompositeType::Silicate,
                cell_count: 8,
                height_km: 20.0, // 80km total asthenosphere
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
