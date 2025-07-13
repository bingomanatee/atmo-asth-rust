use atmo_asth_rust::config::PerlinThermalConfig;
/// Global thermal simulation with RadianceOp integration and PNG heat map export
/// Based on global_thermal_radiance_integrated.rs with added heat map visualization
///
/// Exports PNG heat maps every simulation step at 3 pixels per degree resolution
/// with temperature color coding: 0K=black, 1000K=red, 1500K=yellow, 2000K=white
use atmo_asth_rust::energy_mass_composite::{EnergyMassComposite, MaterialCompositeType};
use atmo_asth_rust::global_thermal::sim_cell::{GlobalH3CellConfig, LayerConfig};
use atmo_asth_rust::planet::Planet;
use atmo_asth_rust::sim::radiance::RadianceSystem;
use atmo_asth_rust::sim::simulation::{SimProps, Simulation};
use atmo_asth_rust::sim_op::atmospheric_generation_op::CrystallizationParams;
use atmo_asth_rust::sim_op::radiance_op::RadianceOpParams;
use atmo_asth_rust::sim_op::{
    AtmosphericGenerationOp, RadianceOp, SurfaceEnergyInitOp, SurfaceEnergyInitParams,
    TemperatureReportingOp, ThermalConductionOp, ThermalConductionParams,
};
use atmo_asth_rust::sim_op::{SimOp, SimOpHandle};
use h3o::Resolution;
use std::rc::Rc;
use export_heat_map::ThermalHeatMapExportOp;

mod export_heat_map;
mod radiance_visualization_op;

pub fn run_global_thermal_radiance_with_heat_map_hi_res() {
    // Create Earth planet with L2 resolution
    let planet = Planet::earth(Resolution::Three);

    // Create radiance system with NO perlin noise - only upwells
    let perlin_config = PerlinThermalConfig {
        average_wm: 0.0,                           // No baseline energy
        variance_wm: 0.0,                          // No perlin amplitude
        scale_range: (1.0, 1.1),                   // Minimal scale (needs range)
        wavelength_km: 996.0,                      // Default wavelength
        transition_period_range: (1000.0, 2000.0), // Long transitions
    };

    let mut radiance_system = RadianceSystem::new_with_perlin_config(0.0, perlin_config);

    // Initialize with sustainable thermal features using doubled populations
    if let Err(_e) = radiance_system.initialize_sustainable_features(Resolution::Two, 0.0) {
        // Continue with default radiance system
    }

    // Create RadianceOp parameters with NO baseline energy - only upwells
    let radiance_params = RadianceOpParams {
        base_core_radiance_j_per_km2_per_year: 0.0, // No base energy
        radiance_system_multiplier: 100.0,          // Full upwell energy
        foundry_temperature_k: 2000.0, // Deep foundry reference temperature (not used for resets)
        enable_reporting: false,       // Enable detailed reporting
        enable_energy_logging: false,  // Disable energy flow debugging
        enable_instant_plumes: false,  // Disable instant plumes for this example
        plume_energy_threshold_j_per_km2_per_year: 5.0e12,
        plume_temperature_threshold_k: 1800.0,
        resolution: RESOLUTION,
    };

    const RESOLUTION: Resolution = Resolution::Three;

    // Create simulation properties with RadianceOp and PNG heat map export
    let sim_props = SimProps {
        planet: planet.clone(),
        res: RESOLUTION,
        layer_count: 24, // Will be overridden by GlobalH3Cell configuration
        sim_steps: 500,  // Very short test to check radiance circles
        years_per_step: 5000,
        name: "GlobalThermalRadianceHeatMap",
        debug: false,
        ops: vec![
            // SurfaceEnergyInitOp establishes baseline thermal state with NO initial energy
            SimOpHandle::new(Box::new(SurfaceEnergyInitOp::new_with_params(
                SurfaceEnergyInitParams {
                    surface_temp_k: 0.0,               // Start with no energy
                    geothermal_gradient_k_per_km: 0.0, // No geothermal gradient
                    core_temp_k: 0.0,                  // No initial core temperature
                },
            ))),
            // RadianceOp adds energy to deepest layer (heat spiral NOT from this)
            SimOpHandle::new(Box::new(RadianceOp::new(radiance_params, radiance_system))),
            // Unified thermal conduction - downstream-only model with overbalance protection
            // Models only hot->cold flows, limits outflow to 50% of source energy (~4 binary transfers per cycle)
            SimOpHandle::new(Box::new(ThermalConductionOp::new_with_params(
                ThermalConductionParams {
                    enable_lateral_conduction: true, // Include lateral heat transfer between neighboring cells
                    lateral_conductivity_factor: 0.5, // Lateral conductivity is 50% of material conductivity
                    temp_diff_threshold_k: 1.0, // 1K threshold - only transfer with significant temperature differences
                    enable_reporting: false,    // Disable reporting for clean output
                },
            ))),
            // Atmospheric generation from lithosphere melting
            SimOpHandle::new(Box::new(
                AtmosphericGenerationOp::with_crystallization_params(CrystallizationParams {
                    outgassing_rate: 0.015, // 1.5% outgassing rate (slightly higher for radiance)
                    volume_decay: 0.7,      // 70% volume decay per layer
                    density_decay: 0.12,    // 12% density per layer (88% reduction)
                    depth_attenuation: 0.8, // 80% contribution from deeper layers
                    crystallization_rate: 0.1, // 10% crystallization loss per atmospheric layer
                    debug: false,           // Disable debug output for clean final effects
                }),
            )),
            // Temperature reporting to track thermal evolution
            SimOpHandle::new(Box::new(TemperatureReportingOp::new())),
            // PNG heat map export for visualization (export every step at 3 ppd)
            SimOpHandle::new(Box::new(ThermalHeatMapExportOp::new(
                RESOLUTION,
                8,
                false,
                "thermal_heat_map_r3"
            ))),
            // Radiance-specific visualization export to subfolder (foundry layer focus)
            // SimOpHandle::new(Box::new(RadianceVisualizationOp::new(Resolution::Two, 3, true))),
        ],
    };

    // Create simulation
    let mut sim = Simulation::new(sim_props);

    // Configure cells with Earth-like layout optimized for radiance system
    let _planet_rc = Rc::new(planet);
    sim.make_cells(|cell_index, planet| {
        // Same layer configuration as baseline
        let layer_schedule = vec![
            LayerConfig {
                cell_type: MaterialCompositeType::Air,
                cell_count: 4,
                height_km: 20.0, // 80km total atmosphere
                is_foundry: false,
            },
            LayerConfig {
                cell_type: MaterialCompositeType::Silicate,
                cell_count: 2,
                height_km: 15.0, // 40km total lithosphere (realistic continental crust)
                is_foundry: false,
            },
            LayerConfig {
                cell_type: MaterialCompositeType::Silicate,
                cell_count: 2,
                height_km: 20.0, // 40km total lithosphere (realistic continental crust)
                is_foundry: false,
            },
            LayerConfig {
                cell_type: MaterialCompositeType::Silicate,
                cell_count: 2,
                height_km: 30.0, // 45km upper asthenosphere (gradual transition from 10km)
                is_foundry: false,
            },
            LayerConfig {
                cell_type: MaterialCompositeType::Silicate,
                cell_count: 2,
                height_km: 40.0, // 60km middle asthenosphere (intermediate thickness)
                is_foundry: false,
            },
            LayerConfig {
                cell_type: MaterialCompositeType::Silicate,
                cell_count: 2,
                height_km: 200.0, // 75km lower asthenosphere (deepest layers)
                is_foundry: true,
            },
        ];

        GlobalH3CellConfig {
            h3_index: cell_index,
            planet,
            layer_schedule,
        }
    });

    // Run simulation silently

    // Run simulation
    sim.run();
}

fn main() {
    run_global_thermal_radiance_with_heat_map_hi_res();
}
