/// Global thermal simulation with RadianceOp integration
/// Replaces foundry temperature with realistic radiance system to see effects on
/// lithosphere production/melting and atmospheric generation
/// 
/// This demonstrates the full integration of the radiance system with:
/// - Realistic hotspot thermal evolution (geological timescales)
/// - Doubled populations for adequate thermal activity
/// - Direct cell_index to 3D coordinate conversion
/// - Lithosphere melting and atmospheric generation

use atmo_asth_rust::energy_mass_composite::MaterialCompositeType;
use atmo_asth_rust::sim_op::{
    AtmosphericGenerationOp, PressureAdjustmentOp, RadianceOp, TemperatureReportingOp,
    HeatRedistributionOp, SurfaceEnergyInitOp, SurfaceEnergyInitParams,
};
use atmo_asth_rust::sim_op::atmospheric_generation_op::CrystallizationParams;
use atmo_asth_rust::sim_op::radiance_op::RadianceOpParams;
use atmo_asth_rust::global_thermal::global_h3_cell::{GlobalH3CellConfig, LayerConfig};
use atmo_asth_rust::planet::Planet;
use atmo_asth_rust::sim_op::SimOpHandle;
use atmo_asth_rust::sim::simulation::{SimProps, Simulation};
use atmo_asth_rust::sim::radiance::RadianceSystem;
use h3o::Resolution;
use std::rc::Rc;

pub fn run_global_thermal_radiance_integrated() {
    println!("🌋 Global Thermal Simulation: Foundry Baseline + RadianceOp Enhancement");
    println!("{}", "=".repeat(70));
    println!("🔥 Thermal System Configuration:");
    println!("   - Foundry Temperature: Establishes baseline thermal state");
    println!("   - RadianceOp: High energy to maintain hot asthenosphere against cooling");
    println!("   - Base core radiance: 2.52e13 J/km²/year (10x Earth's core to overcome cooling)");
    println!("   - Radiance system: 5x multiplier for sustained asthenosphere heat");
    println!("   - Standard hotspots: 70 active (0.5-60 Ma lifetime)");
    println!("   - Major plumes: 16 active (20-200 Ma lifetime)");
    println!("   - Transient upwells: 30 active (0.5-1 Ma lifetime)");
    println!("   - Cooling zones: 24 active (1-50 Ma lifetime)");
    println!("   - Total: ~140 thermal features globally");
    println!("   - Thermal peak: ~1 Ma, then exponential cooling");
    println!();

    // Create Earth planet with L2 resolution
    let planet = Planet::earth(Resolution::Two);

    // Create radiance system with realistic geological features
    let mut radiance_system = RadianceSystem::new(0.0);
    
    // Initialize with sustainable thermal features using doubled populations
    println!("🌍 Initializing radiance system with geological thermal features...");
    if let Err(e) = radiance_system.initialize_sustainable_features(Resolution::Two, 0.0) {
        eprintln!("Warning: Failed to initialize radiance system: {}", e);
        println!("Continuing with default radiance system...");
    }

    // Display radiance system statistics
    let stats = radiance_system.get_statistics(0.0);
    println!("📊 Initial Radiance System Statistics:");
    println!("   - Active inflows: {}", stats.active_inflows);
    println!("   - Active outflows: {}", stats.active_outflows);
    println!("   - Total inflow rate: {:.2} MW", stats.total_inflow_rate_mw);
    println!("   - Total outflow rate: {:.2} MW", stats.total_outflow_rate_mw);
    println!("   - Net flow rate: {:.2} MW", stats.net_flow_rate_mw);
    println!();

    // Create RadianceOp parameters with foundry layers as constant heat sources
    let radiance_params = RadianceOpParams {
        base_core_radiance_j_per_km2_per_year: 2.52e12 * 10.0, // 10x Earth's core radiance to overcome cooling
        radiance_system_multiplier: 5.0, // 5x radiance system contribution for sustained heat
        foundry_temperature_k: 2100.0, // Deep foundry layers maintained at 2100K
        enable_reporting: true, // Enable detailed reporting
        enable_energy_logging: false, // Disable energy flow debugging
    };

    // Create simulation properties with RadianceOp instead of foundry
    let sim_props = SimProps {
        planet: planet.clone(),
        res: Resolution::Two,
        layer_count: 24, // Will be overridden by GlobalH3Cell configuration
        sim_steps: 100,  // Longer run to see thermal evolution
        years_per_step: 50000, // 50,000 years per step = 5 million years total
        name: "GlobalThermalRadianceIntegrated",
        debug: false,
        ops: vec![
            // SurfaceEnergyInitOp establishes baseline thermal state with foundry temperature
            SimOpHandle::new(Box::new(SurfaceEnergyInitOp::new_with_params(SurfaceEnergyInitParams {
                surface_temp_k: 280.0,                    // 280K surface temperature
                geothermal_gradient_k_per_km: 25.0,       // 25K per km depth (realistic gradient)
                core_temp_k: 1800.0,                      // 1800K core temperature (realistic mantle)
                foundry_oscillation_enabled: true,        // Enable foundry temperature oscillation
                foundry_oscillation_period_years: 100_000.0, // 100,000 year oscillation period (geological)
                foundry_min_multiplier: 0.8,              // 80% minimum (1440K minimum)
                foundry_max_multiplier: 1.2,              // 120% maximum (2160K maximum)
            }))),

            // RadianceOp adds incremental energy to already hot asthenosphere root
            SimOpHandle::new(Box::new(RadianceOp::new(radiance_params, radiance_system))),

            // Heat redistribution spreads energy through layers
            SimOpHandle::new(Box::new(HeatRedistributionOp::new())),

            // Pressure adjustment for realistic thermal dynamics (init only - not continuous)
            // Note: Continuous pressure adjustment blocks heat transfer too aggressively
            SimOpHandle::new(Box::new(PressureAdjustmentOp::new())), // Only applies during init
            
            // Atmospheric generation from lithosphere melting
            SimOpHandle::new(Box::new(AtmosphericGenerationOp::with_crystallization_params(
                CrystallizationParams {
                    outgassing_rate: 0.015,  // 1.5% outgassing rate (slightly higher for radiance)
                    volume_decay: 0.7,       // 70% volume decay per layer
                    density_decay: 0.12,     // 12% density per layer (88% reduction)
                    depth_attenuation: 0.8,  // 80% contribution from deeper layers
                    crystallization_rate: 0.1, // 10% crystallization loss per atmospheric layer
                    debug: false,            // Disable debug output for clean final effects
                }
            ))),
            
            // Temperature reporting to track thermal evolution
            SimOpHandle::new(Box::new(TemperatureReportingOp::new())),
        ],
    };

    // Create simulation
    let mut sim = Simulation::new(sim_props);

    // Configure cells with Earth-like layout optimized for radiance system
    let _planet_rc = Rc::new(planet);
    sim.make_cells(|cell_index, planet| {
        // Realistic Earth-like layer structure:
        // - 4×20km atmosphere (80km total)
        // - 4×10km lithosphere (40km total - realistic continental crust thickness)
        // - 10×25km asthenosphere (250km total - realistic asthenosphere depth)
        let layer_schedule = vec![
            LayerConfig {
                cell_type: MaterialCompositeType::Air,
                cell_count: 4,
                height_km: 20.0, // 80km total atmosphere
            },
            LayerConfig {
                cell_type: MaterialCompositeType::Silicate,
                cell_count: 4,
                height_km: 10.0, // 40km total lithosphere (realistic continental crust)
            },
            LayerConfig {
                cell_type: MaterialCompositeType::Silicate,
                cell_count: 10,
                height_km: 25.0, // 250km total asthenosphere (realistic depth)
            },
        ];

        GlobalH3CellConfig {
            h3_index: cell_index,
            planet,
            layer_schedule,
        }
    });

    println!("🚀 Starting simulation with Foundry + RadianceOp thermal system...");
    println!("📈 Foundry establishes baseline, RadianceOp melts asthenosphere...");
    println!("🔥 Expecting molten asthenosphere (25km layers) and solid lithosphere (10km layers)...");
    println!("🌋 Tracking massive lithosphere melting and atmospheric generation...");
    println!("🔧 Enhanced heat transfer for geological equilibration:");
    println!("   - Base transfer rate: 2% per year (increased from 0.5%)");
    println!("   - Max energy transfer: 25% per timestep (increased from 10%)");
    println!("   - Conductivity factor: Up to 2x (increased from 1x cap)");
    println!("   - Pressure effects: Exponentially flattened (minimal influence)");
    println!("   - Goal: Eliminate artificial temperature clamping over geological time");
    println!();

    // Run simulation
    sim.run();

    println!();
    println!("✅ Simulation completed!");
    println!("🔬 Key observations to look for:");
    println!("   - Foundry temperature establishing baseline thermal state");
    println!("   - RadianceOp adding incremental energy to already hot system");
    println!("   - Heat redistribution from asthenosphere to lithosphere");
    println!("   - Lithosphere melting patterns enhanced by radiance hotspots");
    println!("   - Atmospheric generation from melted lithosphere material");
    println!("   - Spatial variation in thermal activity based on hotspot locations");
    println!("   - Temporal evolution as hotspots peak and cool over geological time");
    println!("   - Realistic thermal gradients from foundry baseline + radiance enhancement");
}

fn main() {
    run_global_thermal_radiance_integrated();
}
