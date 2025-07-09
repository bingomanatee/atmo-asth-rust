/// Global thermal simulation with RadianceOp integration and heat map export
/// Replaces foundry temperature with realistic radiance system to see effects on
/// lithosphere production/melting and atmospheric generation
/// 
/// This demonstrates the full integration of the radiance system with:
/// - Realistic hotspot thermal evolution (geological timescales)
/// - Doubled populations for adequate thermal activity
/// - Direct cell_index to 3D coordinate conversion
/// - Lithosphere melting and atmospheric generation
/// - Heat map visualization export

use atmo_asth_rust::energy_mass_composite::{MaterialCompositeType, EnergyMassComposite};
use atmo_asth_rust::sim_op::{
    AtmosphericGenerationOp, RadianceOp, TemperatureReportingOp,
    HeatRedistributionOp, SurfaceEnergyInitOp, SurfaceEnergyInitParams,
};
use atmo_asth_rust::sim_op::atmospheric_generation_op::CrystallizationParams;
use atmo_asth_rust::sim_op::radiance_op::RadianceOpParams;
use atmo_asth_rust::global_thermal::global_h3_cell::{GlobalH3CellConfig, LayerConfig};
use atmo_asth_rust::planet::Planet;
use atmo_asth_rust::sim_op::{SimOp, SimOpHandle};
use atmo_asth_rust::sim::simulation::{SimProps, Simulation};
use atmo_asth_rust::sim::radiance::RadianceSystem;
use h3o::{Resolution, CellIndex};
use std::rc::Rc;
use std::fs::File;
use std::io::Write;

/// Custom visualization export operation for heat map generation
/// Exports temperature data as a heat map image
struct HeatMapExportOp {
    export_interval: i32,
    last_export_step: i32,
}

impl HeatMapExportOp {
    pub fn new(export_interval: i32) -> Self {
        Self {
            export_interval,
            last_export_step: -1,
        }
    }

    /// Calculate mass-weighted average temperature for non-foundry, non-atmospheric layers
    fn calculate_weighted_temperature(&self, cell: &atmo_asth_rust::global_thermal::global_h3_cell::GlobalH3Cell) -> f64 {
        let mut total_weighted_temp = 0.0;
        let mut total_mass = 0.0;

        for (layer, _) in &cell.layers_t {
            // Skip atmospheric layers (Air material)
            if layer.energy_mass.material_composite_type() == MaterialCompositeType::Air {
                continue;
            }

            // Skip foundry layers (deepest layers with very high temperatures)
            // Consider foundry layers as those deeper than 500km
            if layer.start_depth_km > 500.0 {
                continue;
            }

            let mass = layer.mass_kg();
            let temp = layer.temperature_k();
            
            total_weighted_temp += temp * mass;
            total_mass += mass;
        }

        if total_mass > 0.0 {
            total_weighted_temp / total_mass
        } else {
            0.0
        }
    }

    /// Convert temperature to RGB color (0K=black, 1000K=red, 1500K=yellow, 2000K=white)
    fn temperature_to_rgb(&self, temp_k: f64) -> (u8, u8, u8) {
        let clamped_temp = temp_k.max(0.0).min(2000.0);
        
        if clamped_temp < 1000.0 {
            // 0K to 1000K: black to red
            let ratio = clamped_temp / 1000.0;
            ((255.0 * ratio) as u8, 0, 0)
        } else if clamped_temp < 1500.0 {
            // 1000K to 1500K: red to yellow
            let ratio = (clamped_temp - 1000.0) / 500.0;
            (255, (255.0 * ratio) as u8, 0)
        } else {
            // 1500K to 2000K: yellow to white
            let ratio = (clamped_temp - 1500.0) / 500.0;
            (255, 255, (255.0 * ratio) as u8)
        }
    }

    /// Export heat map data as CSV
    fn export_heat_map(&self, sim: &Simulation, step: i32) {
        let filename = format!("heat_map_step_{:04}.csv", step);
        
        if let Ok(mut file) = File::create(&filename) {
            writeln!(file, "cell_index,temperature_k,red,green,blue").unwrap();
            
            for (cell_index, cell) in &sim.cells {
                let temp = self.calculate_weighted_temperature(cell);
                let (r, g, b) = self.temperature_to_rgb(temp);
                
                writeln!(file, "{:?},{:.2},{},{},{}", cell_index, temp, r, g, b).unwrap();
            }
            
            println!("📊 Exported heat map: {}", filename);
        }
    }
}

impl SimOp for HeatMapExportOp {
    fn name(&self) -> &str {
        "HeatMapExport"
    }

    fn init_sim(&mut self, sim: &mut Simulation) {
        // Export initial heat map
        self.export_heat_map(sim, 0);
        self.last_export_step = 0;
    }

    fn update_sim(&mut self, sim: &mut Simulation) {
        let current_step = sim.current_step();
        
        if current_step - self.last_export_step >= self.export_interval {
            self.export_heat_map(sim, current_step);
            self.last_export_step = current_step;
        }
    }
}

pub fn run_global_thermal_radiance_integrated_with_export() {
    println!("🌋 Global Thermal Simulation: Foundry Baseline + RadianceOp Enhancement + Heat Map Export");
    println!("{}", "=".repeat(80));


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

    // Create RadianceOp parameters with Earth baseline energy injection
    let radiance_params = RadianceOpParams {
        base_core_radiance_j_per_km2_per_year: 2.52e12, // 1.0x Earth's core radiance (baseline)
        radiance_system_multiplier: 1.0, 
        foundry_temperature_k: 3000.0, // Deep foundry reference temperature (not used for resets)
        enable_reporting: false, // Enable detailed reporting
        enable_energy_logging: false, // Disable energy flow debugging
    };

    // Create simulation properties with RadianceOp instead of foundry
    let sim_props = SimProps {
        planet: planet.clone(),
        res: Resolution::Two,
        layer_count: 24, // Will be overridden by GlobalH3Cell configuration
        sim_steps: 500,
        years_per_step: 5000,
        name: "GlobalThermalRadianceIntegrated",
        debug: false,
        ops: vec![
            // SurfaceEnergyInitOp establishes baseline thermal state with foundry temperature
            SimOpHandle::new(Box::new(SurfaceEnergyInitOp::new_with_params(SurfaceEnergyInitParams {
                surface_temp_k: 280.0,                    // 280K surface temperature
                geothermal_gradient_k_per_km: 25.0,       // 25K per km depth (realistic gradient)
                core_temp_k: 2400.0,                      
            }))),

            // RadianceOp adds energy to deepest layer (heat spiral NOT from this)
            SimOpHandle::new(Box::new(RadianceOp::new(radiance_params, radiance_system))),

            // Heat redistribution spreads energy through layers (energy conservation fixed)
            SimOpHandle::new(Box::new(HeatRedistributionOp::new())),
            
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
            
            // Heat map export for visualization (export every 50 steps)
            SimOpHandle::new(Box::new(HeatMapExportOp::new(50))),
        ],
    };

    // Create simulation
    let mut sim = Simulation::new(sim_props);

    // Configure cells with Earth-like layout optimized for radiance system
    let _planet_rc = Rc::new(planet);
    sim.make_cells(|cell_index, planet| {
        // Realistic Earth-like layer structure with gradual thickness transition:
        // - 4×20km atmosphere (80km total)
        // - 4×10km lithosphere (40km total - realistic continental crust thickness)
        // - 3×15km upper asthenosphere (45km total - gradual transition)
        // - 3×20km middle asthenosphere (60km total - intermediate thickness)
        // - 3×25km lower asthenosphere (75km total - deepest layers)
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
                height_km: 10.0, // 40km total lithosphere (realistic continental crust)
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
                cell_count: 3,
                height_km: 20.0, // 45km upper asthenosphere (gradual transition from 10km)
                is_foundry: false,
            },
            LayerConfig {
                cell_type: MaterialCompositeType::Silicate,
                cell_count: 3,
                height_km: 25.0, // 60km middle asthenosphere (intermediate thickness)
                is_foundry: false,
            },
            LayerConfig {
                cell_type: MaterialCompositeType::Silicate,
                cell_count: 3,
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

    println!("🚀 Starting simulation with fixed energy conservation...");
    println!("📈 SurfaceEnergyInitOp establishes baseline + RadianceOp adds energy");
    println!("🔥 Testing thermal stability with corrected heat transfer rates");
    println!("🌋 Tracking thermal evolution with proper energy conservation");
    println!("📊 Heat map export: CSV files with temperature visualization data");
    println!("🌡️  Temperature color mapping: 0K=black, 1000K=red, 1500K=yellow, 2000K=white");
    println!("🔧 Thickness-based heat transfer with natural thermal equilibrium:");
    println!("   - Gradual thickness scaling: 10km→15km→20km→25km layers for smooth energy flow");
    println!("   - Moderate energy injection from RadianceOp");
    println!("   - Natural thermal gradients through heat redistribution");
    println!("   - Surface cooling provides energy dissipation");
    println!("   - Goal: Identify if heat spiral is from RadianceOp or other components");
    println!();

    // Run simulation
    sim.run();

    println!();
    println!("✅ Simulation completed!");
    println!("🔬 Key observations to look for:");
    println!("   - Stable thermal system without runaway heating");
    println!("   - Natural thermal gradients from deep core energy to surface cooling");
    println!("   - Controlled heat redistribution through thickness-scaled diffusion");
    println!("   - Realistic asthenosphere temperatures without artificial resets");
    println!("   - Moderate lithosphere heating and natural atmospheric generation");
    println!("   - Spatial variation in thermal activity based on hotspot locations");
    println!("   - Temporal evolution as hotspots peak and cool over geological time");
    println!("   - Thermal equilibrium achieved through energy balance");
}

fn main() {
    run_global_thermal_radiance_integrated_with_export();
}
