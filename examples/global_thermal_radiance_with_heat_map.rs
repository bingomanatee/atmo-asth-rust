/// Global thermal simulation with RadianceOp integration and PNG heat map export
/// Based on global_thermal_radiance_integrated.rs with added heat map visualization
/// 
/// Exports PNG heat maps every simulation step at 3 pixels per degree resolution
/// with temperature color coding: 0K=black, 1000K=red, 1500K=yellow, 2000K=white

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
use atmo_asth_rust::h3o_png::{H3GraphicsConfig, H3GraphicsGenerator};
use h3o::{Resolution, CellIndex};
use image::{Rgb, RgbImage};
use std::rc::Rc;
use std::collections::HashMap;

/// Custom heat map export operation for PNG generation
/// Exports temperature data as PNG images every simulation step
struct ThermalHeatMapExportOp {
    graphics_generator: H3GraphicsGenerator,
    export_every_step: bool,
}

impl ThermalHeatMapExportOp {
    pub fn new(resolution: Resolution, points_per_degree: u32, export_every_step: bool) -> Self {
        let config = H3GraphicsConfig::new(resolution, points_per_degree);
        let mut generator = H3GraphicsGenerator::new(config);
        generator.load_cells();

        Self {
            graphics_generator: generator,
            export_every_step,
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
            if layer.is_foundry {
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
    fn temperature_to_rgb(&self, temp_k: f64) -> Rgb<u8> {
        let clamped_temp = temp_k.max(0.0).min(2000.0);
        
        if clamped_temp < 1000.0 {
            // 0K to 1000K: black to red
            let ratio = clamped_temp / 1000.0;
            Rgb([(255.0 * ratio) as u8, 0, 0])
        } else if clamped_temp < 1500.0 {
            // 1000K to 1500K: red to yellow
            let ratio = (clamped_temp - 1000.0) / 500.0;
            Rgb([255, (255.0 * ratio) as u8, 0])
        } else {
            // 1500K to 2000K: yellow to white
            let ratio = (clamped_temp - 1500.0) / 500.0;
            Rgb([255, 255, (255.0 * ratio) as u8])
        }
    }

    /// Export heat map as PNG image using existing H3 graphics system
    fn export_heat_map_png(&self, sim: &Simulation, step: i32) -> Result<(), Box<dyn std::error::Error>> {
        let filename = format!("examples/thermal_heat_map/heat_map_step_{:04}.png", step);
        
        // Create color map for all cells based on temperature
        let mut cell_colors: HashMap<CellIndex, Rgb<u8>> = HashMap::new();
        
        for (cell_index, cell) in &sim.cells {
            let temp = self.calculate_weighted_temperature(cell);
            let color = self.temperature_to_rgb(temp);
            cell_colors.insert(*cell_index, color);
        }

        // Generate PNG silently - suppress stdout during graphics generation
        self.generate_silent_heat_map(&filename, cell_colors)?;

        Ok(())
    }

    /// Generate heat map PNG without verbose output
    fn generate_silent_heat_map(&self, filename: &str, cell_colors: HashMap<CellIndex, Rgb<u8>>) -> Result<(), Box<dyn std::error::Error>> {
        use image::{ImageBuffer, RgbImage};
        
        // Create image buffer
        let mut image: RgbImage = ImageBuffer::new(self.graphics_generator.config.width, self.graphics_generator.config.height);

        // Fill with black background
        for pixel in image.pixels_mut() {
            *pixel = Rgb([0, 0, 0]);
        }

        // Draw cells silently
        let width_threshold = (self.graphics_generator.config.width as i32) * 20 / 100;
        
        for cell in &self.graphics_generator.cells {
            // Get color for this cell
            let cell_color = cell_colors.get(&cell.cell_index).copied().unwrap_or(Rgb([128, 128, 128]));

            // Convert cell corners to pixel coordinates
            let pixel_coords: Vec<(i32, i32)> = cell.corners.iter()
                .map(|corner| self.graphics_generator.geo_to_pixel(corner.longitude, corner.latitude))
                .collect();

            if pixel_coords.is_empty() {
                continue;
            }

            // Check X extent to filter out broad wraparound cells
            let min_x = pixel_coords.iter().map(|(x, _)| *x).min().unwrap();
            let max_x = pixel_coords.iter().map(|(x, _)| *x).max().unwrap();
            let x_extent = max_x - min_x;

            // Draw cell only if it's coherent (X extent < 20% of width)
            if x_extent < width_threshold {
                // Filter coordinates to image bounds
                let coords: Vec<(i32, i32)> = pixel_coords.iter()
                    .filter(|(x, y)| *x >= 0 && *x < self.graphics_generator.config.width as i32 &&
                                     *y >= 0 && *y < self.graphics_generator.config.height as i32)
                    .cloned()
                    .collect();

                if coords.len() >= 3 {
                    self.fill_polygon_simple(&mut image, &coords, cell_color);
                }
            }
        }

        // Save the image
        image.save(filename)?;
        Ok(())
    }

    /// Simple polygon fill using basic scanline algorithm (copied from h3o_png.rs)
    fn fill_polygon_simple(&self, image: &mut RgbImage, coords: &[(i32, i32)], color: Rgb<u8>) {
        if coords.len() < 3 {
            return;
        }

        // Find bounding box
        let min_y = coords.iter().map(|(_, y)| *y).min().unwrap_or(0).max(0);
        let max_y = coords.iter().map(|(_, y)| *y).max().unwrap_or(0).min(self.graphics_generator.config.height as i32 - 1);

        // For each scanline
        for y in min_y..=max_y {
            let mut intersections = Vec::new();

            // Find intersections with polygon edges
            for i in 0..coords.len() {
                let p1 = coords[i];
                let p2 = coords[(i + 1) % coords.len()];

                // Check if edge crosses this scanline
                if (p1.1 <= y && p2.1 > y) || (p2.1 <= y && p1.1 > y) {
                    // Calculate intersection x
                    let x = if p2.1 == p1.1 {
                        p1.0 // Horizontal line
                    } else {
                        p1.0 + ((y - p1.1) * (p2.0 - p1.0)) / (p2.1 - p1.1)
                    };
                    intersections.push(x);
                }
            }

            // Sort intersections and fill between pairs
            intersections.sort();
            for chunk in intersections.chunks(2) {
                if chunk.len() == 2 {
                    let x1 = chunk[0].max(0).min(self.graphics_generator.config.width as i32 - 1);
                    let x2 = chunk[1].max(0).min(self.graphics_generator.config.width as i32 - 1);
                    
                    for x in x1..=x2 {
                        if x >= 0 && x < self.graphics_generator.config.width as i32 && y >= 0 && y < self.graphics_generator.config.height as i32 {
                            image.put_pixel(x as u32, y as u32, color);
                        }
                    }
                }
            }
        }
    }

    /// Clean up old PNG files from previous runs
    fn cleanup_old_images(&self) {
        use std::fs;
        
        let output_dir = "examples/thermal_heat_map/";
        
        // Read directory and remove any PNG files
        if let Ok(entries) = fs::read_dir(output_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(extension) = path.extension() {
                    if extension == "png" {
                        if let Some(file_name) = path.file_name() {
                            if file_name.to_string_lossy().starts_with("heat_map_step_") {
                                let _ = fs::remove_file(&path); // Ignore errors
                            }
                        }
                    }
                }
            }
        }
    }
}

impl SimOp for ThermalHeatMapExportOp {
    fn name(&self) -> &str {
        "ThermalHeatMapExport"
    }

    fn init_sim(&mut self, sim: &mut Simulation) {
        // Clean up any existing PNG files in the output directory
        self.cleanup_old_images();
        
        // Export initial heat map
        if let Err(e) = self.export_heat_map_png(sim, 0) {
            eprintln!("Warning: Failed to export initial heat map: {}", e);
        }
    }

    fn update_sim(&mut self, sim: &mut Simulation) {
        if self.export_every_step {
            let current_step = sim.current_step();
            if let Err(e) = self.export_heat_map_png(sim, current_step) {
                eprintln!("Warning: Failed to export heat map at step {}: {}", current_step, e);
            }
        }
    }
}

pub fn run_global_thermal_radiance_with_heat_map() {
    println!("🌋 Global Thermal Simulation: Foundry Baseline + RadianceOp Enhancement + PNG Heat Map Export");
    println!("{}", "=".repeat(90));

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

    // Create simulation properties with RadianceOp and PNG heat map export
    let sim_props = SimProps {
        planet: planet.clone(),
        res: Resolution::Two,
        layer_count: 24, // Will be overridden by GlobalH3Cell configuration
        sim_steps: 50, // Reduced for faster heat map generation testing
        years_per_step: 5000,
        name: "GlobalThermalRadianceHeatMap",
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
            
            // PNG heat map export for visualization (export every step at 3 ppd)
            SimOpHandle::new(Box::new(ThermalHeatMapExportOp::new(Resolution::Two, 3, true))),
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

    println!("🚀 Starting simulation with PNG heat map export...");
    println!("📈 SurfaceEnergyInitOp establishes baseline + RadianceOp adds energy");
    println!("🔥 Testing thermal stability with corrected heat transfer rates");
    println!("🌋 Tracking thermal evolution with proper energy conservation");
    println!("🎨 PNG heat map export: 3 pixels per degree resolution");
    println!("🌡️  Temperature color mapping: 0K=black, 1000K=red, 1500K=yellow, 2000K=white");
    println!("📊 PNG files saved to examples/thermal_heat_map/heat_map_step_XXXX.png");
    println!();

    // Run simulation
    sim.run();

    println!();
    println!("✅ Simulation completed!");
    println!("🎨 Heat map PNG files generated in examples/thermal_heat_map/");
    println!("🔬 Key observations to look for:");
    println!("   - Thermal gradients visible as color transitions in PNG images");
    println!("   - Hotspot locations showing higher temperatures (yellow/white regions)");
    println!("   - Temporal evolution of thermal patterns across simulation steps");
    println!("   - Spatial variation in thermal activity based on radiance system");
}

fn main() {
    run_global_thermal_radiance_with_heat_map();
}