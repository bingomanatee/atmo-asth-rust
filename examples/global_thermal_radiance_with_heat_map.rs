use atmo_asth_rust::config::PerlinThermalConfig;
/// Global thermal simulation with RadianceOp integration and PNG heat map export
/// Based on global_thermal_radiance_integrated.rs with added heat map visualization
///
/// Exports PNG heat maps every simulation step at 3 pixels per degree resolution
/// with temperature color coding: 0K=black, 1000K=red, 1500K=yellow, 2000K=white
use atmo_asth_rust::energy_mass_composite::{EnergyMassComposite, MaterialCompositeType};
use atmo_asth_rust::global_thermal::sim_cell::{GlobalH3CellConfig, LayerConfig};
use atmo_asth_rust::h3o_png::{H3GraphicsConfig, H3GraphicsGenerator};
use atmo_asth_rust::planet::Planet;
use atmo_asth_rust::sim::radiance::RadianceSystem;
use atmo_asth_rust::sim::simulation::{SimProps, Simulation};
use atmo_asth_rust::sim_op::atmospheric_generation_op::CrystallizationParams;
use atmo_asth_rust::sim_op::radiance_op::RadianceOpParams;
use atmo_asth_rust::sim_op::radiance_op::{RadianceSource, RadianceSourceType};
use atmo_asth_rust::sim_op::{
    AtmosphericGenerationOp, RadianceOp, SurfaceEnergyInitOp, SurfaceEnergyInitParams,
    TemperatureReportingOp, ThermalConductionOp, ThermalConductionParams,
};
use atmo_asth_rust::sim_op::{SimOp, SimOpHandle};
use h3o::{CellIndex, LatLng, Resolution};
use image::{Rgb, RgbImage};
use std::collections::HashMap;
use std::rc::Rc;

/// Pre-calculated color lookup table for energy density visualization
/// Provides O(1) color lookup instead of expensive per-pixel calculations
struct ColorLookupTable {
    /// Color table indexed by log10 energy density (scaled to 0-65535)
    colors: Vec<Rgb<u8>>,
    /// Minimum log10 energy density value
    min_log_energy: f64,
    /// Maximum log10 energy density value  
    max_log_energy: f64,
    /// Scale factor for converting log energy to table index
    scale_factor: f64,
}

impl ColorLookupTable {
    /// Create a new color lookup table with the specified energy range
    fn new(min_log_energy: f64, max_log_energy: f64, table_size: usize) -> Self {
        let mut colors = Vec::with_capacity(table_size);
        let scale_factor = (table_size - 1) as f64 / (max_log_energy - min_log_energy);

        // Pre-calculate colors for the entire range
        for i in 0..table_size {
            let log_energy = min_log_energy + (i as f64 / scale_factor);
            let color = Self::calculate_color_direct(log_energy);
            colors.push(color);
        }

        Self {
            colors,
            min_log_energy,
            max_log_energy,
            scale_factor,
        }
    }

    /// Direct color calculation (used during table generation)
    fn calculate_color_direct(clamped_log: f64) -> Rgb<u8> {
        if clamped_log < 17.0 {
            // Very low energy (10^15 to 10^17): black to blue
            let ratio = (clamped_log - 15.0) / 2.0;
            Rgb([0, 0, (255.0 * ratio.max(0.0).min(1.0)) as u8])
        } else if clamped_log < 19.0 {
            // Low energy (10^17 to 10^19): blue to green
            let ratio = (clamped_log - 17.0) / 2.0;
            Rgb([0, (255.0 * ratio) as u8, (255.0 * (1.0 - ratio)) as u8])
        } else if clamped_log < 21.0 {
            // Medium-low energy (10^19 to 10^21): green to red
            let ratio = (clamped_log - 19.0) / 2.0;
            Rgb([(255.0 * ratio) as u8, (255.0 * (1.0 - ratio)) as u8, 0])
        } else if clamped_log < 23.0 {
            // Medium-high energy (10^21 to 10^23): red to yellow
            let ratio = (clamped_log - 21.0) / 2.0;
            Rgb([255, (255.0 * ratio) as u8, 0])
        } else {
            // High energy (10^23 to 10^25): yellow to white
            let ratio = (clamped_log - 23.0) / 2.0;
            Rgb([255, 255, (255.0 * ratio.max(0.0).min(1.0)) as u8])
        }
    }

    /// Fast color lookup using pre-calculated table
    fn get_color(&self, energy_density_j_per_km3: f64) -> Rgb<u8> {
        // Calculate log energy (same as original method)
        let log_energy = if energy_density_j_per_km3 > 0.0 {
            energy_density_j_per_km3.log10()
        } else {
            0.0
        };

        // Clamp to table range
        let clamped_log = log_energy.max(self.min_log_energy).min(self.max_log_energy);

        // Convert to table index
        let index = ((clamped_log - self.min_log_energy) * self.scale_factor).round() as usize;
        let safe_index = index.min(self.colors.len() - 1);

        // Fast lookup!
        self.colors[safe_index]
    }
}

/// Custom heat map export operation for PNG generation
/// Exports temperature data as PNG images every simulation step
struct ThermalHeatMapExportOp {
    graphics_generator: H3GraphicsGenerator,
    export_every_step: bool,
    color_lookup: ColorLookupTable,
}

/// Specialized radiance layer visualization operation  
/// Exports radiance-specific data as PNG images to a subfolder
struct RadianceVisualizationOp {
    graphics_generator: H3GraphicsGenerator,
    export_every_step: bool,
    color_lookup: ColorLookupTable,
}

impl ThermalHeatMapExportOp {
    pub fn new(resolution: Resolution, points_per_degree: u32, export_every_step: bool) -> Self {
        let config = H3GraphicsConfig::new(resolution, points_per_degree);
        let mut generator = H3GraphicsGenerator::new(config);
        generator.load_cells();

        // Create color lookup table for energy density range 10^15 to 10^25 J/km³
        // Using 65536 entries for smooth color gradients (16-bit precision)
        let color_lookup = ColorLookupTable::new(15.0, 25.0, 65536);

        Self {
            graphics_generator: generator,
            export_every_step,
            color_lookup,
        }
    }

    /// Calculate energy density (J/km³) for non-foundry, non-atmospheric layers plus plume energy
    fn calculate_energy_density(
        &self,
        cell: &atmo_asth_rust::global_thermal::sim_cell::SimCell,
    ) -> f64 {
        let mut total_energy = 0.0;
        let mut total_volume_km3 = 0.0;

        // Include energy from thermal layers (non-atmospheric, INCLUDING foundry)
        for (layer, _) in &cell.layers_t {
            // Skip atmospheric layers (Air material)
            if layer.energy_mass.material_composite_type() == MaterialCompositeType::Air {
                continue;
            }

            // INCLUDE foundry layers since that's where radiance energy goes!

            let layer_energy = layer.energy_mass.energy_joules;
            let layer_volume_km3 = layer.height_km * layer.surface_area_km2;

            total_energy += layer_energy;
            total_volume_km3 += layer_volume_km3;
        }

        // Include energy from active heat plumes
        for plume in &cell.plumes.plumes {
            let plume_energy = plume.energy_joules;
            // Estimate plume volume based on radius and current layer height
            let plume_radius_km = plume.radius_m / 1000.0;
            let plume_volume_km3 = std::f64::consts::PI * plume_radius_km * plume_radius_km * 10.0; // Assume 10km height

            total_energy += plume_energy;
            total_volume_km3 += plume_volume_km3;
        }

        if total_volume_km3 > 0.0 {
            total_energy / total_volume_km3 // Energy density in J/km³
        } else {
            0.0
        }
    }

    /// Convert energy density to RGB color using fast lookup table
    fn energy_density_to_rgb(&self, energy_density_j_per_km3: f64) -> Rgb<u8> {
        // Fast O(1) color lookup using pre-calculated table
        self.color_lookup.get_color(energy_density_j_per_km3)
    }

    /// Export heat map as PNG image using existing H3 graphics system with radiance source circles
    fn export_heat_map_png(
        &self,
        sim: &Simulation,
        step: i32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let filename = format!("examples/thermal_heat_map/heat_map_step_{:04}.png", step);

        // Create color map for all cells based on temperature
        let mut cell_colors: HashMap<CellIndex, Rgb<u8>> = HashMap::new();

        for (cell_index, cell) in &sim.cells {
            let energy_density = self.calculate_energy_density(cell);
            let color = self.energy_density_to_rgb(energy_density);
            cell_colors.insert(*cell_index, color);
        }

        // Generate PNG silently - suppress stdout during graphics generation
        self.generate_custom_png_with_radiance_circles(&filename, cell_colors, sim)?;
        Ok(())
    }

    /// Extract radiance sources from simulation
    fn extract_radiance_sources(&self, sim: &Simulation) -> Vec<RadianceSource> {
        // Calculate current simulation year
        let current_year = sim.current_step() as f64 * sim.years_per_step as f64;

        // Find RadianceOp in simulation ops and extract sources
        for op in sim.ops.iter() {
            if op.name() == "RadianceOp" {
                // Try to downcast to RadianceOp
                if let Some(radiance_op) = op.as_any().downcast_ref::<RadianceOp>() {
                    return radiance_op.get_radiance_sources(current_year);
                }
            }
        }

        Vec::new() // Return empty if RadianceOp not found
    }

    /// Generate heat map PNG with radiance source circles and no console output
    fn generate_custom_png_with_radiance_circles(
        &self,
        filename: &str,
        cell_colors: HashMap<CellIndex, Rgb<u8>>,
        sim: &Simulation,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use image::{ImageBuffer, RgbImage};

        // Create image buffer
        let mut image: RgbImage = ImageBuffer::new(
            self.graphics_generator.config.width,
            self.graphics_generator.config.height,
        );

        // Fill with black background
        for pixel in image.pixels_mut() {
            *pixel = Rgb([0, 0, 0]);
        }

        // Draw cells silently
        let width_threshold = (self.graphics_generator.config.width as i32) * 20 / 100;

        for cell in &self.graphics_generator.cells {
            // Get color for this cell
            let cell_color = cell_colors
                .get(&cell.cell_index)
                .copied()
                .unwrap_or(Rgb([128, 128, 128]));

            // Convert cell corners to pixel coordinates
            let pixel_coords: Vec<(i32, i32)> = cell
                .corners
                .iter()
                .map(|corner| {
                    self.graphics_generator
                        .geo_to_pixel(corner.longitude, corner.latitude)
                })
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
                let coords: Vec<(i32, i32)> = pixel_coords
                    .iter()
                    .filter(|(x, y)| {
                        *x >= 0
                            && *x < self.graphics_generator.config.width as i32
                            && *y >= 0
                            && *y < self.graphics_generator.config.height as i32
                    })
                    .cloned()
                    .collect();

                if coords.len() >= 3 {
                    self.fill_polygon_simple(&mut image, &coords, cell_color);
                }
            }
        }

        // Draw radiance source circles
        let radiance_sources = self.extract_radiance_sources(sim);

        if !radiance_sources.is_empty() {
            // Draw real radiance sources
            self.draw_radiance_circles(&mut image, &radiance_sources, sim)?;
        } else {
            // Create circles based on actual energy hot spots in the simulation
            let energy_based_sources = self.create_energy_based_radiance_sources(sim);
            self.draw_radiance_circles(&mut image, &energy_based_sources, sim)?;
        }

        // Save the image
        image.save(filename)?;
        Ok(())
    }

    /// Create radiance sources based on actual energy hot spots in the simulation
    fn create_energy_based_radiance_sources(&self, sim: &Simulation) -> Vec<RadianceSource> {
        let mut energy_sources = Vec::new();

        // Calculate energy density for all cells and find the hottest ones
        let mut cell_energies: Vec<(CellIndex, f64)> = sim
            .cells
            .iter()
            .map(|(cell_index, cell)| {
                let energy_density = self.calculate_energy_density(cell);
                (*cell_index, energy_density)
            })
            .collect();

        // Sort by energy density (highest first)
        cell_energies.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take the top energy hot spots as radiance sources
        let hot_spot_count = (cell_energies.len() / 50).min(100).max(10); // 2% of cells, between 10-100

        for (i, (cell_index, energy_density)) in
            cell_energies.iter().take(hot_spot_count).enumerate()
        {
            if *energy_density > 1e15 {
                // Only show significant energy densities
                // Use energy density to determine wattage (higher energy = bigger circle)
                let wattage = (*energy_density / 1e18).min(10000.0).max(100.0); // Scale to reasonable wattage range

                energy_sources.push(RadianceSource {
                    cell_index: *cell_index,
                    wattage,
                    source_type: RadianceSourceType::Inflow, // Show all as inflows (white circles)
                });
            }
        }

        energy_sources
    }

    /// Create test radiance sources for debugging
    fn create_test_radiance_sources(&self, sim: &Simulation) -> Vec<RadianceSource> {
        let mut test_sources = Vec::new();

        // Get a few random cells to place test circles
        let cell_indices: Vec<_> = sim.cells.keys().take(5).cloned().collect();

        for (i, cell_index) in cell_indices.iter().enumerate() {
            let source_type = if i % 2 == 0 {
                RadianceSourceType::Inflow
            } else {
                RadianceSourceType::Outflow
            };
            let wattage = 1000.0 + (i as f64 * 2000.0); // Varying wattage from 1000 to 9000 MW

            test_sources.push(RadianceSource {
                cell_index: *cell_index,
                wattage,
                source_type,
            });
        }

        test_sources
    }

    /// Draw radiance circles on the image
    fn draw_radiance_circles(
        &self,
        image: &mut RgbImage,
        radiance_sources: &[RadianceSource],
        sim: &Simulation,
    ) -> Result<(), Box<dyn std::error::Error>> {
        for source in radiance_sources {
            // Get the cell location
            if let Some(cell) = sim.cells.get(&source.cell_index) {
                // Calculate cell center
                let lat_lng = LatLng::from(cell.h3_index);
                let lat = lat_lng.lat();
                let lng = lat_lng.lng();

                // Convert to pixel coordinates
                let (pixel_x, pixel_y) = self.graphics_generator.geo_to_pixel(lng, lat);

                // Calculate circle radius proportional to wattage
                let radius = self.calculate_circle_radius(source.wattage);

                // Choose color based on source type
                let color = match source.source_type {
                    RadianceSourceType::Inflow => Rgb([255, 255, 255]), // White for inflows
                    RadianceSourceType::Outflow => Rgb([0, 0, 0]),      // Black for outflows
                };

                // Draw circle
                self.draw_circle(image, pixel_x, pixel_y, radius, color);
            }
        }

        Ok(())
    }

    /// Calculate circle radius based on wattage
    fn calculate_circle_radius(&self, wattage: f64) -> u32 {
        // Scale radius based on wattage (adjust these values as needed)
        let min_radius = 8;
        let max_radius = 40;

        // Normalize wattage to 0-1 range (assuming max ~10,000 MW)
        let normalized_wattage = (wattage / 10000.0).min(1.0).max(0.0);

        (min_radius as f64 + normalized_wattage * (max_radius - min_radius) as f64) as u32
    }

    /// Draw a circle outline on the image
    fn draw_circle(
        &self,
        image: &mut RgbImage,
        center_x: i32,
        center_y: i32,
        radius: u32,
        color: Rgb<u8>,
    ) {
        let radius_i32 = radius as i32;

        // Draw circle outline using Bresenham's circle algorithm
        let mut x = 0;
        let mut y = radius_i32;
        let mut d = 3 - 2 * radius_i32;

        self.draw_circle_points(image, center_x, center_y, x, y, color);

        while y >= x {
            x += 1;

            if d > 0 {
                y -= 1;
                d = d + 4 * (x - y) + 10;
            } else {
                d = d + 4 * x + 6;
            }

            self.draw_circle_points(image, center_x, center_y, x, y, color);
        }
    }

    /// Helper function to draw 8 symmetric points of a circle
    fn draw_circle_points(
        &self,
        image: &mut RgbImage,
        center_x: i32,
        center_y: i32,
        x: i32,
        y: i32,
        color: Rgb<u8>,
    ) {
        let points = [
            (center_x + x, center_y + y),
            (center_x - x, center_y + y),
            (center_x + x, center_y - y),
            (center_x - x, center_y - y),
            (center_x + y, center_y + x),
            (center_x - y, center_y + x),
            (center_x + y, center_y - x),
            (center_x - y, center_y - x),
        ];

        for (px, py) in points {
            if px >= 0
                && px < self.graphics_generator.config.width as i32
                && py >= 0
                && py < self.graphics_generator.config.height as i32
            {
                image.put_pixel(px as u32, py as u32, color);
            }
        }
    }

    /// Simple polygon fill using basic scanline algorithm (copied from h3o_png.rs)
    fn fill_polygon_simple(&self, image: &mut RgbImage, coords: &[(i32, i32)], color: Rgb<u8>) {
        if coords.len() < 3 {
            return;
        }

        // Find bounding box
        let min_y = coords.iter().map(|(_, y)| *y).min().unwrap_or(0).max(0);
        let max_y = coords
            .iter()
            .map(|(_, y)| *y)
            .max()
            .unwrap_or(0)
            .min(self.graphics_generator.config.height as i32 - 1);

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
                    let x1 = chunk[0]
                        .max(0)
                        .min(self.graphics_generator.config.width as i32 - 1);
                    let x2 = chunk[1]
                        .max(0)
                        .min(self.graphics_generator.config.width as i32 - 1);

                    for x in x1..=x2 {
                        if x >= 0
                            && x < self.graphics_generator.config.width as i32
                            && y >= 0
                            && y < self.graphics_generator.config.height as i32
                        {
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

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn init_sim(&mut self, sim: &mut Simulation) {
        // Clean up any existing PNG files in the output directory
        self.cleanup_old_images();
       // panic!("images cleared right");
        // Export initial heat map
        let _ = self.export_heat_map_png(sim, 0);
    }

    fn update_sim(&mut self, sim: &mut Simulation) {
        if self.export_every_step || sim.current_step() % 10 == 0 {
            let current_step = sim.current_step();
            let _ = self.export_heat_map_png(sim, current_step);
        }
    }
}

impl RadianceVisualizationOp {
    pub fn new(resolution: Resolution, points_per_degree: u32, export_every_step: bool) -> Self {
        let config = H3GraphicsConfig::new(resolution, points_per_degree);
        let mut generator = H3GraphicsGenerator::new(config);
        generator.load_cells();

        // Create color lookup table for radiance energy density range 10^15 to 10^30 J/km³
        // Using 65536 entries for smooth color gradients (16-bit precision)
        let color_lookup = ColorLookupTable::new(15.0, 30.0, 65536);

        Self {
            graphics_generator: generator,
            export_every_step,
            color_lookup,
        }
    }

    /// Calculate radiance-focused energy density (J/km³) for all non-atmospheric layers plus plume energy
    fn calculate_radiance_energy_density(
        &self,
        cell: &atmo_asth_rust::global_thermal::sim_cell::SimCell,
    ) -> f64 {
        let mut total_energy = 0.0;
        let mut total_volume_km3 = 0.0;

        // Include energy from all solid layers
        for (layer, _) in &cell.layers_t {
            // Skip atmospheric layers (Air material) but include all solid layers
            if layer.energy_mass.material_composite_type() != MaterialCompositeType::Air {
                let layer_energy = layer.energy_mass.energy_joules;
                let layer_volume_km3 = layer.height_km * layer.surface_area_km2;

                total_energy += layer_energy;
                total_volume_km3 += layer_volume_km3;
            }
        }

        // Include energy from active heat plumes (this is where most radiance energy goes!)
        for plume in &cell.plumes.plumes {
            let plume_energy = plume.energy_joules;
            // Estimate plume volume based on radius and current layer height
            let plume_radius_km = plume.radius_m / 1000.0;
            let plume_volume_km3 = std::f64::consts::PI * plume_radius_km * plume_radius_km * 10.0; // Assume 10km height

            total_energy += plume_energy;
            total_volume_km3 += plume_volume_km3;
        }

        if total_volume_km3 > 0.0 {
            total_energy / total_volume_km3 // Energy density in J/km³
        } else {
            0.0 // No solid layers or plumes
        }
    }

    /// Convert energy density to RGB color using fast lookup table optimized for radiance visualization
    fn radiance_energy_density_to_rgb(&self, energy_density_j_per_km3: f64) -> Rgb<u8> {
        // Fast O(1) color lookup using pre-calculated table
        self.color_lookup.get_color(energy_density_j_per_km3)
    }

    /// Export radiance-specific heat map using existing PNG generation infrastructure
    fn export_radiance_png(
        &self,
        sim: &Simulation,
        step: i32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let filename = format!(
            "examples/thermal_heat_map/radiance_visualization/radiance_all_solid_step_{:04}.png",
            step
        );

        // Create output directory if it doesn't exist
        if let Some(parent) = std::path::Path::new(&filename).parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Create color mapping for each cell based on solid layer temperature
        let mut cell_colors = HashMap::new();

        for h3_cell in &self.graphics_generator.cells {
            if let Some(sim_cell) = sim.cells.get(&h3_cell.cell_index) {
                let energy_density = self.calculate_radiance_energy_density(sim_cell);
                let cell_color = self.radiance_energy_density_to_rgb(energy_density);
                cell_colors.insert(h3_cell.cell_index, cell_color);
            } else {
                // Default to black for cells not in simulation
                cell_colors.insert(h3_cell.cell_index, Rgb([0, 0, 0]));
            }
        }

        // Generate PNG with custom method to avoid console output and add radiance circles
        self.generate_custom_png_with_radiance_circles(&filename, cell_colors, sim)?;

        Ok(())
    }

    /// Clean up old radiance PNG files from previous runs
    fn cleanup_old_radiance_images(&self) {
        use std::fs;

        let output_dir = "examples/thermal_heat_map/radiance_visualization/";

        // Read directory and remove any PNG files
        if let Ok(entries) = fs::read_dir(output_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(extension) = path.extension() {
                    if extension == "png" {
                        if let Some(file_name) = path.file_name() {
                            if file_name
                                .to_string_lossy()
                                .starts_with("radiance_all_solid_step_")
                            {
                                let _ = fs::remove_file(&path); // Ignore errors
                            }
                        }
                    }
                }
            }
        }
    }

    /// Extract radiance sources from simulation (same as ThermalHeatMapExportOp)
    fn extract_radiance_sources(&self, sim: &Simulation) -> Vec<RadianceSource> {
        // Calculate current simulation year
        let current_year = sim.current_step() as f64 * sim.years_per_step as f64;

        // Find RadianceOp in simulation ops and extract sources
        for op in &sim.ops {
            if op.name() == "RadianceOp" {
                // Try to downcast to RadianceOp
                if let Some(radiance_op) = op.as_any().downcast_ref::<RadianceOp>() {
                    return radiance_op.get_radiance_sources(current_year);
                }
            }
        }

        Vec::new() // Return empty if RadianceOp not found
    }

    /// Generate radiance PNG with custom method to avoid console output
    fn generate_custom_png_with_radiance_circles(
        &self,
        filename: &str,
        cell_colors: HashMap<CellIndex, Rgb<u8>>,
        sim: &Simulation,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use image::{ImageBuffer, RgbImage};

        // Create image buffer
        let mut image: RgbImage = ImageBuffer::new(
            self.graphics_generator.config.width,
            self.graphics_generator.config.height,
        );

        // Fill with black background
        for pixel in image.pixels_mut() {
            *pixel = Rgb([0, 0, 0]);
        }

        // Draw cells silently
        let width_threshold = (self.graphics_generator.config.width as i32) * 20 / 100;

        for cell in &self.graphics_generator.cells {
            // Get color for this cell
            let cell_color = cell_colors
                .get(&cell.cell_index)
                .copied()
                .unwrap_or(Rgb([128, 128, 128]));

            // Convert cell corners to pixel coordinates
            let pixel_coords: Vec<(i32, i32)> = cell
                .corners
                .iter()
                .map(|corner| {
                    self.graphics_generator
                        .geo_to_pixel(corner.longitude, corner.latitude)
                })
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
                let coords: Vec<(i32, i32)> = pixel_coords
                    .iter()
                    .filter(|(x, y)| {
                        *x >= 0
                            && *x < self.graphics_generator.config.width as i32
                            && *y >= 0
                            && *y < self.graphics_generator.config.height as i32
                    })
                    .cloned()
                    .collect();

                if coords.len() >= 3 {
                    self.fill_polygon_simple(&mut image, &coords, cell_color);
                }
            }
        }

        // Draw radiance source circles
        let radiance_sources = self.extract_radiance_sources(sim);
        self.draw_radiance_circles(&mut image, &radiance_sources, sim)?;

        // Save the image
        image.save(filename)?;
        Ok(())
    }

    /// Draw radiance circles on the image (same as ThermalHeatMapExportOp)
    fn draw_radiance_circles(
        &self,
        image: &mut RgbImage,
        radiance_sources: &[RadianceSource],
        sim: &Simulation,
    ) -> Result<(), Box<dyn std::error::Error>> {
        for source in radiance_sources {
            // Get the cell location
            if let Some(cell) = sim.cells.get(&source.cell_index) {
                // Calculate cell center
                let lat_lng = LatLng::from(cell.h3_index);
                let lat = lat_lng.lat();
                let lng = lat_lng.lng();

                // Convert to pixel coordinates
                let (pixel_x, pixel_y) = self.graphics_generator.geo_to_pixel(lng, lat);

                // Calculate circle radius proportional to wattage
                let radius = self.calculate_circle_radius(source.wattage);

                // Choose color based on source type
                let color = match source.source_type {
                    RadianceSourceType::Inflow => Rgb([255, 255, 255]), // White for inflows
                    RadianceSourceType::Outflow => Rgb([0, 0, 0]),      // Black for outflows
                };

                // Draw circle
                self.draw_circle(image, pixel_x, pixel_y, radius, color);
            }
        }

        Ok(())
    }

    /// Calculate circle radius based on wattage (same as ThermalHeatMapExportOp)
    fn calculate_circle_radius(&self, wattage: f64) -> u32 {
        // Scale radius based on wattage (adjust these values as needed)
        let min_radius = 8;
        let max_radius = 40;

        // Normalize wattage to 0-1 range (assuming max ~10,000 MW)
        let normalized_wattage = (wattage / 10000.0).min(1.0).max(0.0);

        (min_radius as f64 + normalized_wattage * (max_radius - min_radius) as f64) as u32
    }

    /// Draw a circle outline on the image (same as ThermalHeatMapExportOp)
    fn draw_circle(
        &self,
        image: &mut RgbImage,
        center_x: i32,
        center_y: i32,
        radius: u32,
        color: Rgb<u8>,
    ) {
        let radius_i32 = radius as i32;

        // Draw circle outline using Bresenham's circle algorithm
        let mut x = 0;
        let mut y = radius_i32;
        let mut d = 3 - 2 * radius_i32;

        self.draw_circle_points(image, center_x, center_y, x, y, color);

        while y >= x {
            x += 1;

            if d > 0 {
                y -= 1;
                d = d + 4 * (x - y) + 10;
            } else {
                d = d + 4 * x + 6;
            }

            self.draw_circle_points(image, center_x, center_y, x, y, color);
        }
    }

    /// Helper function to draw 8 symmetric points of a circle (same as ThermalHeatMapExportOp)
    fn draw_circle_points(
        &self,
        image: &mut RgbImage,
        center_x: i32,
        center_y: i32,
        x: i32,
        y: i32,
        color: Rgb<u8>,
    ) {
        let points = [
            (center_x + x, center_y + y),
            (center_x - x, center_y + y),
            (center_x + x, center_y - y),
            (center_x - x, center_y - y),
            (center_x + y, center_y + x),
            (center_x - y, center_y + x),
            (center_x + y, center_y - x),
            (center_x - y, center_y - x),
        ];

        for (px, py) in points {
            if px >= 0
                && px < self.graphics_generator.config.width as i32
                && py >= 0
                && py < self.graphics_generator.config.height as i32
            {
                image.put_pixel(px as u32, py as u32, color);
            }
        }
    }

    /// Simple polygon fill using basic scanline algorithm
    fn fill_polygon_simple(&self, image: &mut RgbImage, coords: &[(i32, i32)], color: Rgb<u8>) {
        if coords.len() < 3 {
            return;
        }

        // Find bounding box
        let min_y = coords.iter().map(|(_, y)| *y).min().unwrap_or(0).max(0);
        let max_y = coords
            .iter()
            .map(|(_, y)| *y)
            .max()
            .unwrap_or(0)
            .min(self.graphics_generator.config.height as i32 - 1);

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
                    let x1 = chunk[0]
                        .max(0)
                        .min(self.graphics_generator.config.width as i32 - 1);
                    let x2 = chunk[1]
                        .max(0)
                        .min(self.graphics_generator.config.width as i32 - 1);

                    for x in x1..=x2 {
                        if x >= 0
                            && x < self.graphics_generator.config.width as i32
                            && y >= 0
                            && y < self.graphics_generator.config.height as i32
                        {
                            image.put_pixel(x as u32, y as u32, color);
                        }
                    }
                }
            }
        }
    }
}

impl SimOp for RadianceVisualizationOp {
    fn name(&self) -> &str {
        "RadianceVisualization"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn init_sim(&mut self, sim: &mut Simulation) {
        // Clean up any existing PNG files in the radiance output directory
        self.cleanup_old_radiance_images();

        // Export initial radiance heat map
        let _ = self.export_radiance_png(sim, 0);
    }

    fn update_sim(&mut self, sim: &mut Simulation) {
        if self.export_every_step {
            let current_step = sim.current_step();
            let _ = self.export_radiance_png(sim, current_step);
        }
    }
}

pub fn run_global_thermal_radiance_with_heat_map() {
    // Create Earth planet with L2 resolution
    let planet = Planet::earth(Resolution::Two);

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
    };

    // Create simulation properties with RadianceOp and PNG heat map export
    let sim_props = SimProps {
        planet: planet.clone(),
        res: Resolution::Two,
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
                Resolution::Two,
                1,
                false,
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

    // Run simulation silently

    // Run simulation
    sim.run();
}

fn main() {
    run_global_thermal_radiance_with_heat_map();
}
