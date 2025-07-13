use atmo_asth_rust::h3o_png::{H3GraphicsConfig, H3GraphicsGenerator};
use h3o::{CellIndex, LatLng, Resolution};
use atmo_asth_rust::energy_mass_composite::{EnergyMassComposite, MaterialCompositeType};
use image::{Rgb, RgbImage};
use atmo_asth_rust::sim::simulation::Simulation;
use std::collections::HashMap;
use atmo_asth_rust::sim_op::radiance_op::{RadianceSource, RadianceSourceType};
use atmo_asth_rust::sim_op::{RadianceOp, SimOp};
use crate::export_heat_map::ColorLookupTable;

/// Specialized radiance layer visualization operation
/// Exports radiance-specific data as PNG images to a subfolder
pub struct RadianceVisualizationOp {
    graphics_generator: H3GraphicsGenerator,
    pub export_every_step: bool,
    pub color_lookup: ColorLookupTable,
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