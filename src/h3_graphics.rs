/// H3 Cell Network PNG Graphics Generator
/// 
/// Generates PNG graphics of H3 hexagonal cell networks with configurable resolution
/// and points per degree for longitude/latitude mapping.

use h3o::{CellIndex, LatLng, Resolution};
use image::{ImageBuffer, Rgb, RgbImage};
use imageproc::drawing::draw_polygon_mut;
use imageproc::point::Point;
use rand::Rng;
use rand::seq::SliceRandom;

use noise::{NoiseFn, Perlin};
use crate::h3_utils::H3Utils;

/// Configuration for H3 graphics generation
#[derive(Debug, Clone)]
pub struct H3GraphicsConfig {
    /// H3 resolution level (0-15)
    pub resolution: Resolution,
    /// Points per degree for longitude/latitude mapping
    pub points_per_degree: u32,
    /// Output image width in pixels (calculated from points_per_degree * 360)
    pub width: u32,
    /// Output image height in pixels (calculated from points_per_degree * 180)
    pub height: u32,
}

impl H3GraphicsConfig {
    /// Create a new graphics config with specified resolution and points per degree
    pub fn new(resolution: Resolution, points_per_degree: u32) -> Self {
        Self {
            resolution,
            points_per_degree,
            width: points_per_degree * 360,  // 360 degrees longitude
            height: points_per_degree * 180, // 180 degrees latitude
        }
    }
}

/// H3 cell corner point in longitude/latitude coordinates
#[derive(Debug, Clone)]
pub struct CellCorner {
    pub longitude: f64,
    pub latitude: f64,
}

/// 3D point on unit sphere
#[derive(Debug, Clone, Copy)]
pub struct Point3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Point3D {
    /// Calculate distance to another 3D point
    pub fn distance_to(&self, other: &Point3D) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Normalize to unit sphere
    pub fn normalize(&self) -> Point3D {
        let length = (self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
        if length > 0.0 {
            Point3D {
                x: self.x / length,
                y: self.y / length,
                z: self.z / length,
            }
        } else {
            *self
        }
    }

    /// Add random offset and renormalize to sphere
    pub fn randomize_on_sphere(&self, offset_radius: f64, rng: &mut impl rand::Rng) -> Point3D {
        // Add random offset in 3D space
        let offset_x = rng.random_range(-offset_radius..offset_radius);
        let offset_y = rng.random_range(-offset_radius..offset_radius);
        let offset_z = rng.random_range(-offset_radius..offset_radius);

        let offset_point = Point3D {
            x: self.x + offset_x,
            y: self.y + offset_y,
            z: self.z + offset_z,
        };

        // Project back to unit sphere
        offset_point.normalize()
    }
}

impl CellCorner {
    /// Convert lat/lon to 3D point on unit sphere
    pub fn to_3d(&self) -> Point3D {
        let lat_rad = self.latitude.to_radians();
        let lon_rad = self.longitude.to_radians();

        let cos_lat = lat_rad.cos();
        let sin_lat = lat_rad.sin();
        let cos_lon = lon_rad.cos();
        let sin_lon = lon_rad.sin();

        Point3D {
            x: cos_lat * cos_lon,
            y: cos_lat * sin_lon,
            z: sin_lat,
        }
    }
}

/// H3 cell with its corner points for graphics rendering
#[derive(Debug, Clone)]
pub struct H3CellGraphics {
    pub cell_index: CellIndex,
    pub center: CellCorner,
    pub corners: Vec<CellCorner>,
}

impl H3CellGraphics {
    /// Create H3CellGraphics from a cell index
    pub fn from_cell_index(cell_index: CellIndex) -> Self {
        // Get cell center
        let center_latlng = LatLng::from(cell_index);
        let center = CellCorner {
            longitude: center_latlng.lng_radians().to_degrees(),
            latitude: center_latlng.lat_radians().to_degrees(),
        };

        // Get cell boundary (corner points)
        let boundary = cell_index.boundary();
        let corners: Vec<CellCorner> = boundary
            .iter()
            .map(|latlng| CellCorner {
                longitude: latlng.lng_radians().to_degrees(),
                latitude: latlng.lat_radians().to_degrees(),
            })
            .collect();

        Self {
            cell_index,
            center,
            corners,
        }
    }
}

/// H3 Graphics Generator for creating PNG visualizations
pub struct H3GraphicsGenerator {
    config: H3GraphicsConfig,
    cells: Vec<H3CellGraphics>,
}

impl H3GraphicsGenerator {
    /// Create a new H3 graphics generator
    pub fn new(config: H3GraphicsConfig) -> Self {
        Self {
            config,
            cells: Vec::new(),
        }
    }

    /// Load all H3 cells at the configured resolution
    pub fn load_cells(&mut self) {
        self.cells.clear();
        
        // Iterate through all cells at the specified resolution
        for base_cell in CellIndex::base_cells() {
            for cell_index in base_cell.children(self.config.resolution) {
                let cell_graphics = H3CellGraphics::from_cell_index(cell_index);
                self.cells.push(cell_graphics);
            }
        }
        
        println!("Loaded {} H3 cells at resolution {:?}", 
                 self.cells.len(), self.config.resolution);
    }

    /// Convert longitude to pixel x coordinate with wraparound handling
    fn lon_to_x(&self, longitude: f64) -> i32 {
        // Normalize longitude to -180 to +180 range, handling wraparound coordinates
        let mut normalized_lon = longitude;
        while normalized_lon > 180.0 {
            normalized_lon -= 360.0;
        }
        while normalized_lon < -180.0 {
            normalized_lon += 360.0;
        }

        // Map to 0 to width pixels
        let normalized = (normalized_lon + 180.0) / 360.0;
        let x = (normalized * self.config.width as f64) as i32;

        // Ensure x is within bounds
        x.max(0).min(self.config.width as i32 - 1)
    }

    /// Convert latitude to pixel y coordinate
    fn lat_to_y(&self, latitude: f64) -> i32 {
        // Clamp latitude to valid range -90 to +90 degrees
        let clamped_lat = latitude.max(-90.0).min(90.0);

        // Map to height to 0 pixels (flip Y axis for image coordinates)
        let normalized = (90.0 - clamped_lat) / 180.0;
        let y = (normalized * self.config.height as f64) as i32;

        // Ensure y is within bounds
        y.max(0).min(self.config.height as i32 - 1)
    }

    /// Project 3D point to 2D using equirectangular projection (from XYZ coordinates)
    fn project_3d_to_pixel(&self, point: &Point3D) -> (i32, i32) {
        // Convert 3D point back to lat/lon
        let lat_rad = point.z.asin();
        let lon_rad = point.y.atan2(point.x);

        let latitude = lat_rad.to_degrees();
        let longitude = lon_rad.to_degrees();

        // Use existing projection method
        self.geo_to_pixel(longitude, latitude)
    }

    /// Project 3D point using orthographic projection (view from space)
    fn project_3d_orthographic(&self, point: &Point3D, center_lon: f64, center_lat: f64) -> Option<(i32, i32)> {
        // Rotate to center the view
        let center_lat_rad = center_lat.to_radians();
        let center_lon_rad = center_lon.to_radians();

        let cos_center_lat = center_lat_rad.cos();
        let sin_center_lat = center_lat_rad.sin();
        let cos_center_lon = center_lon_rad.cos();
        let sin_center_lon = center_lon_rad.sin();

        // Rotate point to center view
        let x_rot = point.x * cos_center_lon + point.y * sin_center_lon;
        let y_rot = -point.x * sin_center_lon * cos_center_lat + point.y * cos_center_lon * cos_center_lat + point.z * sin_center_lat;
        let z_rot = point.x * sin_center_lon * sin_center_lat - point.y * cos_center_lon * sin_center_lat + point.z * cos_center_lat;

        // Check if point is on visible hemisphere
        if z_rot < 0.0 {
            return None; // Point is on back side
        }

        // Project to 2D
        let scale = (self.config.width.min(self.config.height) as f64) * 0.4; // Scale factor
        let center_x = self.config.width as f64 / 2.0;
        let center_y = self.config.height as f64 / 2.0;

        let pixel_x = (center_x + x_rot * scale) as i32;
        let pixel_y = (center_y - y_rot * scale) as i32; // Flip Y

        Some((pixel_x, pixel_y))
    }

    /// Convert geographic coordinates to pixel coordinates
    pub fn geo_to_pixel(&self, longitude: f64, latitude: f64) -> (i32, i32) {
        (self.lon_to_x(longitude), self.lat_to_y(latitude))
    }

    /// Get all cell corner points as pixel coordinates
    pub fn get_cell_corner_pixels(&self) -> Vec<(i32, i32)> {
        let mut pixels = Vec::new();
        
        for cell in &self.cells {
            for corner in &cell.corners {
                let (x, y) = self.geo_to_pixel(corner.longitude, corner.latitude);
                // Only include pixels within image bounds
                if x >= 0 && x < self.config.width as i32 && 
                   y >= 0 && y < self.config.height as i32 {
                    pixels.push((x, y));
                }
            }
        }
        
        pixels
    }

    /// Get cell center points as pixel coordinates
    pub fn get_cell_center_pixels(&self) -> Vec<(i32, i32)> {
        let mut pixels = Vec::new();
        
        for cell in &self.cells {
            let (x, y) = self.geo_to_pixel(cell.center.longitude, cell.center.latitude);
            // Only include pixels within image bounds
            if x >= 0 && x < self.config.width as i32 && 
               y >= 0 && y < self.config.height as i32 {
                pixels.push((x, y));
            }
        }
        
        pixels
    }

    /// Get cell boundaries as connected pixel paths for drawing hexagons
    pub fn get_cell_boundary_paths(&self) -> Vec<Vec<(i32, i32)>> {
        let mut paths = Vec::new();

        for cell in &self.cells {
            let mut path = Vec::new();

            for corner in &cell.corners {
                let (x, y) = self.geo_to_pixel(corner.longitude, corner.latitude);
                path.push((x, y));
            }

            // Don't add duplicate closing point - imageproc handles polygon closure
            if !path.is_empty() {
                paths.push(path);
            }
        }

        paths
    }

    /// Get configuration
    pub fn config(&self) -> &H3GraphicsConfig {
        &self.config
    }

    /// Get number of loaded cells
    pub fn cell_count(&self) -> usize {
        self.cells.len()
    }

    /// Generate a PNG with random colored hexagon fills only (no boundary lines)
    pub fn generate_colored_hexagons_png(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        // Create image buffer
        let mut image: RgbImage = ImageBuffer::new(self.config.width, self.config.height);

        // Fill with black background
        for pixel in image.pixels_mut() {
            *pixel = Rgb([0, 0, 0]);
        }

        // Create random number generator
        let mut rng = rand::rng();

        println!("Drawing {} hexagons with colored fills only (no boundaries)...", self.cells.len());

        // Draw filled hexagons with random colors (no boundaries)
        let mut filled_count = 0;
        for path in self.get_cell_boundary_paths_with_wraparound() {
            if path.len() >= 3 {
                // Convert to coordinate pairs and remove duplicates
                let mut coords: Vec<(i32, i32)> = path.iter()
                    .filter(|(x, y)| *x >= 0 && *x < self.config.width as i32 &&
                                     *y >= 0 && *y < self.config.height as i32)
                    .cloned()
                    .collect();

                // Remove consecutive duplicate points
                coords.dedup();

                if coords.len() >= 3 {
                    // Generate random color
                    let random_color = Rgb([
                        rng.random_range(50..255),  // Avoid very dark colors
                        rng.random_range(50..255),
                        rng.random_range(50..255),
                    ]);

                    // Fill polygon using simple approach
                    self.fill_polygon_simple(&mut image, &coords, random_color);
                    filled_count += 1;
                }
            }
        }

        println!("Filled {} hexagons with colors (no boundary lines)", filled_count);

        // Save the image
        image.save(filename)?;
        println!("Generated colored hexagons PNG: {} ({}x{} pixels, {} cells)",
                 filename, self.config.width, self.config.height, self.cells.len());
        Ok(())
    }

    /// Generate colored hexagons PNG using 3D projection approach
    pub fn generate_colored_hexagons_3d_png(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        // Create image buffer
        let mut image: RgbImage = ImageBuffer::new(self.config.width, self.config.height);

        // Fill with black background
        for pixel in image.pixels_mut() {
            *pixel = Rgb([0, 0, 0]);
        }

        // Create random number generator
        let mut rng = rand::rng();

        println!("Drawing {} hexagons using 3D projection (should eliminate wraparound issues)...", self.cells.len());

        // Draw filled hexagons with random colors using 3D approach
        let mut filled_count = 0;
        for path in self.get_cell_boundary_paths_3d() {
            if path.len() >= 3 {
                // Filter coordinates to image bounds
                let mut coords: Vec<(i32, i32)> = path.iter()
                    .filter(|(x, y)| *x >= 0 && *x < self.config.width as i32 &&
                                     *y >= 0 && *y < self.config.height as i32)
                    .cloned()
                    .collect();

                // Remove consecutive duplicate points
                coords.dedup();

                if coords.len() >= 3 {
                    // Generate random color
                    let random_color = Rgb([
                        rng.random_range(50..255),  // Avoid very dark colors
                        rng.random_range(50..255),
                        rng.random_range(50..255),
                    ]);

                    // Fill polygon using simple approach
                    self.fill_polygon_simple(&mut image, &coords, random_color);
                    filled_count += 1;
                }
            }
        }

        println!("Filled {} hexagons with colors using 3D projection", filled_count);

        // Save the image
        image.save(filename)?;
        println!("Generated 3D projected hexagons PNG: {} ({}x{} pixels, {} cells)",
                 filename, self.config.width, self.config.height, self.cells.len());

        Ok(())
    }

    /// Generate a PNG with ONLY random colored hexagon backgrounds (no boundaries) for testing
    pub fn generate_colors_only_png(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        // Create image buffer
        let mut image: RgbImage = ImageBuffer::new(self.config.width, self.config.height);

        // Fill with black background
        for pixel in image.pixels_mut() {
            *pixel = Rgb([0, 0, 0]);
        }

        // Create random number generator
        let mut rng = rand::rng();

        println!("Drawing {} hexagons with ONLY colored fills (no boundaries)...", self.cells.len());

        // Draw ONLY filled hexagons with random colors (no boundaries)
        let mut filled_count = 0;
        for path in self.get_cell_boundary_paths_with_wraparound() {
            if path.len() >= 3 {
                // Convert to coordinate pairs and remove duplicates
                let mut coords: Vec<(i32, i32)> = path.iter()
                    .filter(|(x, y)| *x >= 0 && *x < self.config.width as i32 &&
                                     *y >= 0 && *y < self.config.height as i32)
                    .cloned()
                    .collect();

                // Remove consecutive duplicate points
                coords.dedup();

                if coords.len() >= 3 {
                    // Generate random bright color
                    let random_color = Rgb([
                        rng.random_range(100..255),  // Brighter colors for testing
                        rng.random_range(100..255),
                        rng.random_range(100..255),
                    ]);

                    // Fill polygon
                    self.fill_polygon_simple(&mut image, &coords, random_color);
                    filled_count += 1;

                    // Debug: print first few colors
                    if filled_count <= 5 {
                        println!("  Hexagon {}: color RGB({}, {}, {})",
                                filled_count, random_color.0[0], random_color.0[1], random_color.0[2]);
                    }
                }
            }
        }

        println!("Filled {} hexagons with colors (NO boundaries drawn)", filled_count);

        // Save the image
        image.save(filename)?;
        println!("Generated colors-only PNG: {} ({}x{} pixels, {} cells)",
                 filename, self.config.width, self.config.height, self.cells.len());
        Ok(())
    }

    /// Convenience function to generate H3 cell network PNG with specified parameters
    pub fn generate_h3_network_png(
        resolution: Resolution,
        points_per_degree: u32,
        filename: &str
    ) -> Result<(), Box<dyn std::error::Error>> {
        let config = H3GraphicsConfig::new(resolution, points_per_degree);
        let mut generator = H3GraphicsGenerator::new(config);
        generator.load_cells();
        generator.generate_png(filename)
    }

    /// Convenience function to generate colored hexagons PNG with specified parameters
    pub fn generate_colored_hexagons_network_png(
        resolution: Resolution,
        points_per_degree: u32,
        filename: &str
    ) -> Result<(), Box<dyn std::error::Error>> {
        let config = H3GraphicsConfig::new(resolution, points_per_degree);
        let mut generator = H3GraphicsGenerator::new(config);
        generator.load_cells();
        generator.generate_colored_hexagons_png(filename)
    }

    /// Generate colored hexagons PNG using 3D projection approach (should eliminate wraparound issues)
    pub fn generate_colored_hexagons_3d_network_png(
        resolution: Resolution,
        points_per_degree: u32,
        filename: &str
    ) -> Result<(), Box<dyn std::error::Error>> {
        let config = H3GraphicsConfig::new(resolution, points_per_degree);
        let mut generator = H3GraphicsGenerator::new(config);
        generator.load_cells();
        generator.generate_colored_hexagons_3d_png(filename)
    }

    /// Generate Voronoi upwelling pattern visualization
    pub fn generate_voronoi_upwelling_png(
        l3_resolution: Resolution,
        points_per_degree: u32,
        filename: &str
    ) -> Result<(), Box<dyn std::error::Error>> {
        let config = H3GraphicsConfig::new(l3_resolution, points_per_degree);
        let mut generator = H3GraphicsGenerator::new(config);
        generator.load_cells();
        generator.generate_voronoi_upwelling_visualization(filename)
    }

    /// Analyze energy distribution with histogram (no visualization)
    pub fn analyze_energy_distribution(
        l3_resolution: Resolution,
        points_per_degree: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let config = H3GraphicsConfig::new(l3_resolution, points_per_degree);
        let mut generator = H3GraphicsGenerator::new(config);
        generator.load_cells();
        generator.calculate_energy_distribution_histogram()
    }

    /// Analyze cellular heat diffusion with histogram
    pub fn analyze_cellular_heat_diffusion(
        l3_resolution: Resolution,
        points_per_degree: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let config = H3GraphicsConfig::new(l3_resolution, points_per_degree);
        let mut generator = H3GraphicsGenerator::new(config);
        generator.load_cells();
        generator.calculate_cellular_heat_diffusion_histogram()
    }

    /// Analyze individual hotspot energy propagation with histogram
    pub fn analyze_individual_hotspot_propagation(
        l3_resolution: Resolution,
        points_per_degree: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let config = H3GraphicsConfig::new(l3_resolution, points_per_degree);
        let mut generator = H3GraphicsGenerator::new(config);
        generator.load_cells();
        generator.calculate_individual_hotspot_propagation_histogram()
    }

    /// Analyze individual hotspot energy propagation using H3 grid neighbors
    pub fn analyze_individual_hotspot_propagation_cached(
        l3_resolution: Resolution,
        points_per_degree: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let config = H3GraphicsConfig::new(l3_resolution, points_per_degree);
        let mut generator = H3GraphicsGenerator::new(config);
        generator.load_cells();
        generator.calculate_individual_hotspot_propagation_histogram()
    }

    /// Generate L2 heat map PNG using H3 grid neighbors
    pub fn generate_l2_heat_map_png(
        l2_resolution: Resolution,
        points_per_degree: u32,
        filename: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let config = H3GraphicsConfig::new(l2_resolution, points_per_degree);
        let mut generator = H3GraphicsGenerator::new(config);
        generator.load_cells();
        generator.create_l2_heat_map_visualization(filename)
    }

    /// Generate L2 heat map with Perlin noise overlay
    pub fn generate_l2_perlin_heat_map_png(
        l2_resolution: Resolution,
        points_per_degree: u32,
        filename: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let config = H3GraphicsConfig::new(l2_resolution, points_per_degree);
        let mut generator = H3GraphicsGenerator::new(config);
        generator.load_cells();
        generator.create_l2_perlin_heat_map_visualization(filename)
    }





    /// Quick utility to generate level 3 colored hexagons with 3 points per degree
    pub fn generate_level3_colored_hexagons() -> Result<(), Box<dyn std::error::Error>> {
        Self::generate_colored_hexagons_network_png(
            Resolution::Three,
            3,
            "h3_level3_colored_3ppd.png"
        )
    }

    /// Simple polygon fill using scanline algorithm
    fn fill_polygon(&self, image: &mut RgbImage, points: &[Point<i32>], color: Rgb<u8>) {
        if points.len() < 3 {
            return;
        }

        // Find bounding box
        let min_y = points.iter().map(|p| p.y).min().unwrap_or(0).max(0);
        let max_y = points.iter().map(|p| p.y).max().unwrap_or(0).min(self.config.height as i32 - 1);

        // For each scanline
        for y in min_y..=max_y {
            let mut intersections = Vec::new();

            // Find intersections with polygon edges
            for i in 0..points.len() {
                let p1 = points[i];
                let p2 = points[(i + 1) % points.len()];

                // Check if edge crosses scanline
                if (p1.y <= y && p2.y > y) || (p2.y <= y && p1.y > y) {
                    // Calculate intersection x coordinate
                    let x = p1.x + ((y - p1.y) * (p2.x - p1.x)) / (p2.y - p1.y);
                    intersections.push(x);
                }
            }

            // Sort intersections
            intersections.sort();

            // Fill between pairs of intersections
            for chunk in intersections.chunks(2) {
                if chunk.len() == 2 {
                    let x1 = chunk[0].max(0);
                    let x2 = chunk[1].min(self.config.width as i32 - 1);

                    for x in x1..=x2 {
                        if x >= 0 && x < self.config.width as i32 {
                            image.put_pixel(x as u32, y as u32, color);
                        }
                    }
                }
            }
        }
    }

    /// Simpler polygon fill using bounding box approach
    fn fill_polygon_simple(&self, image: &mut RgbImage, coords: &[(i32, i32)], color: Rgb<u8>) {
        if coords.len() < 3 {
            return;
        }

        // Find bounding box
        let min_x = coords.iter().map(|(x, _)| *x).min().unwrap_or(0).max(0);
        let max_x = coords.iter().map(|(x, _)| *x).max().unwrap_or(0).min(self.config.width as i32 - 1);
        let min_y = coords.iter().map(|(_, y)| *y).min().unwrap_or(0).max(0);
        let max_y = coords.iter().map(|(_, y)| *y).max().unwrap_or(0).min(self.config.height as i32 - 1);

        // For each pixel in bounding box, check if it's inside the polygon
        for y in min_y..=max_y {
            for x in min_x..=max_x {
                if self.point_in_polygon(x, y, coords) {
                    image.put_pixel(x as u32, y as u32, color);
                }
            }
        }
    }

    /// Check if a point is inside a polygon using ray casting algorithm
    fn point_in_polygon(&self, x: i32, y: i32, coords: &[(i32, i32)]) -> bool {
        let mut inside = false;
        let n = coords.len();

        let mut j = n - 1;
        for i in 0..n {
            let (xi, yi) = coords[i];
            let (xj, yj) = coords[j];

            if ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi) {
                inside = !inside;
            }
            j = i;
        }

        inside
    }

    /// Check if a hexagon spans more than 10% of the image width (longitude wraparound)
    fn has_longitude_wraparound(&self, corners: &[CellCorner]) -> bool {
        if corners.is_empty() {
            return false;
        }

        let mut min_lon = corners[0].longitude;
        let mut max_lon = corners[0].longitude;

        for corner in corners {
            min_lon = min_lon.min(corner.longitude);
            max_lon = max_lon.max(corner.longitude);
        }

        let span_degrees = max_lon - min_lon;
        let span_percent = span_degrees / 360.0;

        span_percent > 0.1 // More than 10% of width
    }

    /// Adjust longitude coordinates for wraparound hexagons
    /// Move distant points closer and create duplicate on opposite side
    fn adjust_wraparound_corners(&self, corners: &[CellCorner]) -> (Vec<CellCorner>, Option<Vec<CellCorner>>) {
        if !self.has_longitude_wraparound(corners) {
            return (corners.to_vec(), None);
        }

        // Find the longitude center of mass
        let avg_lon: f64 = corners.iter().map(|c| c.longitude).sum::<f64>() / corners.len() as f64;

        let mut adjusted_corners = Vec::new();
        let mut duplicate_corners = Vec::new();

        for corner in corners {
            let mut adjusted_corner = corner.clone();
            let mut duplicate_corner = corner.clone();

            // If point is more than 180° away from average, it's on the wrong side
            let lon_diff = corner.longitude - avg_lon;

            if lon_diff > 180.0 {
                // Point is too far east, move it west by 360°
                adjusted_corner.longitude -= 360.0;
                // Duplicate stays at original position
            } else if lon_diff < -180.0 {
                // Point is too far west, move it east by 360°
                adjusted_corner.longitude += 360.0;
                // Duplicate stays at original position
            } else {
                // Point is close to center, create duplicate on opposite side
                if avg_lon > 0.0 {
                    duplicate_corner.longitude -= 360.0;
                } else {
                    duplicate_corner.longitude += 360.0;
                }
            }

            adjusted_corners.push(adjusted_corner);
            duplicate_corners.push(duplicate_corner);
        }

        (adjusted_corners, Some(duplicate_corners))
    }

    /// Get cell boundaries using 3D projection approach with wraparound fixing
    pub fn get_cell_boundary_paths_3d(&self) -> Vec<Vec<(i32, i32)>> {
        let mut paths = Vec::new();
        let max_width = (self.config.width as i32) / 2; // 50% of screen width
        let screen_width = self.config.width as i32;

        for cell in &self.cells {
            // Convert all corners to 3D points first
            let points_3d: Vec<Point3D> = cell.corners.iter()
                .map(|corner| corner.to_3d())
                .collect();

            if points_3d.is_empty() {
                continue;
            }

            // Project 3D points to 2D pixels
            let mut pixel_coords: Vec<(i32, i32)> = Vec::new();
            for point_3d in &points_3d {
                let (x, y) = self.project_3d_to_pixel(point_3d);
                pixel_coords.push((x, y));
            }

            if !pixel_coords.is_empty() {
                // Check if X span is too wide (wraparound case)
                let min_x = pixel_coords.iter().map(|(x, _)| *x).min().unwrap();
                let max_x = pixel_coords.iter().map(|(x, _)| *x).max().unwrap();
                let x_span = max_x - min_x;

                if x_span > max_width {
                    // Create polygons on both sides to fill the gap
                    let both_side_polygons = self.create_both_side_polygons(&pixel_coords, screen_width);
                    paths.extend(both_side_polygons);
                } else {
                    // Normal polygon - add as-is
                    paths.push(pixel_coords);
                }
            }
        }

        paths
    }

    /// Create polygons on both sides of the screen to fill wraparound gaps
    /// Ensures X coordinates are either all near one edge or past it, not crossing center
    fn create_both_side_polygons(&self, coords: &[(i32, i32)], screen_width: i32) -> Vec<Vec<(i32, i32)>> {
        if coords.len() < 3 {
            return vec![coords.to_vec()];
        }

        // Find the median X coordinate to determine the split point
        let mut x_coords: Vec<i32> = coords.iter().map(|(x, _)| *x).collect();
        x_coords.sort();
        let median_x = x_coords[x_coords.len() / 2];

        let mut result_polygons = Vec::new();
        let quarter_width = screen_width / 4;  // Use quarter-width as threshold

        // Create left-side polygon: all points should be on the left side of screen
        let mut left_polygon = Vec::new();
        for &(x, y) in coords {
            let adjusted_x = if x > median_x {
                x - screen_width  // Shift right-side points to the left
            } else {
                x  // Keep left-side points as-is
            };
            left_polygon.push((adjusted_x, y));
        }

        // Only add left polygon if all X coordinates are in the left region
        if left_polygon.len() >= 3 {
            let max_left_x = left_polygon.iter().map(|(x, _)| *x).max().unwrap();
            if max_left_x < screen_width - quarter_width {  // All points should be well on the left
                result_polygons.push(left_polygon);
            }
        }

        // Create right-side polygon: all points should be on the right side of screen
        let mut right_polygon = Vec::new();
        for &(x, y) in coords {
            let adjusted_x = if x <= median_x {
                x + screen_width  // Shift left-side points to the right
            } else {
                x  // Keep right-side points as-is
            };
            right_polygon.push((adjusted_x, y));
        }

        // Only add right polygon if all X coordinates are in the right region
        if right_polygon.len() >= 3 {
            let min_right_x = right_polygon.iter().map(|(x, _)| *x).min().unwrap();
            if min_right_x > quarter_width {  // All points should be well on the right
                result_polygons.push(right_polygon);
            }
        }

        // If we couldn't create valid side polygons, fall back to the original
        if result_polygons.is_empty() {
            result_polygons.push(coords.to_vec());
        }

        result_polygons
    }

    /// Get cell boundaries with proper wraparound splitting (original lat/lon approach)
    pub fn get_cell_boundary_paths_with_wraparound(&self) -> Vec<Vec<(i32, i32)>> {
        let mut paths = Vec::new();

        for cell in &self.cells {
            // Convert all corners to pixel coordinates first
            let mut pixel_coords: Vec<(i32, i32)> = Vec::new();
            for corner in &cell.corners {
                let (x, y) = self.geo_to_pixel(corner.longitude, corner.latitude);
                pixel_coords.push((x, y));
            }

            if pixel_coords.is_empty() {
                continue;
            }

            // Check if this polygon has unexpectedly wide X distribution
            let min_x = pixel_coords.iter().map(|(x, _)| *x).min().unwrap();
            let max_x = pixel_coords.iter().map(|(x, _)| *x).max().unwrap();
            let x_span = max_x - min_x;

            // If X span is wider than half the screen, it's a wraparound case
            if x_span > (self.config.width as i32 / 2) {
                // Split the polygon using your algorithm
                let split_paths = self.split_wraparound_polygon(&pixel_coords);
                paths.extend(split_paths);
            } else {
                // Normal polygon - add as-is
                paths.push(pixel_coords);
            }
        }

        paths
    }

    /// Split a wraparound polygon into proper left and right side polygons
    /// Following user's algorithm: find median, shift points by half screen width, duplicate both ways
    fn split_wraparound_polygon(&self, coords: &[(i32, i32)]) -> Vec<Vec<(i32, i32)>> {
        if coords.len() < 3 {
            return vec![];
        }

        // Find the median X coordinate
        let mut x_coords: Vec<i32> = coords.iter().map(|(x, _)| *x).collect();
        x_coords.sort();
        let median_x = x_coords[x_coords.len() / 2];

        let half_width = self.config.width as i32 / 2;

        // Split points into left and right of median
        let mut left_side = Vec::new();
        let mut right_side = Vec::new();

        for &(x, y) in coords {
            if x <= median_x {
                left_side.push((x, y));
            } else {
                right_side.push((x, y));
            }
        }

        let mut result_paths = Vec::new();

        // Create left-side polygon: left points + right points shifted left by half screen width
        if !left_side.is_empty() && !right_side.is_empty() {
            let mut left_polygon = left_side.clone();
            for &(x, y) in &right_side {
                left_polygon.push((x - half_width, y));
            }
            result_paths.push(left_polygon);
        }

        // Create right-side polygon: right points + left points shifted right by half screen width
        if !right_side.is_empty() && !left_side.is_empty() {
            let mut right_polygon = right_side.clone();
            for &(x, y) in &left_side {
                right_polygon.push((x + half_width, y));
            }
            result_paths.push(right_polygon);
        }

        // If we only have points on one side, just return the original polygon
        if left_side.is_empty() || right_side.is_empty() {
            result_paths.push(coords.to_vec());
        }

        result_paths
    }

    /// Generate a PNG image file showing H3 cell network
    pub fn generate_png(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        // Create image buffer
        let mut image: RgbImage = ImageBuffer::new(self.config.width, self.config.height);

        // Fill with black background
        for pixel in image.pixels_mut() {
            *pixel = Rgb([0, 0, 0]);
        }

        // Define colors
        let red = Rgb([255, 0, 0]);     // Corner points
        let blue = Rgb([0, 0, 255]);    // Center points
        let green = Rgb([0, 255, 0]);   // Cell boundaries

        // Draw cell boundaries first (so points appear on top)
        for path in self.get_cell_boundary_paths_with_wraparound() {
            if path.len() >= 3 {
                // Convert to Points for imageproc and remove duplicates
                let mut points: Vec<Point<i32>> = path.iter()
                    .filter(|(x, y)| *x >= 0 && *x < self.config.width as i32 &&
                                     *y >= 0 && *y < self.config.height as i32)
                    .map(|(x, y)| Point::new(*x, *y))
                    .collect();

                // Remove consecutive duplicate points
                points.dedup();

                // Remove duplicate first/last points
                if points.len() > 2 && points.first() == points.last() {
                    points.pop();
                }

                if points.len() >= 3 {
                    // Draw polygon outline
                    draw_polygon_mut(&mut image, &points, green);
                }
            }
        }

        // Draw cell corner points (with wraparound handling)
        for cell in &self.cells {
            let (adjusted_corners, duplicate_corners) = self.adjust_wraparound_corners(&cell.corners);

            // Draw adjusted corners
            for corner in &adjusted_corners {
                let (x, y) = self.geo_to_pixel(corner.longitude, corner.latitude);
                self.draw_point(&mut image, x, y, red);
            }

            // Draw duplicate corners if they exist
            if let Some(dup_corners) = duplicate_corners {
                for corner in &dup_corners {
                    let (x, y) = self.geo_to_pixel(corner.longitude, corner.latitude);
                    self.draw_point(&mut image, x, y, red);
                }
            }
        }

        // Draw cell center points
        for (x, y) in self.get_cell_center_pixels() {
            self.draw_point(&mut image, x, y, blue);
        }

        // Save the image
        image.save(filename)?;
        println!("Generated PNG image: {} ({}x{} pixels, {} cells)",
                 filename, self.config.width, self.config.height, self.cells.len());
        Ok(())
    }

    /// Draw a point (small cross) at the given coordinates
    fn draw_point(&self, image: &mut RgbImage, x: i32, y: i32, color: Rgb<u8>) {
        // Draw a small cross (3x3 pixels)
        for dx in -1..=1 {
            for dy in -1..=1 {
                let px = x + dx;
                let py = y + dy;
                if px >= 0 && px < self.config.width as i32 &&
                   py >= 0 && py < self.config.height as i32 {
                    // Draw cross pattern
                    if dx == 0 || dy == 0 {
                        image.put_pixel(px as u32, py as u32, color);
                    }
                }
            }
        }
    }

    /// Generate Voronoi upwelling pattern visualization
    pub fn generate_voronoi_upwelling_visualization(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        // Create image buffer
        let mut image: RgbImage = ImageBuffer::new(self.config.width, self.config.height);

        // Fill with black background
        for pixel in image.pixels_mut() {
            *pixel = Rgb([0, 0, 0]);
        }

        println!("Generating Voronoi upwelling pattern...");

        // Step 1: Generate L2 cell centers and create upwelling points
        let upwelling_points = self.generate_upwelling_points()?;
        println!("Generated {} upwelling points from L2 cells", upwelling_points.len());

        // Step 2: Calculate L2 radius for distance normalization
        let l2_radius = self.estimate_l2_radius();
        println!("Estimated L2 radius: {:.6}", l2_radius);

        // Step 3: For each L3 cell, find distance to nearest upwelling point and visualize
        let mut processed_count = 0;
        let mut distances = Vec::new();

        for cell in &self.cells {
            // Get cell center in 3D
            let cell_center_3d = self.get_cell_center_3d(cell);

            // Find distance to nearest upwelling point
            let min_distance = upwelling_points.iter()
                .map(|upwell_point| cell_center_3d.distance_to(upwell_point))
                .fold(f64::INFINITY, f64::min);

            // Collect distance for statistics
            distances.push(min_distance);

            // Normalize distance (0 = white, l2_radius = black)
            let normalized_distance = (min_distance / l2_radius).min(1.0);
            let gray_value = ((1.0 - normalized_distance) * 255.0) as u8;
            let color = Rgb([gray_value, gray_value, gray_value]);

            // Draw the cell with the calculated color
            self.draw_cell_with_color(&mut image, cell, color);
            processed_count += 1;
        }

        // Calculate and display distance statistics
        self.analyze_distance_statistics(&distances, l2_radius);

        println!("Processed {} L3 cells for Voronoi distances", processed_count);

        // Save the image
        image.save(filename)?;
        println!("Generated Voronoi upwelling PNG: {} ({}x{} pixels)",
                 filename, self.config.width, self.config.height);

        Ok(())
    }

    /// Generate upwelling points from L2 cells
    fn generate_upwelling_points(&self) -> Result<Vec<Point3D>, Box<dyn std::error::Error>> {
        // Load L2 cells
        let l2_cells = h3o::CellIndex::base_cells()
            .flat_map(|base_cell| base_cell.children(Resolution::Two))
            .collect::<Vec<_>>();

        println!("Loaded {} L2 cells", l2_cells.len());

        // Take half of them randomly
        let mut rng = rand::rng();
        let mut selected_cells = l2_cells;
        selected_cells.shuffle(&mut rng);
        selected_cells.truncate(selected_cells.len() / 2);

        println!("Selected {} L2 cells for upwelling", selected_cells.len());

        // Convert to 3D points and randomize
        let l2_radius = self.estimate_l2_radius();
        let mut upwelling_points = Vec::new();

        for cell in selected_cells {
            let center_ll = LatLng::from(cell);
            let center_3d = Point3D {
                x: center_ll.lat_radians().cos() * center_ll.lng_radians().cos(),
                y: center_ll.lat_radians().cos() * center_ll.lng_radians().sin(),
                z: center_ll.lat_radians().sin(),
            };

            // Randomize position by ±L2 radius
            let randomized_point = center_3d.randomize_on_sphere(l2_radius, &mut rng);
            upwelling_points.push(randomized_point);
        }

        Ok(upwelling_points)
    }

    /// Estimate L2 cell radius for distance calculations
    fn estimate_l2_radius(&self) -> f64 {
        // Approximate L2 cell radius on unit sphere
        // L2 has ~5,882 cells globally, so average area per cell ≈ 4π/5882
        // Radius ≈ sqrt(area/π) ≈ sqrt(4/5882) ≈ 0.026
        // Double the scale to reduce black areas and increase upwelling influence
        0.026 * 2.0
    }

    /// Get cell center as 3D point
    fn get_cell_center_3d(&self, cell: &H3CellGraphics) -> Point3D {
        // Use first corner as approximation (could be improved with actual center)
        if !cell.corners.is_empty() {
            cell.corners[0].to_3d()
        } else {
            Point3D { x: 0.0, y: 0.0, z: 1.0 } // Default to north pole
        }
    }

    /// Draw a single cell with specified color
    fn draw_cell_with_color(&self, image: &mut RgbImage, cell: &H3CellGraphics, color: Rgb<u8>) {
        // Convert cell corners to pixel coordinates
        let mut pixel_coords: Vec<(i32, i32)> = Vec::new();
        for corner in &cell.corners {
            let (x, y) = self.geo_to_pixel(corner.longitude, corner.latitude);
            pixel_coords.push((x, y));
        }

        if pixel_coords.len() >= 3 {
            // Check if this polygon has wraparound issues
            let min_x = pixel_coords.iter().map(|(x, _)| *x).min().unwrap();
            let max_x = pixel_coords.iter().map(|(x, _)| *x).max().unwrap();
            let x_span = max_x - min_x;
            let max_width = (self.config.width as i32) / 2;

            if x_span > max_width {
                // Handle wraparound case
                let polygons = self.create_both_side_polygons(&pixel_coords, self.config.width as i32);
                for polygon in polygons {
                    self.fill_polygon_with_bounds_check(image, &polygon, color);
                }
            } else {
                // Normal case
                self.fill_polygon_with_bounds_check(image, &pixel_coords, color);
            }
        }
    }

    /// Fill polygon with bounds checking
    fn fill_polygon_with_bounds_check(&self, image: &mut RgbImage, coords: &[(i32, i32)], color: Rgb<u8>) {
        // Filter coordinates to image bounds
        let filtered_coords: Vec<(i32, i32)> = coords.iter()
            .filter(|(x, y)| *x >= 0 && *x < self.config.width as i32 &&
                             *y >= 0 && *y < self.config.height as i32)
            .cloned()
            .collect();

        if filtered_coords.len() >= 3 {
            self.fill_polygon_simple(image, &filtered_coords, color);
        }
    }

    /// Analyze and display distance statistics
    fn analyze_distance_statistics(&self, distances: &[f64], l2_radius: f64) {
        if distances.is_empty() {
            println!("No distance data to analyze");
            return;
        }

        // Calculate basic statistics
        let min_distance = distances.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_distance = distances.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let mean_distance = distances.iter().sum::<f64>() / distances.len() as f64;

        // Calculate standard deviation
        let variance = distances.iter()
            .map(|&x| (x - mean_distance).powi(2))
            .sum::<f64>() / distances.len() as f64;
        let std_dev = variance.sqrt();

        // Calculate percentiles
        let mut sorted_distances = distances.to_vec();
        sorted_distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let median = sorted_distances[sorted_distances.len() / 2];
        let p25 = sorted_distances[sorted_distances.len() / 4];
        let p75 = sorted_distances[sorted_distances.len() * 3 / 4];
        let p90 = sorted_distances[sorted_distances.len() * 9 / 10];
        let p95 = sorted_distances[sorted_distances.len() * 95 / 100];

        println!("\n=== Voronoi Distance Statistics ===");
        println!("Sample size: {} L3 cells", distances.len());
        println!("L2 radius (scale): {:.6}", l2_radius);

        println!("\nRaw Distances (3D unit sphere):");
        println!("  Min:     {:.6}", min_distance);
        println!("  Max:     {:.6}", max_distance);
        println!("  Mean:    {:.6}", mean_distance);
        println!("  Median:  {:.6}", median);
        println!("  Std Dev: {:.6}", std_dev);

        println!("\nPercentiles:");
        println!("  25th:    {:.6}", p25);
        println!("  75th:    {:.6}", p75);
        println!("  90th:    {:.6}", p90);
        println!("  95th:    {:.6}", p95);

        println!("\nNormalized by L2 radius:");
        println!("  Min:     {:.3}x L2 radius", min_distance / l2_radius);
        println!("  Max:     {:.3}x L2 radius", max_distance / l2_radius);
        println!("  Mean:    {:.3}x L2 radius", mean_distance / l2_radius);
        println!("  Median:  {:.3}x L2 radius", median / l2_radius);
        println!("  Std Dev: {:.3}x L2 radius", std_dev / l2_radius);

        // Coverage analysis
        let within_1x = distances.iter().filter(|&&d| d <= l2_radius).count();
        let within_0_5x = distances.iter().filter(|&&d| d <= l2_radius * 0.5).count();
        let beyond_1x = distances.iter().filter(|&&d| d > l2_radius).count();

        println!("\nCoverage Analysis:");
        println!("  Within 0.5x L2 radius: {} cells ({:.1}%)",
                 within_0_5x, 100.0 * within_0_5x as f64 / distances.len() as f64);
        println!("  Within 1.0x L2 radius: {} cells ({:.1}%)",
                 within_1x, 100.0 * within_1x as f64 / distances.len() as f64);
        println!("  Beyond 1.0x L2 radius: {} cells ({:.1}%)",
                 beyond_1x, 100.0 * beyond_1x as f64 / distances.len() as f64);

        // Generate histogram with 0.1x L2 radius increments
        self.generate_distance_histogram(&distances, l2_radius);

        println!("========================================\n");
    }

    /// Generate and display histogram of distances in 0.1x L2 radius increments
    fn generate_distance_histogram(&self, distances: &[f64], l2_radius: f64) {
        println!("\n=== Distance Histogram (0.1x L2 radius increments) ===");

        // Create bins from 0.0 to 2.5x L2 radius in 0.1x increments
        let bin_size = 0.1 * l2_radius;
        let max_bins = 25; // 0.0 to 2.5x L2 radius
        let mut bins = vec![0; max_bins];

        // Count distances in each bin
        for &distance in distances {
            let bin_index = ((distance / bin_size).floor() as usize).min(max_bins - 1);
            bins[bin_index] += 1;
        }

        // Find max count for scaling the visual bars
        let max_count = *bins.iter().max().unwrap_or(&1);
        let bar_scale = 50.0 / max_count as f64; // Scale to max 50 characters

        println!("Range (×L2)  │ Count  │ Percent │ Histogram");
        println!("─────────────┼────────┼─────────┼─────────────────────────────────────────────────");

        for (i, &count) in bins.iter().enumerate() {
            if count == 0 && i > 20 { // Skip empty bins beyond 2.0x L2 radius
                continue;
            }

            let range_start = i as f64 * 0.1;
            let range_end = (i + 1) as f64 * 0.1;
            let percentage = 100.0 * count as f64 / distances.len() as f64;
            let bar_length = (count as f64 * bar_scale) as usize;
            let bar = "█".repeat(bar_length);

            println!("{:4.1}-{:4.1}   │ {:6} │ {:5.1}%  │ {}",
                     range_start, range_end, count, percentage, bar);
        }

        // Summary statistics for histogram
        let total_cells = distances.len();
        let cumulative_50 = self.find_cumulative_percentage(&bins, total_cells, 0.5);
        let cumulative_75 = self.find_cumulative_percentage(&bins, total_cells, 0.75);
        let cumulative_90 = self.find_cumulative_percentage(&bins, total_cells, 0.9);

        println!("─────────────┼────────┼─────────┼─────────────────────────────────────────────────");
        println!("Total: {} cells", total_cells);
        println!("50% of cells within: {:.1}x L2 radius", cumulative_50);
        println!("75% of cells within: {:.1}x L2 radius", cumulative_75);
        println!("90% of cells within: {:.1}x L2 radius", cumulative_90);
    }

    /// Find the distance at which a given percentage of cells are included
    fn find_cumulative_percentage(&self, bins: &[usize], total_cells: usize, target_percentage: f64) -> f64 {
        let target_count = (total_cells as f64 * target_percentage) as usize;
        let mut cumulative = 0;

        for (i, &count) in bins.iter().enumerate() {
            cumulative += count;
            if cumulative >= target_count {
                return (i + 1) as f64 * 0.1; // Return upper bound of bin
            }
        }

        2.5 // Default to max range if not found
    }

    /// Calculate energy distribution and generate histogram
    pub fn calculate_energy_distribution_histogram(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("=== Energy Distribution Analysis ===");
        println!("Calculating energy from upwelling sources with linear falloff");

        // Step 1: Generate upwelling energy sources
        let energy_sources = self.generate_energy_sources()?;
        println!("Generated {} energy sources", energy_sources.len());

        // Step 2: Calculate energy at each L3 cell from all sources
        let mut cell_energies = Vec::new();
        let l2_radius = self.estimate_l2_radius();

        println!("Calculating energy contributions for {} L3 cells...", self.cells.len());

        for cell in &self.cells {
            let cell_center_3d = self.get_cell_center_3d(cell);
            let mut total_energy = 0.0;

            // Sum energy contributions from all sources
            for source in &energy_sources {
                let distance = cell_center_3d.distance_to(&source.position);
                let energy_contribution = self.calculate_energy_contribution(distance, source.energy, l2_radius);
                total_energy += energy_contribution;
            }

            cell_energies.push(total_energy);
        }

        // Step 3: Generate histogram of energy values
        self.generate_energy_histogram(&cell_energies);

        Ok(())
    }

    /// Generate energy sources with random positions and energy levels
    fn generate_energy_sources(&self) -> Result<Vec<EnergySource>, Box<dyn std::error::Error>> {
        // Load L2 cells and select half randomly (same as upwelling points)
        let l2_cells = h3o::CellIndex::base_cells()
            .flat_map(|base_cell| base_cell.children(Resolution::Two))
            .collect::<Vec<_>>();

        let mut rng = rand::rng();
        let mut selected_cells = l2_cells;
        selected_cells.shuffle(&mut rng);
        selected_cells.truncate(selected_cells.len() / 2);

        let l2_radius = self.estimate_l2_radius();
        let mut energy_sources = Vec::new();

        for cell in selected_cells {
            let center_ll = LatLng::from(cell);
            let center_3d = Point3D {
                x: center_ll.lat_radians().cos() * center_ll.lng_radians().cos(),
                y: center_ll.lat_radians().cos() * center_ll.lng_radians().sin(),
                z: center_ll.lat_radians().sin(),
            };

            // Randomize position by ±L2 radius
            let randomized_position = center_3d.randomize_on_sphere(l2_radius, &mut rng);

            // Generate random energy between 0.04 and 0.4
            let energy = rng.random_range(0.04..0.4);

            energy_sources.push(EnergySource {
                position: randomized_position,
                energy,
            });
        }

        Ok(energy_sources)
    }

    /// Calculate energy contribution from a source with linear falloff
    /// - Double the influence radius
    /// - Flat energy until 0.25× radius, then linear falloff to zero
    fn calculate_energy_contribution(&self, distance: f64, source_energy: f64, l2_radius: f64) -> f64 {
        let max_influence_radius = l2_radius * source_energy * 2.0; // Double the radius
        let flat_core_radius = max_influence_radius * 0.25; // Flat energy until 25% of radius

        if distance >= max_influence_radius {
            0.0 // No contribution beyond influence radius
        } else if distance <= flat_core_radius {
            source_energy // Full energy in core region (0 to 0.25× radius)
        } else {
            // Linear falloff from flat_core_radius to max_influence_radius
            let falloff_distance = distance - flat_core_radius;
            let falloff_range = max_influence_radius - flat_core_radius;
            source_energy * (1.0 - falloff_distance / falloff_range)
        }
    }

    /// Generate and display histogram of energy values
    fn generate_energy_histogram(&self, energies: &[f64]) {
        println!("\n=== Energy Distribution Histogram ===");

        // Find energy range
        let min_energy = energies.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_energy = energies.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let mean_energy = energies.iter().sum::<f64>() / energies.len() as f64;

        println!("Energy range: {:.4} to {:.4}", min_energy, max_energy);
        println!("Mean energy: {:.4}", mean_energy);

        // Create bins with 0.02 energy increments (since max individual source is 0.4)
        let bin_size = 0.02;
        let max_bins = ((max_energy / bin_size).ceil() as usize + 1).min(50); // Cap at 50 bins
        let mut bins = vec![0; max_bins];

        // Count energies in each bin
        for &energy in energies {
            let bin_index = ((energy / bin_size).floor() as usize).min(max_bins - 1);
            bins[bin_index] += 1;
        }

        // Find max count for scaling
        let max_count = *bins.iter().max().unwrap_or(&1);
        let bar_scale = 50.0 / max_count as f64;

        println!("\nEnergy Range │ Count  │ Percent │ Histogram");
        println!("─────────────┼────────┼─────────┼─────────────────────────────────────────────────");

        for (i, &count) in bins.iter().enumerate() {
            if count == 0 && i > (mean_energy / bin_size) as usize + 10 {
                continue; // Skip empty bins far beyond mean
            }

            let range_start = i as f64 * bin_size;
            let range_end = (i + 1) as f64 * bin_size;
            let percentage = 100.0 * count as f64 / energies.len() as f64;
            let bar_length = (count as f64 * bar_scale) as usize;
            let bar = "█".repeat(bar_length);

            println!("{:4.2}-{:4.2}   │ {:6} │ {:5.1}%  │ {}",
                     range_start, range_end, count, percentage, bar);
        }

        // Calculate percentiles
        let mut sorted_energies = energies.to_vec();
        sorted_energies.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let median = sorted_energies[sorted_energies.len() / 2];
        let p25 = sorted_energies[sorted_energies.len() / 4];
        let p75 = sorted_energies[sorted_energies.len() * 3 / 4];
        let p90 = sorted_energies[sorted_energies.len() * 9 / 10];
        let p95 = sorted_energies[sorted_energies.len() * 95 / 100];

        println!("─────────────┼────────┼─────────┼─────────────────────────────────────────────────");
        println!("Statistics:");
        println!("  Min:     {:.4}", min_energy);
        println!("  25th:    {:.4}", p25);
        println!("  Median:  {:.4}", median);
        println!("  Mean:    {:.4}", mean_energy);
        println!("  75th:    {:.4}", p75);
        println!("  90th:    {:.4}", p90);
        println!("  95th:    {:.4}", p95);
        println!("  Max:     {:.4}", max_energy);

        // Zero energy analysis
        let zero_energy = energies.iter().filter(|&&e| e < 0.001).count();
        println!("  Zero energy cells: {} ({:.1}%)", zero_energy, 100.0 * zero_energy as f64 / energies.len() as f64);
    }
}

/// Energy source with position and energy level
#[derive(Debug, Clone)]
struct EnergySource {
    position: Point3D,
    energy: f64,
}

/// Cellular heat diffusion calculation and histogram
impl H3GraphicsGenerator {
    /// Calculate cellular heat diffusion and generate histogram
    pub fn calculate_cellular_heat_diffusion_histogram(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("=== Cellular Heat Diffusion Analysis ===");
        println!("Discrete hotspots with heat radiating through neighboring cells");

        // Step 1: Place hotspots in specific cells
        let hotspot_cells = self.generate_hotspot_cells()?;
        println!("Generated {} hotspot cells", hotspot_cells.len());

        // Step 2: Create cell index mapping for neighbor lookup
        let cell_map = self.create_cell_index_map();

        // Step 3: Initialize energy array for all cells
        let mut cell_energies = vec![0.0; self.cells.len()];

        // Step 4: Set initial energy for hotspot cells
        for hotspot in &hotspot_cells {
            if let Some(&cell_index) = cell_map.get(&hotspot.cell_index) {
                cell_energies[cell_index] = hotspot.energy;
            }
        }

        // Step 5: Radiate heat through neighbors (5 iterations)
        for iteration in 1..=5 {
            cell_energies = self.radiate_heat_to_neighbors(&cell_energies, &cell_map, iteration);
        }

        // Step 6: Generate histogram of final energy values
        self.generate_cellular_energy_histogram(&cell_energies, &hotspot_cells);

        Ok(())
    }

    /// Generate hotspot cells with random positions and energy levels
    fn generate_hotspot_cells(&self) -> Result<Vec<HotspotCell>, Box<dyn std::error::Error>> {
        // Load L2 cells and select half randomly
        let l2_cells = h3o::CellIndex::base_cells()
            .flat_map(|base_cell| base_cell.children(Resolution::Two))
            .collect::<Vec<_>>();

        let mut rng = rand::rng();
        let mut selected_cells = l2_cells;
        selected_cells.shuffle(&mut rng);
        selected_cells.truncate(selected_cells.len() / 2);

        let mut hotspot_cells = Vec::new();

        for l2_cell in selected_cells {
            // Find the L3 cell that contains this L2 center
            let l2_center = LatLng::from(l2_cell);

            // Find closest L3 cell to this L2 center
            let mut closest_cell_index = 0;
            let mut min_distance = f64::INFINITY;

            for (i, l3_cell) in self.cells.iter().enumerate() {
                let l3_center_distance = self.calculate_latlng_distance(
                    l2_center.lat_radians().to_degrees(), l2_center.lng_radians().to_degrees(),
                    l3_cell.center.latitude, l3_cell.center.longitude
                );

                if l3_center_distance < min_distance {
                    min_distance = l3_center_distance;
                    closest_cell_index = i;
                }
            }

            // Generate random energy between 0.04 and 0.4
            let energy = rng.random_range(0.04..0.4);

            hotspot_cells.push(HotspotCell {
                cell_index: closest_cell_index,
                energy,
            });
        }

        Ok(hotspot_cells)
    }

    /// Create mapping from cell array index to H3 cell index for neighbor lookup
    fn create_cell_index_map(&self) -> std::collections::HashMap<usize, usize> {
        let mut map = std::collections::HashMap::new();
        for (i, _cell) in self.cells.iter().enumerate() {
            map.insert(i, i);
        }
        map
    }

    /// Radiate 75% of heat to neighboring cells
    fn radiate_heat_to_neighbors(
        &self,
        current_energies: &[f64],
        _cell_map: &std::collections::HashMap<usize, usize>,
        iteration: usize
    ) -> Vec<f64> {
        let mut new_energies = current_energies.to_vec();

        println!("Heat diffusion iteration {}/5...", iteration);

        // For each cell with energy, radiate 75% to neighbors
        for (cell_index, &energy) in current_energies.iter().enumerate() {
            if energy > 0.001 { // Only radiate if significant energy
                let radiated_energy = energy * 0.75;
                let neighbors = self.find_cell_neighbors(cell_index);

                if !neighbors.is_empty() {
                    let energy_per_neighbor = radiated_energy / neighbors.len() as f64;

                    // Add energy to each neighbor
                    for neighbor_index in neighbors {
                        if neighbor_index < new_energies.len() {
                            new_energies[neighbor_index] += energy_per_neighbor;
                        }
                    }

                    // Reduce energy in source cell
                    new_energies[cell_index] = energy * 0.25; // Keep 25%
                }
            }
        }

        new_energies
    }

    /// Find neighboring cells (simplified - using geographic proximity)
    fn find_cell_neighbors(&self, cell_index: usize) -> Vec<usize> {
        if cell_index >= self.cells.len() {
            return Vec::new();
        }

        let source_cell = &self.cells[cell_index];
        let mut neighbors = Vec::new();

        // Find cells within reasonable distance (simplified neighbor detection)
        let neighbor_threshold = 2.0; // degrees - adjust as needed

        for (i, cell) in self.cells.iter().enumerate() {
            if i != cell_index {
                let distance = self.calculate_latlng_distance(
                    source_cell.center.latitude, source_cell.center.longitude,
                    cell.center.latitude, cell.center.longitude
                );

                if distance < neighbor_threshold {
                    neighbors.push(i);
                }
            }
        }

        // Limit to reasonable number of neighbors
        neighbors.truncate(12); // Hexagonal cells typically have 6 direct neighbors
        neighbors
    }

    /// Calculate distance between two lat/lng points in degrees
    fn calculate_latlng_distance(&self, lat1: f64, lng1: f64, lat2: f64, lng2: f64) -> f64 {
        let dlat = lat2 - lat1;
        let dlng = lng2 - lng1;
        (dlat * dlat + dlng * dlng).sqrt()
    }

    /// Generate and display histogram of cellular energy values
    fn generate_cellular_energy_histogram(&self, energies: &[f64], hotspots: &[HotspotCell]) {
        println!("\n=== Cellular Heat Diffusion Histogram ===");

        // Find energy range
        let min_energy = energies.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_energy = energies.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let mean_energy = energies.iter().sum::<f64>() / energies.len() as f64;

        println!("Hotspot count: {}", hotspots.len());
        println!("Energy range: {:.4} to {:.4}", min_energy, max_energy);
        println!("Mean energy: {:.4}", mean_energy);

        // Create bins with 0.01 energy increments
        let bin_size = 0.01;
        let max_bins = ((max_energy / bin_size).ceil() as usize + 1).min(100);
        let mut bins = vec![0; max_bins];

        // Count energies in each bin
        for &energy in energies {
            let bin_index = ((energy / bin_size).floor() as usize).min(max_bins - 1);
            bins[bin_index] += 1;
        }

        // Find max count for scaling
        let max_count = *bins.iter().max().unwrap_or(&1);
        let bar_scale = 50.0 / max_count as f64;

        println!("\nEnergy Range │ Count  │ Percent │ Histogram");
        println!("─────────────┼────────┼─────────┼─────────────────────────────────────────────────");

        for (i, &count) in bins.iter().enumerate() {
            if count == 0 && i > (mean_energy / bin_size) as usize + 20 {
                continue;
            }

            let range_start = i as f64 * bin_size;
            let range_end = (i + 1) as f64 * bin_size;
            let percentage = 100.0 * count as f64 / energies.len() as f64;
            let bar_length = (count as f64 * bar_scale) as usize;
            let bar = "█".repeat(bar_length);

            println!("{:4.2}-{:4.2}   │ {:6} │ {:5.1}%  │ {}",
                     range_start, range_end, count, percentage, bar);
        }

        // Calculate percentiles
        let mut sorted_energies = energies.to_vec();
        sorted_energies.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let median = sorted_energies[sorted_energies.len() / 2];
        let p75 = sorted_energies[sorted_energies.len() * 3 / 4];
        let p90 = sorted_energies[sorted_energies.len() * 9 / 10];
        let p95 = sorted_energies[sorted_energies.len() * 95 / 100];

        println!("─────────────┼────────┼─────────┼─────────────────────────────────────────────────");
        println!("Statistics:");
        println!("  Min:     {:.4}", min_energy);
        println!("  Median:  {:.4}", median);
        println!("  Mean:    {:.4}", mean_energy);
        println!("  75th:    {:.4}", p75);
        println!("  90th:    {:.4}", p90);
        println!("  95th:    {:.4}", p95);
        println!("  Max:     {:.4}", max_energy);

        // Zero energy analysis
        let zero_energy = energies.iter().filter(|&&e| e < 0.001).count();
        println!("  Zero energy cells: {} ({:.1}%)", zero_energy, 100.0 * zero_energy as f64 / energies.len() as f64);

        // Heat diffusion analysis
        let low_energy = energies.iter().filter(|&&e| e >= 0.001 && e < 0.01).count();
        let medium_energy = energies.iter().filter(|&&e| e >= 0.01 && e < 0.1).count();
        let high_energy = energies.iter().filter(|&&e| e >= 0.1).count();

        println!("\nHeat Distribution:");
        println!("  Cold (0.000-0.001): {} cells ({:.1}%)", zero_energy, 100.0 * zero_energy as f64 / energies.len() as f64);
        println!("  Warm (0.001-0.010): {} cells ({:.1}%)", low_energy, 100.0 * low_energy as f64 / energies.len() as f64);
        println!("  Hot (0.010-0.100):  {} cells ({:.1}%)", medium_energy, 100.0 * medium_energy as f64 / energies.len() as f64);
        println!("  Very Hot (>0.100):  {} cells ({:.1}%)", high_energy, 100.0 * high_energy as f64 / energies.len() as f64);
    }
}

/// Hotspot cell with energy level
#[derive(Debug, Clone)]
struct HotspotCell {
    cell_index: usize,
    energy: f64,
}

/// Individual hotspot energy propagation calculation
impl H3GraphicsGenerator {


    /// Generate hotspots with 3D positions
    fn generate_3d_hotspots(&self) -> Result<Vec<Hotspot3D>, Box<dyn std::error::Error>> {
        // Load L2 cells and select half randomly
        let l2_cells = h3o::CellIndex::base_cells()
            .flat_map(|base_cell| base_cell.children(Resolution::Two))
            .collect::<Vec<_>>();

        let mut rng = rand::rng();
        let mut selected_cells = l2_cells;
        selected_cells.shuffle(&mut rng);
        selected_cells.truncate(selected_cells.len() / 2);

        let l2_radius = self.estimate_l2_radius();
        let mut hotspots = Vec::new();

        for cell in selected_cells {
            let center_ll = LatLng::from(cell);
            let center_3d = Point3D {
                x: center_ll.lat_radians().cos() * center_ll.lng_radians().cos(),
                y: center_ll.lat_radians().cos() * center_ll.lng_radians().sin(),
                z: center_ll.lat_radians().sin(),
            };

            // Randomize position by ±L2 radius
            let randomized_position = center_3d.randomize_on_sphere(l2_radius, &mut rng);

            // Generate random energy between 0.04 and 0.4
            let energy = rng.random_range(0.04..0.4);

            hotspots.push(Hotspot3D {
                position: randomized_position,
                energy,
            });
        }

        Ok(hotspots)
    }

    /// Build neighbor cache for all L3 cells
    fn build_neighbor_cache(&self) -> Vec<Vec<usize>> {
        let mut neighbor_cache = vec![Vec::new(); self.cells.len()];

        println!("Building neighbor cache...");
        let neighbor_threshold = 2.0; // degrees - adjust as needed

        for (i, cell_i) in self.cells.iter().enumerate() {
            for (j, cell_j) in self.cells.iter().enumerate() {
                if i != j {
                    let distance = self.calculate_latlng_distance(
                        cell_i.center.latitude, cell_i.center.longitude,
                        cell_j.center.latitude, cell_j.center.longitude
                    );

                    if distance < neighbor_threshold {
                        neighbor_cache[i].push(j);
                    }
                }
            }

            // Limit neighbors to reasonable number
            neighbor_cache[i].truncate(12);
        }

        neighbor_cache
    }

    /// Estimate L3 cell radius for distance calculations
    fn estimate_l3_radius(&self) -> f64 {
        // L3 has ~41,162 cells globally, so average area per cell ≈ 4π/41162
        // Radius ≈ sqrt(area/π) ≈ sqrt(4/41162) ≈ 0.0098
        0.0098
    }

    /// Propagate energy from a single hotspot through neighbor network
    fn propagate_hotspot_energy(
        &self,
        hotspot: &Hotspot3D,
        cell_energies: &mut [f64],
        neighbor_cache: &[Vec<usize>],
        l3_radius: f64
    ) {
        let mut visited_cells = std::collections::HashSet::new();
        let mut current_wave = Vec::new();

        // Step 1: Find the L3 cell containing this hotspot
        let hotspot_cell_index = self.find_closest_cell_to_hotspot(hotspot);

        // Step 2: Add 100% energy to hotspot cell (with distance-based scaling)
        let distance_to_center = self.calculate_hotspot_to_cell_distance(hotspot, hotspot_cell_index);
        let energy_contribution = self.calculate_exponential_energy_contribution(
            hotspot.energy, distance_to_center, l3_radius
        );

        cell_energies[hotspot_cell_index] += energy_contribution;
        visited_cells.insert(hotspot_cell_index);
        current_wave.push((hotspot_cell_index, energy_contribution));

        // Step 3: Propagate through neighbors for 5 iterations or until energy < 0.001
        for iteration in 1..=5 {
            let mut next_wave = Vec::new();

            for &(cell_index, cell_energy) in &current_wave {
                let propagated_energy = cell_energy * 0.75; // 75% propagation

                if propagated_energy < 0.001 {
                    continue; // Stop propagating if energy too low
                }

                // Add energy to all unvisited neighbors
                for &neighbor_index in &neighbor_cache[cell_index] {
                    if !visited_cells.contains(&neighbor_index) {
                        // Calculate distance-based energy contribution
                        let distance_to_neighbor = self.calculate_hotspot_to_cell_distance(hotspot, neighbor_index);
                        let neighbor_energy = self.calculate_exponential_energy_contribution(
                            propagated_energy, distance_to_neighbor, l3_radius
                        );

                        cell_energies[neighbor_index] += neighbor_energy;
                        visited_cells.insert(neighbor_index);
                        next_wave.push((neighbor_index, neighbor_energy));
                    }
                }
            }

            if next_wave.is_empty() {
                break; // No more cells to propagate to
            }

            current_wave = next_wave;
        }
    }

    /// Find the closest L3 cell to a hotspot
    fn find_closest_cell_to_hotspot(&self, hotspot: &Hotspot3D) -> usize {
        let mut closest_index = 0;
        let mut min_distance = f64::INFINITY;

        for (i, cell) in self.cells.iter().enumerate() {
            let cell_3d = Point3D {
                x: cell.center.latitude.to_radians().cos() * cell.center.longitude.to_radians().cos(),
                y: cell.center.latitude.to_radians().cos() * cell.center.longitude.to_radians().sin(),
                z: cell.center.latitude.to_radians().sin(),
            };

            let distance = hotspot.position.distance_to(&cell_3d);
            if distance < min_distance {
                min_distance = distance;
                closest_index = i;
            }
        }

        closest_index
    }

    /// Calculate distance from hotspot to cell center
    fn calculate_hotspot_to_cell_distance(&self, hotspot: &Hotspot3D, cell_index: usize) -> f64 {
        let cell = &self.cells[cell_index];
        let cell_3d = Point3D {
            x: cell.center.latitude.to_radians().cos() * cell.center.longitude.to_radians().cos(),
            y: cell.center.latitude.to_radians().cos() * cell.center.longitude.to_radians().sin(),
            z: cell.center.latitude.to_radians().sin(),
        };

        hotspot.position.distance_to(&cell_3d)
    }

    /// Calculate exponential energy contribution based on distance
    fn calculate_exponential_energy_contribution(&self, base_energy: f64, distance: f64, l3_radius: f64) -> f64 {
        let scale_radius = l3_radius * 2.0; // 2x L3 radius scaling
        let normalized_distance = distance / scale_radius;

        // Exponential falloff: energy * exp(-distance/scale)
        base_energy * (-normalized_distance).exp()
    }

    /// Generate histogram for individual hotspot propagation
    fn generate_individual_hotspot_histogram(&self, energies: &[f64], hotspots: &[Hotspot3D]) {
        println!("\n=== Individual Hotspot Propagation Histogram ===");

        // Find energy range
        let min_energy = energies.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_energy = energies.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let mean_energy = energies.iter().sum::<f64>() / energies.len() as f64;

        println!("Hotspot count: {}", hotspots.len());
        println!("Energy range: {:.4} to {:.4}", min_energy, max_energy);
        println!("Mean energy: {:.4}", mean_energy);

        // Create bins with 0.01 energy increments
        let bin_size = 0.01;
        let max_bins = ((max_energy / bin_size).ceil() as usize + 1).min(100);
        let mut bins = vec![0; max_bins];

        // Count energies in each bin
        for &energy in energies {
            let bin_index = ((energy / bin_size).floor() as usize).min(max_bins - 1);
            bins[bin_index] += 1;
        }

        // Find max count for scaling
        let max_count = *bins.iter().max().unwrap_or(&1);
        let bar_scale = 50.0 / max_count as f64;

        println!("\nEnergy Range │ Count  │ Percent │ Histogram");
        println!("─────────────┼────────┼─────────┼─────────────────────────────────────────────────");

        for (i, &count) in bins.iter().enumerate() {
            if count == 0 && i > (mean_energy / bin_size) as usize + 20 {
                continue;
            }

            let range_start = i as f64 * bin_size;
            let range_end = (i + 1) as f64 * bin_size;
            let percentage = 100.0 * count as f64 / energies.len() as f64;
            let bar_length = (count as f64 * bar_scale) as usize;
            let bar = "█".repeat(bar_length);

            println!("{:4.2}-{:4.2}   │ {:6} │ {:5.1}%  │ {}",
                     range_start, range_end, count, percentage, bar);
        }

        // Calculate percentiles and statistics
        let mut sorted_energies = energies.to_vec();
        sorted_energies.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let median = sorted_energies[sorted_energies.len() / 2];
        let p75 = sorted_energies[sorted_energies.len() * 3 / 4];
        let p90 = sorted_energies[sorted_energies.len() * 9 / 10];
        let p95 = sorted_energies[sorted_energies.len() * 95 / 100];

        println!("─────────────┼────────┼─────────┼─────────────────────────────────────────────────");
        println!("Statistics:");
        println!("  Min:     {:.4}", min_energy);
        println!("  Median:  {:.4}", median);
        println!("  Mean:    {:.4}", mean_energy);
        println!("  75th:    {:.4}", p75);
        println!("  90th:    {:.4}", p90);
        println!("  95th:    {:.4}", p95);
        println!("  Max:     {:.4}", max_energy);

        // Energy distribution analysis
        let zero_energy = energies.iter().filter(|&&e| e < 0.001).count();
        let low_energy = energies.iter().filter(|&&e| e >= 0.001 && e < 0.01).count();
        let medium_energy = energies.iter().filter(|&&e| e >= 0.01 && e < 0.1).count();
        let high_energy = energies.iter().filter(|&&e| e >= 0.1).count();

        println!("\nEnergy Distribution:");
        println!("  Cold (0.000-0.001): {} cells ({:.1}%)", zero_energy, 100.0 * zero_energy as f64 / energies.len() as f64);
        println!("  Warm (0.001-0.010): {} cells ({:.1}%)", low_energy, 100.0 * low_energy as f64 / energies.len() as f64);
        println!("  Hot (0.010-0.100):  {} cells ({:.1}%)", medium_energy, 100.0 * medium_energy as f64 / energies.len() as f64);
        println!("  Very Hot (>0.100):  {} cells ({:.1}%)", high_energy, 100.0 * high_energy as f64 / energies.len() as f64);
    }
}

/// 3D hotspot with position and energy
#[derive(Debug, Clone)]
struct Hotspot3D {
    position: Point3D,
    energy: f64,
}

/// Individual hotspot energy propagation with RocksDB caching
impl H3GraphicsGenerator {
    /// Calculate individual hotspot energy propagation with RocksDB neighbor cache
    pub fn calculate_individual_hotspot_propagation_histogram(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("=== Individual Hotspot Energy Propagation Analysis ===");
        println!("Using H3 grid neighbors for fast neighbor lookups");

        // Prepare cell data for neighbor lookups
        let cell_data: Vec<(CellIndex, f64, f64)> = self.cells.iter()
            .map(|cell| {
                (cell.cell_index, cell.center.latitude, cell.center.longitude)
            })
            .collect();

        // Step 4: Generate hotspots with 3D positions
        let hotspots = self.generate_3d_hotspots()?;
        println!("Generated {} hotspots with 3D positions", hotspots.len());

        // Step 5: Calculate L3 cell radius for distance scaling
        let l3_radius = self.estimate_l3_radius();
        println!("Estimated L3 radius: {:.6}", l3_radius);

        // Step 6: Initialize energy array
        let mut cell_energies = vec![0.0; self.cells.len()];

        // Step 7: Process each hotspot individually using H3 grid neighbors
        for (i, hotspot) in hotspots.iter().enumerate() {
            if i % 500 == 0 {
                println!("Processing hotspot {}/{}", i + 1, hotspots.len());
            }
            self.propagate_hotspot_energy_h3(hotspot, &mut cell_energies, &cell_data, l3_radius)?;
        }

        // Step 8: Generate histogram
        self.generate_individual_hotspot_histogram(&cell_energies, &hotspots);

        Ok(())
    }

    /// Propagate energy from a single hotspot using H3 grid neighbors
    fn propagate_hotspot_energy_h3(
        &self,
        hotspot: &Hotspot3D,
        cell_energies: &mut [f64],
        cell_data: &[(CellIndex, f64, f64)],
        l3_radius: f64
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut visited_cells = std::collections::HashSet::new();
        let mut current_wave = Vec::new();

        // Step 1: Find the L3 cell containing this hotspot
        let hotspot_cell_index = self.find_closest_cell_to_hotspot(hotspot);

        // Step 2: Add 100% energy to hotspot cell (with distance-based scaling)
        let distance_to_center = self.calculate_hotspot_to_cell_distance(hotspot, hotspot_cell_index);
        let energy_contribution = self.calculate_exponential_energy_contribution(
            hotspot.energy, distance_to_center, l3_radius
        );

        cell_energies[hotspot_cell_index] += energy_contribution;
        visited_cells.insert(hotspot_cell_index);
        current_wave.push((hotspot_cell_index, energy_contribution));

        // Step 3: Propagate through cached neighbors for 5 iterations or until energy < 0.001
        for _iteration in 1..=5 {
            let mut next_wave = Vec::new();

            for &(cell_index, cell_energy) in &current_wave {
                let propagated_energy = cell_energy * 0.75; // 75% propagation

                if propagated_energy < 0.001 {
                    continue; // Stop propagating if energy too low
                }

                // Get H3 grid neighbors
                let cell_index_h3 = cell_data[cell_index].0;
                let neighbor_cell_indices = H3Utils::neighbors_for(cell_index_h3);

                // Add energy to all unvisited neighbors
                for neighbor_cell_index in neighbor_cell_indices {
                    // Find the usize index for this CellIndex
                    if let Some(neighbor_index) = cell_data.iter().position(|(cell_idx, _, _)| *cell_idx == neighbor_cell_index) {
                        if neighbor_index < cell_energies.len() && !visited_cells.contains(&neighbor_index) {
                            // Calculate distance-based energy contribution
                            let distance_to_neighbor = self.calculate_hotspot_to_cell_distance(hotspot, neighbor_index);
                            let neighbor_energy = self.calculate_exponential_energy_contribution(
                                propagated_energy, distance_to_neighbor, l3_radius
                            );

                            cell_energies[neighbor_index] += neighbor_energy;
                            visited_cells.insert(neighbor_index);
                            next_wave.push((neighbor_index, neighbor_energy));
                        }
                    }
                }
            }

            if next_wave.is_empty() {
                break; // No more cells to propagate to
            }

            current_wave = next_wave;
        }

        Ok(())
    }

    /// Create L2 heat map visualization
    pub fn create_l2_heat_map_visualization(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        println!("=== Creating L2 Heat Map Visualization ===");
        println!("Generating thermal heat map for L2 resolution cells");

        // Prepare cell data for thermal calculations
        let cell_data: Vec<(CellIndex, f64, f64)> = self.cells.iter()
            .map(|cell| {
                (cell.cell_index, cell.center.latitude, cell.center.longitude)
            })
            .collect();

        println!("Loaded {} L2 cells for heat map", cell_data.len());

        // Generate simple thermal values for demonstration
        let thermal_values = self.generate_demo_thermal_values(&cell_data)?;

        // Create image buffer
        let mut image: RgbImage = ImageBuffer::new(self.config.width, self.config.height);

        // Fill with black background
        for pixel in image.pixels_mut() {
            *pixel = Rgb([0, 0, 0]);
        }

        println!("Drawing L2 heat map...");

        // Draw each cell with thermal color
        for (i, cell) in self.cells.iter().enumerate() {
            let thermal_value = thermal_values[i];
            let color = self.thermal_value_to_color(thermal_value);
            self.draw_cell_with_color(&mut image, cell, color);
        }

        // Save the image
        image.save(filename)?;
        println!("Generated L2 heat map PNG: {} ({}x{} pixels)",
                 filename, self.config.width, self.config.height);

        // Display thermal statistics
        self.display_thermal_statistics(&thermal_values);

        Ok(())
    }

    /// Generate demo thermal values for L2 cells
    fn generate_demo_thermal_values(&self, cell_data: &[(CellIndex, f64, f64)]) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        let mut rng = rand::rng();
        let mut thermal_values = Vec::new();

        // Generate random thermal values with some geographic patterns
        for (_, lat, lng) in cell_data {
            // Base temperature varies with latitude (warmer at equator)
            let lat_factor = (lat.to_radians().cos()).abs(); // 0 at poles, 1 at equator
            let base_temp = 0.3 + (lat_factor * 0.4); // 0.3 to 0.7 base range

            // Add some longitude variation (simulate continental effects)
            let lng_factor = (lng.to_radians() * 2.0).sin() * 0.1; // ±0.1 variation

            // Add random hotspots and cold spots
            let random_factor = rng.random_range(-0.2..0.3);

            let thermal_value = (base_temp + lng_factor + random_factor).clamp(0.0, 1.0);
            thermal_values.push(thermal_value);
        }

        Ok(thermal_values)
    }

    /// Convert thermal value (0.0-1.0) to heat map color
    fn thermal_value_to_color(&self, value: f64) -> Rgb<u8> {
        // Clamp value to 0.0-1.0 range
        let clamped = value.clamp(0.0, 1.0);

        // Create heat map: blue (cold) -> green -> yellow -> red (hot)
        if clamped < 0.25 {
            // Blue to cyan (cold)
            let t = clamped * 4.0;
            Rgb([0, (t * 255.0) as u8, 255])
        } else if clamped < 0.5 {
            // Cyan to green
            let t = (clamped - 0.25) * 4.0;
            Rgb([0, 255, ((1.0 - t) * 255.0) as u8])
        } else if clamped < 0.75 {
            // Green to yellow
            let t = (clamped - 0.5) * 4.0;
            Rgb([(t * 255.0) as u8, 255, 0])
        } else {
            // Yellow to red (hot)
            let t = (clamped - 0.75) * 4.0;
            Rgb([255, ((1.0 - t) * 255.0) as u8, 0])
        }
    }

    /// Display thermal statistics
    fn display_thermal_statistics(&self, thermal_values: &[f64]) {
        let min_temp = thermal_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_temp = thermal_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let mean_temp = thermal_values.iter().sum::<f64>() / thermal_values.len() as f64;

        // Count temperature ranges
        let cold_count = thermal_values.iter().filter(|&&t| t < 0.3).count();
        let cool_count = thermal_values.iter().filter(|&&t| t >= 0.3 && t < 0.5).count();
        let warm_count = thermal_values.iter().filter(|&&t| t >= 0.5 && t < 0.7).count();
        let hot_count = thermal_values.iter().filter(|&&t| t >= 0.7).count();

        println!("\n=== L2 Heat Map Statistics ===");
        println!("Thermal range: {:.3} to {:.3}", min_temp, max_temp);
        println!("Mean thermal: {:.3}", mean_temp);
        println!("Temperature distribution:");
        println!("  Cold (0.0-0.3):  {} cells ({:.1}%)", cold_count, 100.0 * cold_count as f64 / thermal_values.len() as f64);
        println!("  Cool (0.3-0.5):  {} cells ({:.1}%)", cool_count, 100.0 * cool_count as f64 / thermal_values.len() as f64);
        println!("  Warm (0.5-0.7):  {} cells ({:.1}%)", warm_count, 100.0 * warm_count as f64 / thermal_values.len() as f64);
        println!("  Hot (0.7-1.0):   {} cells ({:.1}%)", hot_count, 100.0 * hot_count as f64 / thermal_values.len() as f64);

        println!("\nColor Legend:");
        println!("  Blue:   Cold regions (0.0-0.25)");
        println!("  Cyan:   Cool regions (0.25-0.5)");
        println!("  Green:  Moderate regions (0.5-0.75)");
        println!("  Yellow: Warm regions (0.75-1.0)");
        println!("  Red:    Hot regions (approaching 1.0)");
    }

    /// Create L2 heat map with Perlin noise overlay
    pub fn create_l2_perlin_heat_map_visualization(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        println!("=== Creating L2 Heat Map with Perlin Noise Overlay ===");
        println!("Generating thermal heat map with 3D Perlin noise (3 L2 hex wavelength)");

        // Prepare cell data for thermal calculations
        let cell_data: Vec<(CellIndex, f64, f64)> = self.cells.iter()
            .map(|cell| {
                (cell.cell_index, cell.center.latitude, cell.center.longitude)
            })
            .collect();

        println!("Loaded {} L2 cells for Perlin heat map", cell_data.len());

        // Generate thermal values with Perlin noise overlay
        let thermal_values = self.generate_perlin_thermal_values(&cell_data)?;

        // Create image buffer
        let mut image: RgbImage = ImageBuffer::new(self.config.width, self.config.height);

        // Fill with black background
        for pixel in image.pixels_mut() {
            *pixel = Rgb([0, 0, 0]);
        }

        println!("Drawing L2 Perlin heat map...");

        // Draw each cell with thermal color
        for (i, cell) in self.cells.iter().enumerate() {
            let thermal_value = thermal_values[i];
            let color = self.thermal_value_to_color(thermal_value);
            self.draw_cell_with_color(&mut image, cell, color);
        }

        // Save the image
        image.save(filename)?;
        println!("Generated L2 Perlin heat map PNG: {} ({}x{} pixels)",
                 filename, self.config.width, self.config.height);

        // Display thermal statistics
        self.display_perlin_thermal_statistics(&thermal_values);

        Ok(())
    }

    /// Generate thermal values with Perlin noise overlay
    fn generate_perlin_thermal_values(&self, cell_data: &[(CellIndex, f64, f64)]) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        let mut rng = rand::rng();
        let mut thermal_values = Vec::new();

        // Initialize Perlin noise generator
        let perlin = Perlin::new(rng.random());

        // Calculate L2 cell radius for wavelength scaling
        let l2_radius_km = 166.0; // From previous calculations
        let wavelength_km = l2_radius_km * 3.0 * 2.0; // 3 L2 hex wavelength (diameter)
        let earth_radius_km = 6371.0;

        // Scale factor for Perlin noise (smaller = larger features)
        let perlin_scale = earth_radius_km / wavelength_km; // ~6371/996 ≈ 6.4

        println!("Perlin noise configuration:");
        println!("  L2 radius: {:.0} km", l2_radius_km);
        println!("  Wavelength: {:.0} km (3 L2 hex)", wavelength_km);
        println!("  Perlin scale: {:.2}", perlin_scale);

        // Generate thermal values with Perlin noise overlay
        for (_, lat, lng) in cell_data {
            // Base temperature varies with latitude (warmer at equator)
            let lat_factor = (lat.to_radians().cos()).abs(); // 0 at poles, 1 at equator
            let base_temp = 0.3 + (lat_factor * 0.4); // 0.3 to 0.7 base range

            // Convert lat/lng to 3D coordinates for Perlin noise
            let lat_rad = lat.to_radians();
            let lng_rad = lng.to_radians();
            let x = lat_rad.cos() * lng_rad.cos();
            let y = lat_rad.cos() * lng_rad.sin();
            let z = lat_rad.sin();

            // Sample 3D Perlin noise at scaled coordinates
            let perlin_value = perlin.get([x * perlin_scale, y * perlin_scale, z * perlin_scale]);

            // Normalize Perlin noise from [-1,1] to [0,1] and scale
            let perlin_normalized = (perlin_value + 1.0) / 2.0;
            let perlin_contribution = (perlin_normalized - 0.5) * 0.6; // ±0.3 variation

            // Add some longitude variation (simulate continental effects)
            let lng_factor = (lng_rad * 2.0).sin() * 0.05; // ±0.05 variation

            // Combine all factors
            let thermal_value = (base_temp + perlin_contribution + lng_factor).clamp(0.0, 1.0);
            thermal_values.push(thermal_value);
        }

        Ok(thermal_values)
    }

    /// Display Perlin thermal statistics
    fn display_perlin_thermal_statistics(&self, thermal_values: &[f64]) {
        let min_temp = thermal_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_temp = thermal_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let mean_temp = thermal_values.iter().sum::<f64>() / thermal_values.len() as f64;

        // Calculate standard deviation
        let variance = thermal_values.iter()
            .map(|&x| (x - mean_temp).powi(2))
            .sum::<f64>() / thermal_values.len() as f64;
        let std_dev = variance.sqrt();

        // Count temperature ranges
        let cold_count = thermal_values.iter().filter(|&&t| t < 0.3).count();
        let cool_count = thermal_values.iter().filter(|&&t| t >= 0.3 && t < 0.5).count();
        let warm_count = thermal_values.iter().filter(|&&t| t >= 0.5 && t < 0.7).count();
        let hot_count = thermal_values.iter().filter(|&&t| t >= 0.7).count();

        println!("\n=== L2 Perlin Heat Map Statistics ===");
        println!("Thermal range: {:.3} to {:.3}", min_temp, max_temp);
        println!("Mean thermal: {:.3}", mean_temp);
        println!("Standard deviation: {:.3}", std_dev);
        println!("Temperature distribution:");
        println!("  Cold (0.0-0.3):  {} cells ({:.1}%)", cold_count, 100.0 * cold_count as f64 / thermal_values.len() as f64);
        println!("  Cool (0.3-0.5):  {} cells ({:.1}%)", cool_count, 100.0 * cool_count as f64 / thermal_values.len() as f64);
        println!("  Warm (0.5-0.7):  {} cells ({:.1}%)", warm_count, 100.0 * warm_count as f64 / thermal_values.len() as f64);
        println!("  Hot (0.7-1.0):   {} cells ({:.1}%)", hot_count, 100.0 * hot_count as f64 / thermal_values.len() as f64);

        println!("\nPerlin Noise Effects:");
        println!("  Wavelength: ~996 km (3 L2 hex diameter)");
        println!("  Variation: ±0.3 thermal units");
        println!("  Pattern: Large-scale thermal provinces");
        println!("  3D mapping: Seamless across sphere");

        println!("\nColor Legend:");
        println!("  Blue:   Cold regions (0.0-0.25)");
        println!("  Cyan:   Cool regions (0.25-0.5)");
        println!("  Green:  Moderate regions (0.5-0.75)");
        println!("  Yellow: Warm regions (0.75-1.0)");
        println!("  Red:    Hot regions (approaching 1.0)");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_h3_graphics_config() {
        let config = H3GraphicsConfig::new(Resolution::Two, 3);
        assert_eq!(config.points_per_degree, 3);
        assert_eq!(config.width, 1080);  // 3 * 360
        assert_eq!(config.height, 540);  // 3 * 180
    }

    #[test]
    fn test_coordinate_conversion() {
        let config = H3GraphicsConfig::new(Resolution::One, 2);
        let generator = H3GraphicsGenerator::new(config);
        
        // Test longitude conversion
        assert_eq!(generator.lon_to_x(-180.0), 0);
        assert_eq!(generator.lon_to_x(0.0), 360);   // 2 * 180
        assert_eq!(generator.lon_to_x(180.0), 720); // 2 * 360
        
        // Test latitude conversion  
        assert_eq!(generator.lat_to_y(90.0), 0);    // North pole at top
        assert_eq!(generator.lat_to_y(0.0), 180);   // Equator at middle
        assert_eq!(generator.lat_to_y(-90.0), 360); // South pole at bottom
    }

    #[test]
    fn test_cell_loading() {
        let config = H3GraphicsConfig::new(Resolution::Zero, 1);
        let mut generator = H3GraphicsGenerator::new(config);
        
        generator.load_cells();
        assert_eq!(generator.cell_count(), 122); // H3 resolution 0 has 122 cells
    }
}
