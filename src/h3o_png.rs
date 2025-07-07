/// H3 Cell Network PNG Graphics Generator
/// 
/// Generates PNG graphics of H3 hexagonal cell networks with configurable resolution
/// and points per degree for longitude/latitude mapping.

use h3o::{CellIndex, LatLng, Resolution};
use image::{ImageBuffer, Rgb, RgbImage};
use rand::Rng;

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
    pub config: H3GraphicsConfig,
    pub cells: Vec<H3CellGraphics>,
}

impl H3GraphicsGenerator {
    /// Create a new H3 graphics generator
    pub fn new(config: H3GraphicsConfig) -> Self {
        Self {
            config,
            cells: Vec::new(),
        }
    }

    /// Get reference to loaded cells for external color mapping
    pub fn get_cells(&self) -> &Vec<H3CellGraphics> {
        &self.cells
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

        println!("Drawing {} hexagons with colored fills (filtering out broad cells)...", self.cells.len());

        // Draw only coherent cells (X extent < 20% of screen width)
        let width_threshold = (self.config.width as i32) * 20 / 100;
        let mut filled_count = 0;

        for cell in &self.cells {
            // Convert cell corners to pixel coordinates
            let pixel_coords: Vec<(i32, i32)> = cell.corners.iter()
                .map(|corner| self.geo_to_pixel(corner.longitude, corner.latitude))
                .collect();

            if pixel_coords.is_empty() {
                continue;
            }

            // Check X extent to filter out broad wraparound cells
            let min_x = pixel_coords.iter().map(|(x, _)| *x).min().unwrap();
            let max_x = pixel_coords.iter().map(|(x, _)| *x).max().unwrap();
            let x_extent = max_x - min_x;

            // Only draw cell if it's coherent (X extent < 20% of width)
            if x_extent < width_threshold {
                // Filter coordinates to image bounds
                let coords: Vec<(i32, i32)> = pixel_coords.iter()
                    .filter(|(x, y)| *x >= 0 && *x < self.config.width as i32 &&
                                     *y >= 0 && *y < self.config.height as i32)
                    .cloned()
                    .collect();

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

    /// Generate a PNG with random colored hexagon fills and lat/lon grid overlay
    pub fn generate_colored_hexagons_with_grid_png(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        // Create image buffer
        let mut image: RgbImage = ImageBuffer::new(self.config.width, self.config.height);

        // Fill with black background
        for pixel in image.pixels_mut() {
            *pixel = Rgb([0, 0, 0]);
        }

        // Create random number generator
        let mut rng = rand::rng();

        println!("Drawing {} hexagons with colored fills and lat/lon grid...", self.cells.len());

        // Draw only coherent cells (X extent < 20% of screen width)
        let width_threshold = (self.config.width as i32) * 20 / 100;
        let mut filled_count = 0;

        for cell in &self.cells {
            // Convert cell corners to pixel coordinates
            let pixel_coords: Vec<(i32, i32)> = cell.corners.iter()
                .map(|corner| self.geo_to_pixel(corner.longitude, corner.latitude))
                .collect();

            if pixel_coords.is_empty() {
                continue;
            }

            // Check X extent to filter out broad wraparound cells
            let min_x = pixel_coords.iter().map(|(x, _)| *x).min().unwrap();
            let max_x = pixel_coords.iter().map(|(x, _)| *x).max().unwrap();
            let x_extent = max_x - min_x;

            // Only draw cell if it's coherent (X extent < 20% of width)
            if x_extent < width_threshold {
                // Filter coordinates to image bounds
                let coords: Vec<(i32, i32)> = pixel_coords.iter()
                    .filter(|(x, y)| *x >= 0 && *x < self.config.width as i32 &&
                                     *y >= 0 && *y < self.config.height as i32)
                    .cloned()
                    .collect();

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

        println!("Filled {} hexagons with colors", filled_count);

        // Draw lat/lon grid overlay
        self.draw_lat_lon_grid(&mut image);

        // Save the image
        image.save(filename)?;
        println!("Generated colored hexagons with grid PNG: {} ({}x{} pixels, {} cells)",
                 filename, self.config.width, self.config.height, self.cells.len());
        Ok(())
    }

    /// Generate PNG with coherent cells only + optional nearest neighbor fill
    pub fn generate_coherent_cells_with_fill_png(&self, filename: &str, enable_fill: bool) -> Result<(), Box<dyn std::error::Error>> {
        self.generate_coherent_cells_with_colors_png(filename, enable_fill, None)
    }

    /// Generate PNG with coherent cells using provided colors per cell
    pub fn generate_coherent_cells_with_colors_png(
        &self, 
        filename: &str, 
        enable_fill: bool,
        cell_colors_map: Option<std::collections::HashMap<h3o::CellIndex, Rgb<u8>>>
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Create image buffer
        let mut image: RgbImage = ImageBuffer::new(self.config.width, self.config.height);

        // Fill with black background
        for pixel in image.pixels_mut() {
            *pixel = Rgb([0, 0, 0]);
        }

        // Create random number generator and store cell colors
        let mut rng = rand::rng();
        let mut cell_colors = Vec::new();
        let mut cell_centers = Vec::new();

        println!("Step 1: Drawing only coherent cells (X extent < 20% of screen width)...");

        // Step 1: Draw only cells where X extent is < 20% of screen width
        let width_threshold = (self.config.width as i32) * 20 / 100; // 20% of screen width
        let mut coherent_count = 0;
        let mut rejected_cells = Vec::new();

        for (i, cell) in self.cells.iter().enumerate() {
            // Convert cell corners to pixel coordinates
            let pixel_coords: Vec<(i32, i32)> = cell.corners.iter()
                .map(|corner| self.geo_to_pixel(corner.longitude, corner.latitude))
                .collect();

            if pixel_coords.is_empty() {
                continue;
            }

            // Check X extent
            let min_x = pixel_coords.iter().map(|(x, _)| *x).min().unwrap();
            let max_x = pixel_coords.iter().map(|(x, _)| *x).max().unwrap();
            let x_extent = max_x - min_x;

            // Get color for this cell - use provided color or generate random
            let cell_color = if let Some(ref color_map) = cell_colors_map {
                // Use provided color if available, otherwise default to gray
                color_map.get(&cell.cell_index).copied().unwrap_or(Rgb([128, 128, 128]))
            } else {
                // Generate random color as fallback
                Rgb([
                    rng.random_range(50..255),
                    rng.random_range(50..255),
                    rng.random_range(50..255),
                ])
            };

            // Store cell center and color for later nearest neighbor filling
            let center_pixel = self.geo_to_pixel(cell.center.longitude, cell.center.latitude);
            cell_centers.push(center_pixel);
            cell_colors.push(cell_color);

            // Draw cell only if it's coherent (X extent < 20% of width)
            if x_extent < width_threshold {
                // Filter coordinates to image bounds
                let coords: Vec<(i32, i32)> = pixel_coords.iter()
                    .filter(|(x, y)| *x >= 0 && *x < self.config.width as i32 &&
                                     *y >= 0 && *y < self.config.height as i32)
                    .cloned()
                    .collect();

                if coords.len() >= 3 {
                    self.fill_polygon_simple(&mut image, &coords, cell_color);
                    coherent_count += 1;
                }
            } else {
                // Store rejected cell for coordinate variation testing
                rejected_cells.push((i, cell, cell_color));
            }
        }

        println!("Drew {} coherent cells out of {}", coherent_count, self.cells.len());

        println!("Skipping coordinate variations - original approach was correct");
        println!("Total cells drawn: {} out of {}", coherent_count, self.cells.len());

        // Step 2: Optionally fill uncovered pixels with nearest cell center color
        if enable_fill {
            println!("Step 2: Filling uncovered pixels with nearest cell center colors...");
            
            let mut filled_pixels = 0;
            for y in 0..self.config.height {
                for x in 0..self.config.width {
                    let current_pixel = image.get_pixel(x, y);
                    
                    // If pixel is black (uncovered), find nearest cell center
                    if current_pixel.0 == [0, 0, 0] {
                        let mut min_distance = f64::INFINITY;
                        let mut nearest_color = Rgb([0, 0, 0]);

                        for (i, &(cx, cy)) in cell_centers.iter().enumerate() {
                            let dx = x as f64 - cx as f64;
                            let dy = y as f64 - cy as f64;
                            let distance = (dx * dx + dy * dy).sqrt();

                            if distance < min_distance {
                                min_distance = distance;
                                nearest_color = cell_colors[i];
                            }
                        }

                        image.put_pixel(x, y, nearest_color);
                        filled_pixels += 1;
                    }
                }
            }

            println!("Filled {} uncovered pixels with nearest neighbor colors", filled_pixels);
        } else {
            println!("Skipping nearest neighbor fill for faster rendering");
        }

        // Draw lat/lon grid overlay
        self.draw_lat_lon_grid(&mut image);

        // Save the image
        image.save(filename)?;
        println!("Generated coherent cells with fill PNG: {} ({}x{} pixels, {} total cells, {} coherent)",
                 filename, self.config.width, self.config.height, self.cells.len(), coherent_count);
        Ok(())
    }

    /// Draw latitude/longitude grid lines at 30-degree increments
    fn draw_lat_lon_grid(&self, image: &mut RgbImage) {
        let grid_color = Rgb([128, 128, 128]); // Gray grid lines

        // Draw longitude lines (vertical) every 30 degrees
        for lon in (-180..=180).step_by(30) {
            let x = self.lon_to_x(lon as f64);
            
            // Draw vertical line from top to bottom
            for y in 0..self.config.height {
                if x >= 0 && x < self.config.width as i32 {
                    image.put_pixel(x as u32, y, grid_color);
                }
            }
        }

        // Draw latitude lines (horizontal) every 30 degrees
        for lat in (-90..=90).step_by(30) {
            let y = self.lat_to_y(lat as f64);
            
            // Draw horizontal line from left to right
            for x in 0..self.config.width {
                if y >= 0 && y < self.config.height as i32 {
                    image.put_pixel(x, y as u32, grid_color);
                }
            }
        }

        println!("Added lat/lon grid with 30-degree increments");
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

        // Draw only coherent cells (X extent < 20% of screen width)
        let width_threshold = (self.config.width as i32) * 20 / 100;
        let mut filled_count = 0;

        for cell in &self.cells {
            // Convert cell corners to pixel coordinates using 3D projection
            let pixel_coords: Vec<(i32, i32)> = cell.corners.iter()
                .map(|corner| {
                    let point_3d = corner.to_3d();
                    self.project_3d_to_pixel(&point_3d)
                })
                .collect();

            if pixel_coords.is_empty() {
                continue;
            }

            // Check X extent to filter out broad wraparound cells
            let min_x = pixel_coords.iter().map(|(x, _)| *x).min().unwrap();
            let max_x = pixel_coords.iter().map(|(x, _)| *x).max().unwrap();
            let x_extent = max_x - min_x;

            // Only draw cell if it's coherent (X extent < 20% of width)
            if x_extent < width_threshold {
                // Filter coordinates to image bounds
                let coords: Vec<(i32, i32)> = pixel_coords.iter()
                    .filter(|(x, y)| *x >= 0 && *x < self.config.width as i32 &&
                                     *y >= 0 && *y < self.config.height as i32)
                    .cloned()
                    .collect();

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

        // Draw only coherent cells (X extent < 20% of screen width)
        let width_threshold = (self.config.width as i32) * 20 / 100;
        let mut filled_count = 0;

        for cell in &self.cells {
            // Convert cell corners to pixel coordinates
            let pixel_coords: Vec<(i32, i32)> = cell.corners.iter()
                .map(|corner| self.geo_to_pixel(corner.longitude, corner.latitude))
                .collect();

            if pixel_coords.is_empty() {
                continue;
            }

            // Check X extent to filter out broad wraparound cells
            let min_x = pixel_coords.iter().map(|(x, _)| *x).min().unwrap();
            let max_x = pixel_coords.iter().map(|(x, _)| *x).max().unwrap();
            let x_extent = max_x - min_x;

            // Only draw cell if it's coherent (X extent < 20% of width)
            if x_extent < width_threshold {
                // Filter coordinates to image bounds
                let coords: Vec<(i32, i32)> = pixel_coords.iter()
                    .filter(|(x, y)| *x >= 0 && *x < self.config.width as i32 &&
                                     *y >= 0 && *y < self.config.height as i32)
                    .cloned()
                    .collect();

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

    /// Simple polygon fill using basic scanline algorithm
    fn fill_polygon_simple(&self, image: &mut RgbImage, coords: &[(i32, i32)], color: Rgb<u8>) {
        if coords.len() < 3 {
            return;
        }

        // Find bounding box
        let min_y = coords.iter().map(|(_, y)| *y).min().unwrap_or(0).max(0);
        let max_y = coords.iter().map(|(_, y)| *y).max().unwrap_or(0).min(self.config.height as i32 - 1);

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
                    let x1 = chunk[0].max(0).min(self.config.width as i32 - 1);
                    let x2 = chunk[1].max(0).min(self.config.width as i32 - 1);
                    
                    for x in x1..=x2 {
                        if x >= 0 && x < self.config.width as i32 && y >= 0 && y < self.config.height as i32 {
                            image.put_pixel(x as u32, y as u32, color);
                        }
                    }
                }
            }
        }
    }

    /// Get cell boundary paths with simple wraparound handling
    fn get_cell_boundary_paths_with_wraparound(&self) -> Vec<Vec<(i32, i32)>> {
        // For now, just use the basic paths - can be enhanced later
        self.get_cell_boundary_paths()
    }

    /// Get cell boundary paths using 3D projection
    fn get_cell_boundary_paths_3d(&self) -> Vec<Vec<(i32, i32)>> {
        let mut paths = Vec::new();

        for cell in &self.cells {
            let mut path = Vec::new();

            for corner in &cell.corners {
                // Convert to 3D and back for better wraparound handling
                let point_3d = corner.to_3d();
                let (x, y) = self.project_3d_to_pixel(&point_3d);
                path.push((x, y));
            }

            if !path.is_empty() {
                paths.push(path);
            }
        }

        paths
    }

    /// Generate basic PNG with cell boundaries (compatibility function)
    pub fn generate_png(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        self.generate_colored_hexagons_png(filename)
    }

    /// Generate Voronoi upwelling visualization (placeholder)
    fn generate_voronoi_upwelling_visualization(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        // Placeholder - just use colored hexagons for now
        self.generate_colored_hexagons_png(filename)
    }
    
    /// Draw a cell with specified color
    fn draw_cell_with_color(&self, image: &mut RgbImage, cell: &H3CellGraphics, color: Rgb<u8>) {
        // Convert cell corners to pixel coordinates
        let pixel_coords: Vec<(i32, i32)> = cell.corners.iter()
            .map(|corner| self.geo_to_pixel(corner.longitude, corner.latitude))
            .collect();

        if pixel_coords.len() >= 3 {
            // Filter coordinates to image bounds
            let coords: Vec<(i32, i32)> = pixel_coords.iter()
                .filter(|(x, y)| *x >= 0 && *x < self.config.width as i32 &&
                                 *y >= 0 && *y < self.config.height as i32)
                .cloned()
                .collect();

            if coords.len() >= 3 {
                self.fill_polygon_simple(image, &coords, color);
            }
        }
    }

    /// Draw thermal visualization from color map
    pub fn draw_thermal_png_from_colors(&self, cell_colors: &Vec<Rgb<u8>>, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        // Convert Vec to HashMap mapping cell indices to colors
        let mut color_map = std::collections::HashMap::new();
        for (i, cell) in self.cells.iter().enumerate() {
            if i < cell_colors.len() {
                color_map.insert(cell.cell_index, cell_colors[i]);
            }
        }
        self.generate_coherent_cells_with_colors_png(filename, false, Some(color_map))
    }

    /// Convert thermal value to color (0.0 = black, 1.0 = white)
    pub fn thermal_value_to_color(&self, value: f64) -> Rgb<u8> {
        let clamped = value.max(0.0).min(1.0);
        let intensity = (clamped * 255.0) as u8;
        Rgb([intensity, intensity, intensity])
    }
}

