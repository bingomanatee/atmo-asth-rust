use atmo_asth_rust::asth_cell::{AsthCellColumn, AsthCellParams};
use atmo_asth_rust::constants::EARTH_RADIUS_KM;
use atmo_asth_rust::energy_mass::EnergyMass;
use atmo_asth_rust::h3_utils::H3Utils;
use h3o::{CellIndex, Resolution};

/// Analyze energy content of asthenosphere cells at different layer heights
fn main() {
    println!("🌍 Asthenosphere Cell Energy Analysis");
    println!("=====================================\n");

    // Test different resolutions and layer heights
    let resolutions = vec![Resolution::Zero, Resolution::One, Resolution::Two, Resolution::Three];
    let layer_heights = vec![10.0, 20.0, 50.0, 100.0];
    let layer_counts = vec![1, 3, 5];
    
    // Standard Earth parameters
    let planet_radius_km = EARTH_RADIUS_KM as f64;
    let surface_temp_k = 1500.0; // Typical asthenosphere surface temperature
    
    println!("Earth Parameters:");
    println!("  Radius: {:.0} km", planet_radius_km);
    println!("  Surface Temperature: {:.0}K", surface_temp_k);
    println!();

    // Analyze cell areas by resolution
    println!("Cell Areas by H3 Resolution:");
    println!("----------------------------");
    for &resolution in &resolutions {
        let area_km2 = H3Utils::cell_area(resolution, planet_radius_km);
        let total_cells = H3Utils::cell_count_at_resolution(resolution);
        
        println!("Resolution {:?}:", resolution);
        println!("  Area per cell: {:.2e} km²", area_km2);
        println!("  Total cells: {}", total_cells);
        println!("  Total surface area: {:.2e} km²", area_km2 * total_cells as f64);
        println!();
    }

    // Analyze energy content for different configurations
    println!("Energy Content Analysis:");
    println!("------------------------");
    
    for &resolution in &resolutions {
        println!("Resolution {:?}:", resolution);
        
        let area_km2 = H3Utils::cell_area(resolution, planet_radius_km);
        
        // Create a sample cell index for this resolution
        let sample_cell = get_sample_cell_for_resolution(resolution);
        
        for &layer_count in &layer_counts {
            for &layer_height_km in &layer_heights {
                // Create a sample asthenosphere cell
                let cell = AsthCellColumn::new(AsthCellParams {
                    cell_index: sample_cell,
                    volume: area_km2,
                    energy: 0.0, // Will be calculated by constructor
                    layer_count,
                    layer_height_km,
                    planet_radius_km,
                    surface_temp_k,
                });
                
                // Calculate total energy and per-layer energy
                let total_energy: f64 = cell.asth_layers.iter().map(|layer| layer.energy_joules()).sum();
                let energy_per_layer = total_energy / layer_count as f64;
                let volume_per_layer = area_km2 * layer_height_km;
                let total_volume = volume_per_layer * layer_count as f64;
                
                // Calculate energy density
                let energy_density_j_per_km3 = total_energy / total_volume;
                let energy_density_j_per_m3 = energy_density_j_per_km3 / 1e9;
                
                println!("  {} layers × {:.0}km height:", layer_count, layer_height_km);
                println!("    Total volume: {:.2e} km³", total_volume);
                println!("    Total energy: {:.2e} J", total_energy);
                println!("    Energy per layer: {:.2e} J", energy_per_layer);
                println!("    Energy density: {:.2e} J/km³ ({:.2e} J/m³)", 
                    energy_density_j_per_km3, energy_density_j_per_m3);
                
                // Calculate temperature for verification
                if let Some(first_layer) = cell.asth_layers.first() {
                    println!("    Surface layer temp: {:.1}K", first_layer.kelvin());
                }
                if let Some(bottom_layer) = cell.asth_layers.last() {
                    println!("    Bottom layer temp: {:.1}K", bottom_layer.kelvin());
                }
                println!();
            }
        }
        println!();
    }

    // Special focus on typical simulation parameters
    println!("Typical Simulation Configurations:");
    println!("----------------------------------");
    
    let typical_configs = vec![
        ("Small Scale", Resolution::Two, 3, 10.0),
        ("Medium Scale", Resolution::Two, 3, 50.0),
        ("Large Scale", Resolution::One, 5, 50.0),
        ("High Resolution", Resolution::Three, 3, 20.0),
    ];
    
    for (name, resolution, layer_count, layer_height_km) in typical_configs {
        let area_km2 = H3Utils::cell_area(resolution, planet_radius_km);
        let sample_cell = get_sample_cell_for_resolution(resolution);
        
        let cell = AsthCellColumn::new(AsthCellParams {
            cell_index: sample_cell,
            volume: area_km2,
            energy: 0.0,
            layer_count,
            layer_height_km,
            planet_radius_km,
            surface_temp_k,
        });
        
        let total_energy: f64 = cell.asth_layers.iter().map(|layer| layer.energy_joules()).sum();
        let energy_per_layer = total_energy / layer_count as f64;
        
        println!("{} ({:?}, {} × {:.0}km):", name, resolution, layer_count, layer_height_km);
        println!("  Cell area: {:.2e} km²", area_km2);
        println!("  Total energy: {:.2e} J", total_energy);
        println!("  Energy per layer: {:.2e} J", energy_per_layer);
        println!("  Energy transfer at 20% throttle: {:.2e} J", energy_per_layer * 0.2);
        println!();
    }

    // Energy transfer comparison
    println!("Energy Transfer Scale Comparison:");
    println!("--------------------------------");
    println!("For context, here are some energy transfer magnitudes:");
    
    let res2_area = H3Utils::cell_area(Resolution::Two, planet_radius_km);
    let sample_cell = get_sample_cell_for_resolution(Resolution::Two);
    
    // 10km layer
    let cell_10km = AsthCellColumn::new(AsthCellParams {
        cell_index: sample_cell,
        volume: res2_area,
        energy: 0.0,
        layer_count: 3,
        layer_height_km: 10.0,
        planet_radius_km,
        surface_temp_k,
    });
    
    // 50km layer  
    let cell_50km = AsthCellColumn::new(AsthCellParams {
        cell_index: sample_cell,
        volume: res2_area,
        energy: 0.0,
        layer_count: 3,
        layer_height_km: 50.0,
        planet_radius_km,
        surface_temp_k,
    });
    
    let energy_10km = cell_10km.asth_layers[0].energy_joules();
    let energy_50km = cell_50km.asth_layers[0].energy_joules();
    
    println!("Resolution 2 cell (area: {:.2e} km²):", res2_area);
    println!("  10km layer energy: {:.2e} J", energy_10km);
    println!("  50km layer energy: {:.2e} J", energy_50km);
    println!("  Energy ratio (50km/10km): {:.1}x", energy_50km / energy_10km);
    println!();
    println!("Thermal transfer from our experiment: ~1.14e11 J");
    println!("  As % of 10km layer: {:.1}%", (1.14e11 / energy_10km) * 100.0);
    println!("  As % of 50km layer: {:.1}%", (1.14e11 / energy_50km) * 100.0);
}

/// Get a sample cell index for the given resolution
fn get_sample_cell_for_resolution(resolution: Resolution) -> CellIndex {
    // Get the first cell from the resolution
    H3Utils::iter_cells_with_base(resolution).next().unwrap().0
}
