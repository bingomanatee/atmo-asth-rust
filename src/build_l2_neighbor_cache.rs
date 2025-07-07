

use atmo_asth_rust::h3_neighbor_cache::H3NeighborCache;
use h3o::{CellIndex, LatLng, Resolution};

fn build_l2_cache(
    resolution: Resolution, 
    cache_path: &str, 
    neighbor_threshold: f64
) -> Result<usize, Box<dyn std::error::Error>> {
    println!("\nInitializing RocksDB cache...");
    
    // Initialize RocksDB neighbor cache
    let cache = H3NeighborCache::new(cache_path, neighbor_threshold)?;
    
    // Check if cache already exists
    if cache.has_cache_for_resolution(resolution)? {
        println!("L2 cache already exists, clearing and rebuilding...");
        cache.clear_cache_for_resolution(resolution)?;
    }
    
    // Load all L2 cells
    println!("Loading L2 cells...");
    let l2_cells = h3o::CellIndex::base_cells()
        .flat_map(|base_cell| base_cell.children(resolution))
        .collect::<Vec<_>>();
    
    println!("Found {} L2 cells", l2_cells.len());
    
    // Prepare cell data for cache (CellIndex, lat, lng)
    let cell_data: Vec<(CellIndex, f64, f64)> = l2_cells.iter()
        .map(|&cell_index| {
            let latlng = LatLng::from(cell_index);
            (cell_index, latlng.lat_radians().to_degrees(), latlng.lng_radians().to_degrees())
        })
        .collect();
    
    println!("Prepared {} cell data entries", cell_data.len());
    
    // Populate the cache
    println!("Building neighbor relationships...");
    cache.populate_cache_for_resolution(resolution, &cell_data)?;
    
    // Verify cache was built
    let stats = cache.get_cache_stats()?;
    if let Some(&count) = stats.get(&format!("resolution_{}", resolution as u8)) {
        println!("Cache verification: {} neighbor entries stored", count);
    }
    
    Ok(cell_data.len())
}
