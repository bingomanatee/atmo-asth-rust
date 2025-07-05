/// H3 Neighbor Cache using RocksDB
/// 
/// Persistent caching of H3 cell neighbors with column families per resolution
/// for fast neighbor lookups during thermal simulation

use h3o::{CellIndex, Resolution};
use rocksdb::{ColumnFamilyDescriptor, DB, Options};
use std::collections::HashMap;
use std::path::Path;

/// H3 neighbor cache using RocksDB with column families per resolution
pub struct H3NeighborCache {
    db: DB,
    neighbor_threshold: f64,
}

impl H3NeighborCache {
    /// Create or open H3 neighbor cache
    pub fn new<P: AsRef<Path>>(db_path: P, neighbor_threshold: f64) -> Result<Self, Box<dyn std::error::Error>> {
        // Create column families for each H3 resolution (0-15)
        let mut cf_descriptors = Vec::new();
        
        // Default column family
        cf_descriptors.push(ColumnFamilyDescriptor::new("default", Options::default()));
        
        // Column family for each resolution
        for resolution in 0..=15 {
            let cf_name = format!("resolution_{}", resolution);
            cf_descriptors.push(ColumnFamilyDescriptor::new(cf_name, Options::default()));
        }
        
        // Open database with column families
        let mut db_opts = Options::default();
        db_opts.create_if_missing(true);
        db_opts.create_missing_column_families(true);
        
        let db = DB::open_cf_descriptors(&db_opts, db_path, cf_descriptors)?;
        
        Ok(Self {
            db,
            neighbor_threshold,
        })
    }
    
    /// Get neighbors for a cell, building cache if needed
    pub fn get_neighbors(&self, cell_index: CellIndex, all_cells: &[(CellIndex, f64, f64)]) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
        let resolution = cell_index.resolution();
        let cf_name = format!("resolution_{}", resolution as u8);
        let cf = self.db.cf_handle(&cf_name)
            .ok_or_else(|| format!("Column family {} not found", cf_name))?;
        
        // Create key from cell index
        let key = self.cell_index_to_key(cell_index);
        
        // Try to get from cache first
        if let Some(cached_data) = self.db.get_cf(cf, &key)? {
            return Ok(self.deserialize_neighbors(&cached_data)?);
        }
        
        // Not in cache, compute neighbors
        let neighbors = self.compute_neighbors(cell_index, all_cells)?;
        
        // Store in cache
        let serialized = self.serialize_neighbors(&neighbors)?;
        self.db.put_cf(cf, &key, &serialized)?;
        
        Ok(neighbors)
    }
    
    /// Check if cache exists for a resolution
    pub fn has_cache_for_resolution(&self, resolution: Resolution) -> Result<bool, Box<dyn std::error::Error>> {
        let cf_name = format!("resolution_{}", resolution as u8);
        let cf = self.db.cf_handle(&cf_name)
            .ok_or_else(|| format!("Column family {} not found", cf_name))?;
        
        // Check if any keys exist in this column family
        let mut iter = self.db.iterator_cf(cf, rocksdb::IteratorMode::Start);
        Ok(iter.next().is_some())
    }
    
    /// Pre-populate cache for a resolution
    pub fn populate_cache_for_resolution(
        &self, 
        resolution: Resolution, 
        all_cells: &[(CellIndex, f64, f64)]
    ) -> Result<(), Box<dyn std::error::Error>> {
        println!("Populating neighbor cache for resolution {:?}...", resolution);
        
        let cf_name = format!("resolution_{}", resolution as u8);
        let cf = self.db.cf_handle(&cf_name)
            .ok_or_else(|| format!("Column family {} not found", cf_name))?;
        
        let resolution_cells: Vec<_> = all_cells.iter()
            .filter(|(cell_index, _, _)| cell_index.resolution() == resolution)
            .collect();
        
        println!("Processing {} cells for resolution {:?}", resolution_cells.len(), resolution);
        
        for (i, &(cell_index, _, _)) in resolution_cells.iter().enumerate() {
            if i % 1000 == 0 {
                println!("  Processed {}/{} cells", i, resolution_cells.len());
            }

            let key = self.cell_index_to_key(*cell_index);

            // Skip if already cached
            if self.db.get_cf(cf, &key)?.is_some() {
                continue;
            }

            // Compute and cache neighbors
            let neighbors = self.compute_neighbors(*cell_index, all_cells)?;
            let serialized = self.serialize_neighbors(&neighbors)?;
            self.db.put_cf(cf, &key, &serialized)?;
        }
        
        println!("Completed neighbor cache for resolution {:?}", resolution);
        Ok(())
    }
    
    /// Clear cache for a specific resolution
    pub fn clear_cache_for_resolution(&self, resolution: Resolution) -> Result<(), Box<dyn std::error::Error>> {
        let cf_name = format!("resolution_{}", resolution as u8);
        let cf = self.db.cf_handle(&cf_name)
            .ok_or_else(|| format!("Column family {} not found", cf_name))?;
        
        // Delete all keys in this column family
        let iter = self.db.iterator_cf(cf, rocksdb::IteratorMode::Start);
        for item in iter {
            let (key, _) = item?;
            self.db.delete_cf(cf, &key)?;
        }
        
        println!("Cleared neighbor cache for resolution {:?}", resolution);
        Ok(())
    }
    
    /// Get cache statistics
    pub fn get_cache_stats(&self) -> Result<HashMap<String, usize>, Box<dyn std::error::Error>> {
        let mut stats = HashMap::new();
        
        for resolution in 0..=15 {
            let cf_name = format!("resolution_{}", resolution);
            if let Some(cf) = self.db.cf_handle(&cf_name) {
                let iter = self.db.iterator_cf(cf, rocksdb::IteratorMode::Start);
                let count = iter.count();
                stats.insert(cf_name, count);
            }
        }
        
        Ok(stats)
    }
    
    /// Convert cell index to cache key
    fn cell_index_to_key(&self, cell_index: CellIndex) -> Vec<u8> {
        // Use the cell index as bytes for the key
        cell_index.to_string().into_bytes()
    }
    
    /// Compute neighbors for a cell using geographic distance
    fn compute_neighbors(&self, cell_index: CellIndex, all_cells: &[(CellIndex, f64, f64)]) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
        // Get center of target cell
        let target_latlng = h3o::LatLng::from(cell_index);
        let target_lat = target_latlng.lat_radians().to_degrees();
        let target_lng = target_latlng.lng_radians().to_degrees();
        
        let mut neighbors = Vec::new();
        
        // Find cells within threshold distance
        for (i, &(other_cell_index, other_lat, other_lng)) in all_cells.iter().enumerate() {
            if cell_index != other_cell_index {
                let distance = self.calculate_latlng_distance(target_lat, target_lng, other_lat, other_lng);
                
                if distance < self.neighbor_threshold {
                    neighbors.push(i);
                }
            }
        }
        
        // Limit to reasonable number of neighbors
        neighbors.truncate(12);
        Ok(neighbors)
    }
    
    /// Calculate distance between two lat/lng points in degrees
    fn calculate_latlng_distance(&self, lat1: f64, lng1: f64, lat2: f64, lng2: f64) -> f64 {
        let dlat = lat2 - lat1;
        let dlng = lng2 - lng1;
        (dlat * dlat + dlng * dlng).sqrt()
    }
    
    /// Serialize neighbor indices to bytes
    fn serialize_neighbors(&self, neighbors: &[usize]) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        Ok(serde_json::to_vec(neighbors)?)
    }
    
    /// Deserialize neighbor indices from bytes
    fn deserialize_neighbors(&self, data: &[u8]) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
        Ok(serde_json::from_slice(data)?)
    }
}

impl Drop for H3NeighborCache {
    fn drop(&mut self) {
        // RocksDB will automatically flush and close
    }
}
