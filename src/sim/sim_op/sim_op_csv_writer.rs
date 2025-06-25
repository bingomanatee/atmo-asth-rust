use crate::sim::sim_op::{SimOp, SimOpHandle};
use crate::sim::Simulation;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

/// CSV Writer Operator
/// 
/// This operator writes simulation statistics to a CSV file at each step.
/// It tracks temperature, energy, volume, and lithosphere statistics across all cells.
/// 
/// The CSV includes columns for:
/// - step: simulation step number
/// - avg_temp_k, min_temp_k, max_temp_k: temperature statistics in Kelvin
/// - avg_energy_j, min_energy_j, max_energy_j: energy statistics in Joules
/// - avg_volume_km3, min_volume_km3, max_volume_km3: volume statistics in km³
/// - avg_lithosphere_km, min_lithosphere_km, max_lithosphere_km: lithosphere thickness statistics
/// - total_energy_j: total energy across all cells
/// - total_lithosphere_km: total lithosphere thickness across all cells
pub struct CsvWriterOp {
    /// Path to the CSV file to write
    pub file_path: String,
    
    /// Whether the header has been written
    header_written: bool,
}

impl CsvWriterOp {
    /// Create a new CSV writer operator
    /// 
    /// # Arguments
    /// * `file_path` - Path to the CSV file to write (will be created/overwritten)
    pub fn new(file_path: String) -> Self {
        Self {
            file_path,
            header_written: false,
        }
    }

    /// Create a handle for the CSV writer operator
    pub fn handle(file_path: String) -> SimOpHandle {
        SimOpHandle::new(Box::new(Self::new(file_path)))
    }

    /// Write the CSV header if not already written
    fn write_header(&mut self) -> Result<(), std::io::Error> {
        if self.header_written {
            return Ok(());
        }

        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&self.file_path)?;

        writeln!(file, "step,avg_temp_k,avg_energy_j,avg_volume_km3,avg_lithosphere_km,total_energy_j,total_lithosphere_km")?;
        
        self.header_written = true;
        Ok(())
    }

    /// Calculate statistics for the current simulation state
    fn calculate_stats(&self, sim: &Simulation) -> SimulationStats {
        let mut temperatures = Vec::new();
        let mut energies = Vec::new();
        let mut volumes = Vec::new();
        let mut lithosphere_thicknesses = Vec::new();

        for column in sim.cells.values() {
            // Get surface layer (layer 0) statistics
            if let Some(layer) = column.layers.first() {
                temperatures.push(layer.kelvin());
                energies.push(layer.energy_joules());
                volumes.push(layer.volume_km3());
            }
            
            // Get lithosphere thickness
            let lithosphere_thickness = column.total_lithosphere_height();
            lithosphere_thicknesses.push(lithosphere_thickness);
        }

        // Calculate statistics
        let temp_stats = calculate_min_max_avg(&temperatures);
        let energy_stats = calculate_min_max_avg(&energies);
        let volume_stats = calculate_min_max_avg(&volumes);
        let lithosphere_stats = calculate_min_max_avg(&lithosphere_thicknesses);

        SimulationStats {
            step: sim.current_step(),
            temp_stats,
            energy_stats,
            volume_stats,
            lithosphere_stats,
            total_energy: energies.iter().sum(),
            total_lithosphere: lithosphere_thicknesses.iter().sum(),
        }
    }

    /// Write statistics to the CSV file
    fn write_stats(&self, stats: &SimulationStats) -> Result<(), std::io::Error> {
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .append(true)
            .open(&self.file_path)?;

        // Cap energy values to prevent overflow when converting to integers
        // When values exceed i64::MAX, just use i64::MAX
        let cap_energy = |energy: f64| -> i64 {
            if energy > i64::MAX as f64 {
                i64::MAX
            } else if energy < 0.0 {
                0
            } else {
                energy as i64
            }
        };

        writeln!(
            file,
            "{},{},{},{},{},{},{}",
            stats.step,
            stats.temp_stats.avg as i32,// stats.temp_stats.min as i32, stats.temp_stats.max as i32,
            cap_energy(stats.energy_stats.avg),// cap_energy(stats.energy_stats.min), cap_energy(stats.energy_stats.max),
            stats.volume_stats.avg as i32, //stats.volume_stats.min as i32, stats.volume_stats.max as i32,
            stats.lithosphere_stats.avg as i32, //stats.lithosphere_stats.min as i32, stats.lithosphere_stats.max as i32,
            cap_energy(stats.total_energy),
            stats.total_lithosphere as i32
        )?;

        Ok(())
    }
}

impl SimOp for CsvWriterOp {
    fn init_sim(&mut self, sim: &mut Simulation) {
        // Write header and initial state (step 0)
        if let Err(e) = self.write_header() {
            eprintln!("Warning: Failed to write CSV header to {}: {}", self.file_path, e);
            return;
        }

        let stats = self.calculate_stats(sim);
        if let Err(e) = self.write_stats(&stats) {
            eprintln!("Warning: Failed to write initial CSV data to {}: {}", self.file_path, e);
        }
    }

    fn update_sim(&mut self, sim: &mut Simulation) {
        let stats = self.calculate_stats(sim);
        if let Err(e) = self.write_stats(&stats) {
            eprintln!("Warning: Failed to write CSV data to {}: {}", self.file_path, e);
        }
    }
}

/// Statistics for a single value type (min, max, average)
#[derive(Debug, Clone)]
struct Stats {
    min: f64,
    max: f64,
    avg: f64,
}

/// Complete simulation statistics for one step
#[derive(Debug, Clone)]
struct SimulationStats {
    step: i32,
    temp_stats: Stats,
    energy_stats: Stats,
    volume_stats: Stats,
    lithosphere_stats: Stats,
    total_energy: f64,
    total_lithosphere: f64,
}

/// Calculate min, max, and average for a vector of values
fn calculate_min_max_avg(values: &[f64]) -> Stats {
    if values.is_empty() {
        return Stats { min: 0.0, max: 0.0, avg: 0.0 };
    }

    let min = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let sum: f64 = values.iter().sum();
    let avg = sum / values.len() as f64;

    Stats { min, max, avg }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::{ASTHENOSPHERE_SURFACE_START_TEMP_K, EARTH_RADIUS_KM};
    use crate::planet::Planet;
    use crate::sim::{SimProps, Simulation};
    use h3o::Resolution;
    use std::fs;

    #[test]
    fn test_csv_writer_creates_file() {
        let test_file = "test_output.csv";
        
        // Clean up any existing test file
        let _ = fs::remove_file(test_file);

        let mut sim = Simulation::new(SimProps {
            name: "csv_writer_test",
            planet: Planet {
                radius_km: EARTH_RADIUS_KM as f64,
                resolution: Resolution::One,
            },
            ops: vec![CsvWriterOp::handle(test_file.to_string())],
            res: Resolution::Two,
            layer_count: 4,
            layer_height: 10.0,
            layer_height_km: 10.0,
            sim_steps: 3,
            years_per_step: 1000,
            debug: false,
            alert_freq: 1,
            starting_surface_temp_k: ASTHENOSPHERE_SURFACE_START_TEMP_K,
        });

        sim.simulate();

        // Check that file was created and has content
        assert!(Path::new(test_file).exists(), "CSV file should be created");
        
        let content = fs::read_to_string(test_file).expect("Should be able to read CSV file");
        let lines: Vec<&str> = content.lines().collect();
        
        // Should have header + initial state + 3 simulation steps = 5 lines total
        assert_eq!(lines.len(), 5, "Should have header + 4 data rows");
        
        // Check header
        assert!(lines[0].contains("step,avg_temp_k"), "Should have proper header");
        
        // Check that we have data for steps 0, 1, 2, 3
        assert!(lines[1].starts_with("0,"), "First data row should be step 0");
        assert!(lines[2].starts_with("1,"), "Second data row should be step 1");
        assert!(lines[3].starts_with("2,"), "Third data row should be step 2");
        assert!(lines[4].starts_with("3,"), "Fourth data row should be step 3");

        // Clean up
        let _ = fs::remove_file(test_file);
    }

    #[test]
    fn test_calculate_min_max_avg() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = calculate_min_max_avg(&values);
        
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        assert_eq!(stats.avg, 3.0);
    }

    #[test]
    fn test_calculate_min_max_avg_empty() {
        let values = vec![];
        let stats = calculate_min_max_avg(&values);
        
        assert_eq!(stats.min, 0.0);
        assert_eq!(stats.max, 0.0);
        assert_eq!(stats.avg, 0.0);
    }
}
