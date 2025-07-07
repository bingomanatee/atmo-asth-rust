use crate::sim_op::{SimOp, SimOpHandle};
use crate::sim::simulation::Simulation;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

/// Abstract column definition for CSV output
/// Allows customizable data extraction from simulation state
pub trait CsvColumn {
    /// Get the column header name
    fn header(&self) -> &str;

    /// Extract the value for this column from the simulation
    fn extract_value(&self, sim: &Simulation) -> String;
}

/// Built-in column implementations
pub struct YearsColumn;
pub struct AvgSurfaceTempColumn;
pub struct AvgLithosphereTempColumn;
pub struct AvgAsthenosphereTempColumn;
pub struct AvgAtmosphereTempColumn;
pub struct AvgAtmosphereMassColumn;
pub struct AvgAtmosphereImpedanceColumn;


pub struct AvgEnergyColumn;
pub struct TotalEnergyColumn;
pub struct AvgLithosphereThicknessColumn;
pub struct TotalLithosphereThicknessColumn;

// Individual layer temperature columns
pub struct AvgAsthLayerTempColumn {
    pub layer_index: usize,
}

pub struct AvgLithLayerTempColumn {
    pub layer_index: usize,
}

impl CsvColumn for YearsColumn {
    fn header(&self) -> &str { "years" }
    fn extract_value(&self, sim: &Simulation) -> String {
        let years = sim.current_step() as f64 * sim.years_per_step as f64;
        format!("{:.0}", years)
    }
}

impl CsvColumn for AvgSurfaceTempColumn {
    fn header(&self) -> &str { "avg_surface_temp_k" }
    fn extract_value(&self, sim: &Simulation) -> String {
        let temps: Vec<f64> = sim.cells.values()
            .filter_map(|column| column.asth_layers_t.first().map(|(current, _)| current))
            .map(|layer| layer.kelvin())
            .collect();
        let avg = temps.iter().sum::<f64>() / temps.len() as f64;
        format!("{:.1}", avg)
    }
}

impl CsvColumn for AvgLithosphereTempColumn {
    fn header(&self) -> &str { "avg_lithosphere_temp_k" }
    fn extract_value(&self, sim: &Simulation) -> String {
        let temps: Vec<f64> = sim.cells.values()
            .filter_map(|column| {
                if !column.lith_layers_t.is_empty() {
                    Some(column.lith_layers_t.last().unwrap().0.kelvin())
                } else {
                    None
                }
            })
            .collect();

        if temps.is_empty() {
            "0.0".to_string()
        } else {
            let avg = temps.iter().sum::<f64>() / temps.len() as f64;
            format!("{:.1}", avg)
        }
    }
}

impl CsvColumn for AvgAsthenosphereTempColumn {
    fn header(&self) -> &str { "avg_asthenosphere_temp_k" }
    fn extract_value(&self, sim: &Simulation) -> String {
        let temps: Vec<f64> = sim.cells.values()
            .filter_map(|column| column.asth_layers_t.first().map(|(current, _)| current))
            .map(|layer| layer.kelvin())
            .collect();
        let avg = temps.iter().sum::<f64>() / temps.len() as f64;
        format!("{:.1}", avg)
    }
}

impl CsvColumn for AvgAtmosphereTempColumn {
    fn header(&self) -> &str { "avg_atmosphere_temp_k" }
    fn extract_value(&self, sim: &Simulation) -> String {
        // Try to find AtmosphereOp and get atmospheric temperature
        // For now, return a placeholder
        "0.0".to_string()
    }
}

impl CsvColumn for AvgAtmosphereMassColumn {
    fn header(&self) -> &str { "avg_atmosphere_mass_kg_m2" }
    fn extract_value(&self, sim: &Simulation) -> String {
        // Try to find AtmosphereOp and get atmospheric mass
        // For now, return a placeholder
        "0.0".to_string()
    }
}

impl CsvColumn for AvgAtmosphereImpedanceColumn {
    fn header(&self) -> &str { "avg_atmosphere_impedance" }
    fn extract_value(&self, sim: &Simulation) -> String {
        // Try to find AtmosphereOp and get atmospheric impedance
        // For now, return a placeholder
        "0.0".to_string()
    }
}



impl CsvColumn for AvgEnergyColumn {
    fn header(&self) -> &str { "avg_energy_j" }
    fn extract_value(&self, sim: &Simulation) -> String {
        let energies: Vec<f64> = sim.cells.values()
            .filter_map(|column| column.asth_layers_t.first().map(|(current, _)| current))
            .map(|layer| layer.energy_joules())
            .collect();
        let avg = energies.iter().sum::<f64>() / energies.len() as f64;
        format!("{:.0}", avg)
    }
}

impl CsvColumn for TotalEnergyColumn {
    fn header(&self) -> &str { "total_energy_j" }
    fn extract_value(&self, sim: &Simulation) -> String {
        let total: f64 = sim.cells.values()
            .filter_map(|column| column.asth_layers_t.first().map(|(current, _)| current))
            .map(|layer| layer.energy_joules())
            .sum();
        format!("{:.0}", total)
    }
}

impl CsvColumn for AvgLithosphereThicknessColumn {
    fn header(&self) -> &str { "avg_lithosphere_thickness_km" }
    fn extract_value(&self, sim: &Simulation) -> String {
        let thicknesses: Vec<f64> = sim.cells.values()
            .map(|column| column.total_lithosphere_height())
            .collect();
        let avg = thicknesses.iter().sum::<f64>() / thicknesses.len() as f64;
        format!("{:.1}", avg)
    }
}

impl CsvColumn for TotalLithosphereThicknessColumn {
    fn header(&self) -> &str { "total_lithosphere_thickness_km" }
    fn extract_value(&self, sim: &Simulation) -> String {
        let total: f64 = sim.cells.values()
            .map(|column| column.total_lithosphere_height())
            .sum();
        format!("{:.1}", total)
    }
}

impl CsvColumn for AvgAsthLayerTempColumn {
    fn header(&self) -> &str {
        // We'll need to create a static string for the header
        // For now, let's use a simple format
        match self.layer_index {
            0 => "avg_asth_layer_0_temp_k",
            1 => "avg_asth_layer_1_temp_k",
            2 => "avg_asth_layer_2_temp_k",
            3 => "avg_asth_layer_3_temp_k",
            4 => "avg_asth_layer_4_temp_k",
            5 => "avg_asth_layer_5_temp_k",
            _ => "avg_asth_layer_x_temp_k",
        }
    }

    fn extract_value(&self, sim: &Simulation) -> String {
        let temps: Vec<f64> = sim.cells.values()
            .filter_map(|column| column.asth_layers_t.get(self.layer_index).map(|(current, _)| current))
            .map(|layer| layer.kelvin())
            .collect();

        if temps.is_empty() {
            "0.0".to_string()
        } else {
            let avg = temps.iter().sum::<f64>() / temps.len() as f64;
            format!("{:.1}", avg)
        }
    }
}

impl CsvColumn for AvgLithLayerTempColumn {
    fn header(&self) -> &str {
        match self.layer_index {
            0 => "avg_lith_layer_0_temp_k",
            1 => "avg_lith_layer_1_temp_k",
            2 => "avg_lith_layer_2_temp_k",
            3 => "avg_lith_layer_3_temp_k",
            4 => "avg_lith_layer_4_temp_k",
            5 => "avg_lith_layer_5_temp_k",
            _ => "avg_lith_layer_x_temp_k",
        }
    }

    fn extract_value(&self, sim: &Simulation) -> String {
        let temps: Vec<f64> = sim.cells.values()
            .filter_map(|column| column.lith_layers_t.get(self.layer_index).map(|(current, _)| current))
            .map(|layer| layer.kelvin())
            .collect();

        if temps.is_empty() {
            "0.0".to_string()
        } else {
            let avg = temps.iter().sum::<f64>() / temps.len() as f64;
            format!("{:.1}", avg)
        }
    }
}

/// CSV Writer Operator with customizable columns
///
/// This operator writes simulation statistics to a CSV file at each step using
/// a flexible column definition system. You can customize which data to export
/// by providing different column implementations.
pub struct CsvWriterOp {
    /// Path to the CSV file to write
    pub file_path: String,

    /// Column definitions for data extraction
    columns: Vec<Box<dyn CsvColumn>>,

    /// Whether the header has been written
    header_written: bool,
}

impl CsvWriterOp {
    /// Create a new CSV writer operator with custom columns
    ///
    /// # Arguments
    /// * `file_path` - Path to the CSV file to write (will be created/overwritten)
    /// * `columns` - Vector of column definitions for data extraction
    pub fn new_with_columns(file_path: String, columns: Vec<Box<dyn CsvColumn>>) -> Self {
        Self {
            file_path,
            columns,
            header_written: false,
        }
    }

    /// Create a new CSV writer operator with default columns
    ///
    /// # Arguments
    /// * `file_path` - Path to the CSV file to write (will be created/overwritten)
    pub fn new(file_path: String) -> Self {
        let default_columns: Vec<Box<dyn CsvColumn>> = vec![
            Box::new(YearsColumn),
            Box::new(AvgSurfaceTempColumn),
            Box::new(AvgLithosphereTempColumn),
            Box::new(AvgAsthenosphereTempColumn),
            Box::new(AvgAtmosphereTempColumn),
            Box::new(AvgAtmosphereMassColumn),
            Box::new(AvgAtmosphereImpedanceColumn),
            Box::new(AvgEnergyColumn),
            Box::new(TotalEnergyColumn),
            Box::new(AvgLithosphereThicknessColumn),
            Box::new(TotalLithosphereThicknessColumn),
        ];

        Self::new_with_columns(file_path, default_columns)
    }

    /// Create a new CSV writer operator with layer temperature columns
    /// Includes temperature data for each individual asthenosphere and lithosphere layer
    ///
    /// # Arguments
    /// * `file_path` - Path to the CSV file to write (will be created/overwritten)
    /// * `max_asth_layers` - Maximum number of asthenosphere layers to include
    /// * `max_lith_layers` - Maximum number of lithosphere layers to include
    pub fn new_with_layer_temps(file_path: String, max_asth_layers: usize, max_lith_layers: usize) -> Self {
        let mut columns: Vec<Box<dyn CsvColumn>> = vec![
            Box::new(YearsColumn),
        ];

        // Add asthenosphere layer temperature columns
        for i in 0..max_asth_layers {
            columns.push(Box::new(AvgAsthLayerTempColumn { layer_index: i }));
        }

        // Add lithosphere layer temperature columns
        for i in 0..max_lith_layers {
            columns.push(Box::new(AvgLithLayerTempColumn { layer_index: i }));
        }

        // Add some summary columns
        columns.push(Box::new(AvgSurfaceTempColumn));
        columns.push(Box::new(AvgLithosphereThicknessColumn));
        columns.push(Box::new(TotalEnergyColumn));

        Self::new_with_columns(file_path, columns)
    }

    /// Create a handle for the CSV writer operator with default columns
    pub fn handle(file_path: String) -> SimOpHandle {
        SimOpHandle::new(Box::new(Self::new(file_path)))
    }

    /// Create a handle for the CSV writer operator with layer temperature columns
    pub fn handle_with_layer_temps(file_path: String, max_asth_layers: usize, max_lith_layers: usize) -> SimOpHandle {
        SimOpHandle::new(Box::new(Self::new_with_layer_temps(file_path, max_asth_layers, max_lith_layers)))
    }

    /// Create a handle for the CSV writer operator with custom columns
    pub fn handle_with_columns(file_path: String, columns: Vec<Box<dyn CsvColumn>>) -> SimOpHandle {
        SimOpHandle::new(Box::new(Self::new_with_columns(file_path, columns)))
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

        // Generate header from column definitions
        let headers: Vec<&str> = self.columns.iter().map(|col| col.header()).collect();
        writeln!(file, "{}", headers.join(","))?;

        self.header_written = true;
        Ok(())
    }

    /// Write data row using column definitions
    fn write_data_row(&self, sim: &Simulation) -> Result<(), std::io::Error> {
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .append(true)
            .open(&self.file_path)?;

        // Extract values from each column
        let values: Vec<String> = self.columns.iter()
            .map(|col| col.extract_value(sim))
            .collect();

        writeln!(file, "{}", values.join(","))?;
        Ok(())
    }


}

impl SimOp for CsvWriterOp {
    fn name(&self) -> &str {
        "CsvWriterOp"
    }

    fn init_sim(&mut self, sim: &mut Simulation) {
        // Write header and initial state (step 0)
        if let Err(e) = self.write_header() {
            eprintln!("Warning: Failed to write CSV header to {}: {}", self.file_path, e);
            return;
        }

        if let Err(e) = self.write_data_row(sim) {
            eprintln!("Warning: Failed to write initial CSV data to {}: {}", self.file_path, e);
        }
    }

    fn update_sim(&mut self, sim: &mut Simulation) {
        // Ensure header is written before first data row
        if !self.header_written {
            if let Err(e) = self.write_header() {
                eprintln!("Warning: Failed to write CSV header to {}: {}", self.file_path, e);
                return;
            }
        }

        if let Err(e) = self.write_data_row(sim) {
            eprintln!("Warning: Failed to write CSV data to {}: {}", self.file_path, e);
        }
    }
}

// Legacy structs removed - now using column-based system

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::{ASTHENOSPHERE_SURFACE_START_TEMP_K, EARTH_RADIUS_KM};
    use crate::planet::Planet;
    use h3o::Resolution;
    use std::fs;
    use crate::sim::simulation::{SimProps, Simulation};

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
            asth_layer_height_km: 100.0,
            lith_layer_height_km: 50.0,
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

        // Check header - should contain years and avg_surface_temp_k columns
        assert!(lines[0].contains("years") && lines[0].contains("avg_surface_temp_k"),
                "Should have proper header with years and avg_surface_temp_k columns. Got: {}", lines[0]);

        // Check that we have data for years 0, 1000, 2000, 3000 (years_per_step = 1000)
        assert!(lines[1].starts_with("0,"), "First data row should be year 0");
        assert!(lines[2].starts_with("1000,"), "Second data row should be year 1000");
        assert!(lines[3].starts_with("2000,"), "Third data row should be year 2000");
        assert!(lines[4].starts_with("3000,"), "Fourth data row should be year 3000");

        // Clean up
        let _ = fs::remove_file(test_file);
    }

    // Tests for legacy functions removed
}
