/// Temperature reporting operation
/// Reports average surface temperature every 10% of the simulation cycle

use crate::sim_op::SimOp;
use crate::energy_mass_composite::{EnergyMassComposite, MaterialPhase};
use crate::sim::simulation::Simulation;
use std::any::Any;


pub struct TemperatureReportingOp {
    pub report_frequency_percent: f64,
    last_reported_step: i32,
}

impl TemperatureReportingOp {
    pub fn new() -> Self {
        Self {
            report_frequency_percent: 10.0, // Report every 10% of simulation
            last_reported_step: -1,
        }
    }
    
    pub fn with_frequency(report_frequency_percent: f64) -> Self {
        Self {
            report_frequency_percent,
            last_reported_step: -1,
        }
    }
    
    fn should_report(&mut self, sim: &Simulation) -> bool {
        if sim.sim_steps == 0 {
            return false;
        }
        
        let progress_percent = (sim.step as f64 / sim.sim_steps as f64) * 100.0;
        let report_interval = self.report_frequency_percent;
        let current_report_milestone = (progress_percent / report_interval).floor() as i32;
        let last_report_milestone = if self.last_reported_step < 0 {
            -1
        } else {
            ((self.last_reported_step as f64 / sim.sim_steps as f64) * 100.0 / report_interval).floor() as i32
        };
        
        current_report_milestone > last_report_milestone
    }
    
    fn calculate_average_surface_temperature(&self, sim: &Simulation) -> f64 {
        let mut total_temp = 0.0;
        let mut surface_layer_count = 0;
        
        for cell in sim.cells.values() {
            // Find surface layers (layers that cross or are near the surface)
            for (current, _) in &cell.layers_t {
                let layer_center_depth = current.start_depth_km + (current.height_km / 2.0);
                
                // Consider layers within 10km of surface as "surface" layers
                if layer_center_depth >= -10.0 && layer_center_depth <= 10.0 {
                    total_temp += current.temperature_k();
                    surface_layer_count += 1;
                }
            }
        }
        
        if surface_layer_count > 0 {
            total_temp / surface_layer_count as f64
        } else {
            0.0
        }
    }
    
    fn calculate_global_statistics(&self, sim: &Simulation) -> (f64, f64, f64, usize) {
        let mut total_temp = 0.0;
        let mut min_temp = f64::INFINITY;
        let mut max_temp = f64::NEG_INFINITY;
        let mut layer_count = 0;
        
        for cell in sim.cells.values() {
            for (current, _) in &cell.layers_t {
                let temp = current.temperature_k();
                total_temp += temp;
                min_temp = min_temp.min(temp);
                max_temp = max_temp.max(temp);
                layer_count += 1;
            }
        }
        
        let avg_temp = if layer_count > 0 { total_temp / layer_count as f64 } else { 0.0 };
        (avg_temp, min_temp, max_temp, layer_count)
    }

    /// Report detailed information about the first cell's layers (wide experiment style)
    fn report_first_cell_details(&self, sim: &Simulation, label: &str) {
        if let Some(first_cell) = sim.cells.values().next() {
            println!("🔬 {} Cell Details (Cell {:?}):", label, first_cell.h3_index);
            println!("   Surface Area: {:.2e} km²", first_cell.surface_area_km2());
            println!("   Planet: {}", first_cell.planet.name);
            println!("   Total layers in this cell: {}", first_cell.layers_t.len());

            // Print column headers with consistent widths
            println!();
            println!("   {:>3} {:>12} {:>8} {:>3} {:>8} {:>12} {:>8} {:>8} {:>12} {:>12} {:>10} {:>8}",
                     "Lyr", "Depth Range", "Height", "", "Phase", "Material", "Temp(K)", "Temp(°C)", "Mass(kg)", "Volume(km³)", "Density(kg/m³)", "Opacity");
            println!("   {:->3} {:->12} {:->8} {:->3} {:->8} {:->12} {:->8} {:->8} {:->12} {:->12} {:->10} {:->8}",
                     "", "", "", "", "", "", "", "", "", "", "", "");

            // Show ALL layers for complete analysis
            for (i, (current, _)) in first_cell.layers_t.iter().enumerate() {
                let temp_k = current.temperature_k();
                let temp_c = temp_k - 273.15;
                let mass_kg = current.mass_kg();
                let volume_km3 = current.volume_km3();
                let density_kg_m3 = current.current_density_kg_m3();
                let phase = current.phase();
                let material_type = current.energy_mass.material_composite_type();
                let depth_range = format!("{:.1}-{:.1}km", current.start_depth_km, current.end_depth_km());

                // Format similar to wide experiment thermal state
                let layer_emoji = if current.is_atmospheric() { "☁️" } else { "🗻" };
                let phase_emoji = match phase {
                    MaterialPhase::Solid => "🧊",
                    MaterialPhase::Liquid => "💧",
                    MaterialPhase::Gas => "💨",
                };

                // Calculate opacity for gas layers (using same formula as space radiation operation)
                let opacity = if phase == MaterialPhase::Gas && current.height_km > 0.0 {
                    density_kg_m3 * 0.001 / current.height_km
                } else {
                    0.0 // Non-gas layers don't contribute to atmospheric opacity
                };

                // Format with consistent column widths
                println!("   {:>3} {:>12} {:>8.1} {:>3} {:>8} {:>12} {:>8.0} {:>8.0} {:>12.2e} {:>12.2e} {:>10.0} {:>8.6}",
                         format!("{}{}", layer_emoji, i),
                         depth_range,
                         current.height_km,
                         phase_emoji,
                         format!("{:?}", phase),
                         format!("{:?}", material_type),
                         temp_k,
                         temp_c,
                         mass_kg,
                         volume_km3,
                         density_kg_m3,
                         opacity);
            }
        }
    }
}

impl SimOp for TemperatureReportingOp {
    fn name(&self) -> &str {
        "TemperatureReporting"
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn init_sim(&mut self, sim: &mut Simulation) {
        println!("📊 Temperature reporting initialized (every {:.1}% of simulation)",
                 self.report_frequency_percent);

        // Report initial state
        let surface_temp = self.calculate_average_surface_temperature(sim);
        let (global_avg, min_temp, max_temp, layer_count) = self.calculate_global_statistics(sim);

        println!("🌡️  Initial Temperature Report:");
        println!("   - Average surface temperature: {:.1}K ({:.1}°C)",
                 surface_temp, surface_temp - 273.15);
        println!("   - Global average temperature: {:.1}K ({:.1}°C)",
                 global_avg, global_avg - 273.15);
        println!("   - Temperature range: {:.1}K to {:.1}K", min_temp, max_temp);
        println!("   - Total layers: {}", layer_count);

        // Show detailed first cell state (wide experiment style)
        self.report_first_cell_details(sim, "Initial");
    }
    
    fn update_sim(&mut self, sim: &mut Simulation) {
        if !self.should_report(sim) {
            return;
        }
        
        let progress = (sim.step as f64 / sim.sim_steps as f64) * 100.0;
        
        println!("step {} of {}: {}% progress",
                 sim.step, sim.sim_steps, progress.round() as i64);
        
       
        self.last_reported_step = sim.step;
    }

    fn after_sim(&mut self, sim: &mut Simulation) {
        println!("🏁 Final Temperature Report:");

        let surface_temp = self.calculate_average_surface_temperature(sim);
        let (global_avg, min_temp, max_temp, layer_count) = self.calculate_global_statistics(sim);

        println!("   - Final average surface temperature: {:.1}K ({:.1}°C)",
                 surface_temp, surface_temp - 273.15);
        println!("   - Final global average temperature: {:.1}K ({:.1}°C)",
                 global_avg, global_avg - 273.15);
        println!("   - Final temperature range: {:.1}K to {:.1}K", min_temp, max_temp);
        println!("   - Total layers: {}", layer_count);

        // Show detailed final cell state (wide experiment style)
        self.report_first_cell_details(sim, "Final");
    }
}
