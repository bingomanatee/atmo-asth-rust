/// Temperature reporting operation
/// Reports average surface temperature every 10% of the simulation cycle

use crate::sim::sim_op::SimOp;
use crate::sim::simulation::Simulation;
use crate::energy_mass_composite::MaterialPhase;

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
}

impl SimOp for TemperatureReportingOp {
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
    }
    
    fn update_sim(&mut self, sim: &mut Simulation) {
        if !self.should_report(sim) {
            return;
        }
        
        let surface_temp = self.calculate_average_surface_temperature(sim);
        let (global_avg, min_temp, max_temp, _) = self.calculate_global_statistics(sim);
        let progress_percent = (sim.step as f64 / sim.sim_steps as f64) * 100.0;
        
        println!("🌡️  Temperature Report - Step {} ({:.1}% complete):", 
                 sim.step, progress_percent);
        println!("   - Average surface temperature: {:.1}K ({:.1}°C)", 
                 surface_temp, surface_temp - 273.15);
        println!("   - Global average temperature: {:.1}K ({:.1}°C)", 
                 global_avg, global_avg - 273.15);
        println!("   - Temperature range: {:.1}K to {:.1}K", min_temp, max_temp);
        
        self.last_reported_step = sim.step;
    }
}
