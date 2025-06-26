use crate::sim::sim_op::{SimOp, SimOpHandle};
use crate::sim::Simulation;

/// Progress Reporter Operator
/// 
/// Provides intermittent progress updates during simulation runs.
/// Reports key thermal parameters at configurable intervals.
#[derive(Debug, Clone)]
pub struct ProgressReporterOp {
    pub name: String,
    pub report_interval: i32,  // Report every N steps
    pub show_atmosphere: bool, // Whether to show atmospheric data
    pub show_lithosphere: bool, // Whether to show lithosphere data
}

impl ProgressReporterOp {
    /// Create a new progress reporter with default settings
    pub fn new(report_interval: i32) -> Self {
        Self {
            name: "ProgressReporterOp".to_string(),
            report_interval,
            show_atmosphere: true,
            show_lithosphere: true,
        }
    }

    /// Create a new progress reporter with custom settings
    pub fn new_with_options(
        report_interval: i32,
        show_atmosphere: bool,
        show_lithosphere: bool,
    ) -> Self {
        Self {
            name: "ProgressReporterOp".to_string(),
            report_interval,
            show_atmosphere,
            show_lithosphere,
        }
    }

    /// Create a handle for the progress reporter with default settings
    pub fn handle(report_interval: i32) -> SimOpHandle {
        SimOpHandle::new(Box::new(Self::new(report_interval)))
    }

    /// Create a handle for the progress reporter with custom settings
    pub fn handle_with_options(
        report_interval: i32,
        show_atmosphere: bool,
        show_lithosphere: bool,
    ) -> SimOpHandle {
        SimOpHandle::new(Box::new(Self::new_with_options(
            report_interval,
            show_atmosphere,
            show_lithosphere,
        )))
    }

    /// Calculate average surface temperature across all cells
    fn calculate_avg_surface_temp(&self, sim: &Simulation) -> f64 {
        let temps: Vec<f64> = sim.cells.values()
            .map(|cell| cell.layers[0].kelvin())
            .collect();
        
        if temps.is_empty() {
            0.0
        } else {
            temps.iter().sum::<f64>() / temps.len() as f64
        }
    }

    /// Calculate average lithosphere thickness across all cells
    fn calculate_avg_lithosphere_thickness(&self, sim: &Simulation) -> f64 {
        let thicknesses: Vec<f64> = sim.cells.values()
            .map(|cell| cell.total_lithosphere_height())
            .collect();
        
        if thicknesses.is_empty() {
            0.0
        } else {
            thicknesses.iter().sum::<f64>() / thicknesses.len() as f64
        }
    }

    /// Calculate total system energy across all cells
    fn calculate_total_energy(&self, sim: &Simulation) -> f64 {
        sim.cells.values()
            .map(|cell| {
                let asth_energy: f64 = cell.layers.iter().map(|l| l.energy_joules()).sum();
                let lith_energy: f64 = cell.lithospheres.iter().map(|l| l.energy()).sum();
                asth_energy + lith_energy
            })
            .sum()
    }

    /// Find atmosphere data from AtmosphereOp if available
    fn find_atmosphere_data(&self, _sim: &Simulation) -> (f64, f64, f64) {
        // TODO: In the future, we could search through sim.ops to find AtmosphereOp
        // and extract actual atmospheric data. For now, return placeholders.
        (0.0, 0.0, 0.0) // (mass_kg_m2, outgassing_rate, impedance)
    }
}

impl SimOp for ProgressReporterOp {
    fn name(&self) -> &str {
        &self.name
    }

    fn init_sim(&mut self, sim: &mut Simulation) {
        println!("🚀 Starting thermal simulation...");
        
        // Report initial state
        let surface_temp = self.calculate_avg_surface_temp(sim);
        let lithosphere_thickness = self.calculate_avg_lithosphere_thickness(sim);
        
        println!("   Step {}: Surface temp = {:.1}K, Lithosphere = {:.1}km", 
                 sim.current_step(), surface_temp, lithosphere_thickness);
        
        if self.show_atmosphere {
            let (atm_mass, outgassing, impedance) = self.find_atmosphere_data(sim);
            println!("Step {}: Atmosphere = {:.2} kg/m², Outgassing = {:.2e} kg/year, Impedance = {:.1}%", 
                     sim.current_step(), atm_mass, outgassing, impedance * 100.0);
        }
    }

    fn update_sim(&mut self, sim: &mut Simulation) {
        let current_step = sim.current_step();
        
        // Only report at specified intervals
        if current_step % self.report_interval == 0 {
            let surface_temp = self.calculate_avg_surface_temp(sim);
            
            if self.show_lithosphere {
                let lithosphere_thickness = self.calculate_avg_lithosphere_thickness(sim);
                println!("   Step {}: Surface temp = {:.1}K, Lithosphere = {:.1}km", 
                         current_step, surface_temp, lithosphere_thickness);
            } else {
                println!("   Step {}: Surface temp = {:.1}K", current_step, surface_temp);
            }
        }
        
        // Show atmosphere data at different intervals if enabled
        if self.show_atmosphere && current_step % (self.report_interval + 20) == 0 {
            let (atm_mass, outgassing, impedance) = self.find_atmosphere_data(sim);
            println!("Step {}: Atmosphere = {:.2} kg/m², Outgassing = {:.2e} kg/year, Impedance = {:.1}%", 
                     current_step, atm_mass, outgassing, impedance * 100.0);
        }
    }

    fn after_sim(&mut self, sim: &mut Simulation) {
        println!("\n📈 Simulation Complete!");
        
        // Final summary
        let surface_temp = self.calculate_avg_surface_temp(sim);
        let lithosphere_thickness = self.calculate_avg_lithosphere_thickness(sim);
        let total_energy = self.calculate_total_energy(sim);
        
        println!("   Final surface temperature: {:.1}K ({:.1}°C)", 
                 surface_temp, surface_temp - 273.15);
        println!("   Final lithosphere thickness: {:.1}km", lithosphere_thickness);
        println!("   Total system energy: {:.2e} J", total_energy);
        
        // Calculate simulation time span
        let total_years = sim.current_step() as f64 * sim.years_per_step as f64;
        if total_years >= 1_000_000.0 {
            println!("   Simulation time span: {:.1} million years", total_years / 1_000_000.0);
        } else if total_years >= 1_000.0 {
            println!("   Simulation time span: {:.0} thousand years", total_years / 1_000.0);
        } else {
            println!("   Simulation time span: {:.0} years", total_years);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::EARTH_RADIUS_KM;
    use crate::planet::Planet;
    use crate::sim::{SimProps, Simulation};
    use h3o::Resolution;

    #[test]
    fn test_progress_reporter_creation() {
        let reporter = ProgressReporterOp::new(10);
        assert_eq!(reporter.report_interval, 10);
        assert_eq!(reporter.show_atmosphere, true);
        assert_eq!(reporter.show_lithosphere, true);
    }

    #[test]
    fn test_progress_reporter_with_options() {
        let reporter = ProgressReporterOp::new_with_options(5, false, true);
        assert_eq!(reporter.report_interval, 5);
        assert_eq!(reporter.show_atmosphere, false);
        assert_eq!(reporter.show_lithosphere, true);
    }

    #[test]
    fn test_progress_reporter_calculations() {
        let mut sim = Simulation::new(SimProps {
            name: "progress_test",
            planet: Planet {
                radius_km: EARTH_RADIUS_KM as f64,
                resolution: Resolution::Zero,
            },
            ops: vec![],
            res: Resolution::Two,
            layer_count: 3,
            layer_height_km: 50.0,
            sim_steps: 1,
            years_per_step: 1000,
            debug: false,
            alert_freq: 1,
            starting_surface_temp_k: 1500.0,
        });

        let reporter = ProgressReporterOp::new(1);
        
        // Test temperature calculation
        let avg_temp = reporter.calculate_avg_surface_temp(&sim);
        assert!(avg_temp > 1000.0); // Should be reasonable temperature
        
        // Test lithosphere thickness calculation
        let avg_thickness = reporter.calculate_avg_lithosphere_thickness(&sim);
        assert!(avg_thickness >= 0.0); // Should be non-negative
        
        // Test energy calculation
        let total_energy = reporter.calculate_total_energy(&sim);
        assert!(total_energy > 0.0); // Should have positive energy
    }
}
