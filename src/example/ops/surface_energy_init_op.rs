/// Surface energy initialization operation
/// Distributes initial surface energy based on geothermal gradient similar to wide experiment

use crate::sim::sim_op::SimOp;
use crate::sim::simulation::Simulation;
use crate::energy_mass_composite::EnergyMassComposite;

pub struct SurfaceEnergyInitOp {
    pub surface_temp_k: f64,
    pub geothermal_gradient_k_per_km: f64,
    pub core_temp_k: f64,
}

impl SurfaceEnergyInitOp {
    pub fn new() -> Self {
        Self {
            surface_temp_k: 280.0,           // 280K surface temperature
            geothermal_gradient_k_per_km: 25.0, // 25K per km depth
            core_temp_k: 1800.0,             // 1800K core temperature
        }
    }
    
    pub fn with_temperatures(surface_temp_k: f64, geothermal_gradient_k_per_km: f64, core_temp_k: f64) -> Self {
        Self {
            surface_temp_k,
            geothermal_gradient_k_per_km,
            core_temp_k,
        }
    }
    
    fn calculate_temperature_at_depth(&self, depth_km: f64) -> f64 {
        if depth_km <= 0.0 {
            // Atmospheric layers get surface temperature
            self.surface_temp_k
        } else {
            // Subsurface layers get geothermal gradient
            let temp = self.surface_temp_k + (depth_km * self.geothermal_gradient_k_per_km);
            temp.min(self.core_temp_k) // Cap at core temperature
        }
    }
}

impl SimOp for SurfaceEnergyInitOp {
    fn name(&self) -> &str {
        "SurfaceEnergyInit"
    }

    fn init_sim(&mut self, sim: &mut Simulation) {
        println!("🌡️  Initializing surface energy distribution with geothermal gradient...");

        let mut total_energy_applied = 0.0;
        let mut layers_initialized = 0;

        for cell in sim.cells.values_mut() {
            for (i, (current, next)) in cell.layers_t.iter_mut().enumerate() {
                // Calculate target temperature based on layer depth
                let layer_center_depth = current.start_depth_km + (current.height_km / 2.0);
                let target_temp_k = self.calculate_temperature_at_depth(layer_center_depth);

                // Set temperature for both current and next states
                current.energy_mass.set_temperature(target_temp_k);
                next.energy_mass.set_temperature(target_temp_k);

                total_energy_applied += current.energy_j();
                layers_initialized += 1;

                if sim.debug && i < 3 {
                    println!("  Layer {} at {:.1}km depth: {:.1}K",
                             i, layer_center_depth, target_temp_k);
                }
            }
        }

        println!("✅ Surface energy initialization complete:");
        println!("   - {} layers initialized", layers_initialized);
        println!("   - {:.2e} J total energy applied", total_energy_applied);
        println!("   - Surface temp: {:.1}K, Gradient: {:.1}K/km",
                 self.surface_temp_k, self.geothermal_gradient_k_per_km);
    }

    fn update_sim(&mut self, _sim: &mut Simulation) {
        // No ongoing updates needed - this is initialization only
    }
}
