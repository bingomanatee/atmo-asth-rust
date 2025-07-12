/// Surface energy initialization operation
/// Distributes initial surface energy based on geothermal gradient similar to wide experiment

use crate::sim_op::SimOp;
use crate::energy_mass_composite::EnergyMassComposite;
use crate::sim::simulation::Simulation;
use std::any::Any;
use rayon::prelude::*;


/// Parameters for surface energy initialization
#[derive(Debug, Clone)]
pub struct SurfaceEnergyInitParams {
    pub surface_temp_k: f64,
    pub geothermal_gradient_k_per_km: f64,
    pub core_temp_k: f64,
}

impl Default for SurfaceEnergyInitParams {
    fn default() -> Self {
        Self {
            surface_temp_k: 280.0,           // 280K surface temperature
            geothermal_gradient_k_per_km: 50.0, // 25K per km depth
            core_temp_k: 1800.0,             // 1800K core temperature
        }
    }
}

impl SurfaceEnergyInitParams {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_temperatures(surface_temp_k: f64, geothermal_gradient_k_per_km: f64, core_temp_k: f64) -> Self {
        Self {
            surface_temp_k,
            geothermal_gradient_k_per_km,
            core_temp_k,
        }
    }

}

pub struct SurfaceEnergyInitOp {
    pub params: SurfaceEnergyInitParams,
}

impl SurfaceEnergyInitOp {
    pub fn new() -> Self {
        Self {
            params: SurfaceEnergyInitParams::new(),
        }
    }

    pub fn new_with_params(params: SurfaceEnergyInitParams) -> Self {
        Self { params }
    }

    pub fn with_temperatures(surface_temp_k: f64, geothermal_gradient_k_per_km: f64, core_temp_k: f64) -> Self {
        Self {
            params: SurfaceEnergyInitParams::with_temperatures(surface_temp_k, geothermal_gradient_k_per_km, core_temp_k),
        }
    }

    fn calculate_temperature_at_depth(&self, depth_km: f64) -> f64 {
        if depth_km <= 0.0 {
            // Atmospheric layers get surface temperature
            self.params.surface_temp_k
        } else {
            // Subsurface layers get geothermal gradient
            let temp = self.params.surface_temp_k + (depth_km * self.params.geothermal_gradient_k_per_km);
            // Cap at a reasonable mantle temperature (~2000K) to prevent unrealistic deep temperatures
            // Foundry layers can still be heated by radiance beyond this
            temp.min(2000.0)
        }
    }


    /// Enforce foundry temperature at layers explicitly marked as foundry
    fn enforce_foundry_temperature(&self, sim: &mut Simulation) {
        // Process cells sequentially (parallel processing has borrowing constraints with HashMap)
        for (_cell_id, cell) in sim.cells.iter_mut() {
            for layer_tuple in &mut cell.layers_t {
                // Only set temperature for layers explicitly marked as foundry
                if layer_tuple.0.is_foundry {
                    layer_tuple.0.energy_mass.set_temperature(self.params.core_temp_k);
                    layer_tuple.1.energy_mass.set_temperature(self.params.core_temp_k);
                }
            }
        }
    }
}

impl SimOp for SurfaceEnergyInitOp {
    fn name(&self) -> &str {
        "SurfaceEnergyInit"
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn init_sim(&mut self, sim: &mut Simulation) {

        let mut _total_energy_applied = 0.0;
        let mut _layers_initialized = 0;
        let mut _foundry_layers_count = 0;

        // Process cells sequentially for temperature initialization
        for (_cell_id, cell) in sim.cells.iter_mut() {
            for (_i, (current, next)) in cell.layers_t.iter_mut().enumerate() {
                // Calculate target temperature based on layer depth
                let layer_center_depth = current.start_depth_km + (current.height_km / 2.0);
                let target_temp_k = self.calculate_temperature_at_depth(layer_center_depth);

                // Set temperature for both current and next states
                current.energy_mass.set_temperature(target_temp_k);
                next.energy_mass.set_temperature(target_temp_k);

                // Track if this layer is at foundry temperature
                if (target_temp_k - self.params.core_temp_k).abs() < 1.0 {
                    _foundry_layers_count += 1;
                }

                _total_energy_applied += current.energy_j();
                _layers_initialized += 1;
            }
        }

        // Ensure foundry temperature is enforced at deepest layers
        // Temporarily disabled to allow radiance-driven temperatures
        // self.enforce_foundry_temperature(sim);

        // Surface energy initialization complete
    }

    fn update_sim(&mut self, sim: &mut Simulation) {
        // No longer enforce foundry temperature every step - let radiance drive temperatures
        // self.enforce_foundry_temperature(sim);
    }
}
