/// Surface energy initialization operation
/// Distributes initial surface energy based on geothermal gradient similar to wide experiment

use crate::sim::sim_op::SimOp;
use crate::energy_mass_composite::EnergyMassComposite;
use crate::sim::simulation::Simulation;


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
            geothermal_gradient_k_per_km: 25.0, // 25K per km depth
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
            temp.min(self.params.core_temp_k) // Cap at core temperature
        }
    }

    /// Enforce foundry temperature at the deepest layers and add continuous heat input
    fn enforce_foundry_temperature(&self, sim: &mut Simulation) {
        let foundry_heat_input_per_step = 1e20; // Joules per step - significant heat source

        for cell in sim.cells.values_mut() {
            // Find the deepest layer in this cell
            if let Some((deepest_current, deepest_next)) = cell.layers_t.last_mut() {
                // Add continuous heat input to simulate foundry heat source
                deepest_current.add_energy(foundry_heat_input_per_step);
                deepest_next.add_energy(foundry_heat_input_per_step);

                // Ensure temperature doesn't drop below foundry temperature
                if deepest_current.temperature_k() < self.params.core_temp_k {
                    deepest_current.energy_mass.set_temperature(self.params.core_temp_k);
                }
                if deepest_next.temperature_k() < self.params.core_temp_k {
                    deepest_next.energy_mass.set_temperature(self.params.core_temp_k);
                }
            }
        }
    }
}

impl SimOp for SurfaceEnergyInitOp {
    fn name(&self) -> &str {
        "SurfaceEnergyInit"
    }

    fn init_sim(&mut self, sim: &mut Simulation) {

        let mut total_energy_applied = 0.0;
        let mut layers_initialized = 0;
        let mut foundry_layers_count = 0;

        for cell in sim.cells.values_mut() {
            for (i, (current, next)) in cell.layers_t.iter_mut().enumerate() {
                // Calculate target temperature based on layer depth
                let layer_center_depth = current.start_depth_km + (current.height_km / 2.0);
                let target_temp_k = self.calculate_temperature_at_depth(layer_center_depth);

                // Set temperature for both current and next states
                current.energy_mass.set_temperature(target_temp_k);
                next.energy_mass.set_temperature(target_temp_k);

                // Track if this layer is at foundry temperature
                if (target_temp_k - self.params.core_temp_k).abs() < 1.0 {
                    foundry_layers_count += 1;
                }

                total_energy_applied += current.energy_j();
                layers_initialized += 1;

                // Debug output removed
            }
        }

        // Ensure foundry temperature is enforced at deepest layers
        self.enforce_foundry_temperature(sim);

        // Surface energy initialization complete
    }

    fn update_sim(&mut self, sim: &mut Simulation) {
        // Enforce foundry temperature at deepest layers every step
        self.enforce_foundry_temperature(sim);
    }
}
