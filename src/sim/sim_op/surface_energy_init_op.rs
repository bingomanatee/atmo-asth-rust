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
    pub foundry_oscillation_enabled: bool,
    pub foundry_oscillation_period_years: f64,
    pub foundry_min_multiplier: f64,
    pub foundry_max_multiplier: f64,
}

impl Default for SurfaceEnergyInitParams {
    fn default() -> Self {
        Self {
            surface_temp_k: 280.0,           // 280K surface temperature
            geothermal_gradient_k_per_km: 25.0, // 25K per km depth
            core_temp_k: 1800.0,             // 1800K core temperature
            foundry_oscillation_enabled: true,  // Enable foundry temperature oscillation
            foundry_oscillation_period_years: 500.0, // 500-year period
            foundry_min_multiplier: 0.25,    // 25% minimum
            foundry_max_multiplier: 1.75,    // 175% maximum
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
            foundry_oscillation_enabled: true,
            foundry_oscillation_period_years: 500.0,
            foundry_min_multiplier: 0.25,
            foundry_max_multiplier: 1.75,
        }
    }

    pub fn with_foundry_oscillation(
        surface_temp_k: f64,
        geothermal_gradient_k_per_km: f64,
        core_temp_k: f64,
        oscillation_enabled: bool,
        period_years: f64,
        min_multiplier: f64,
        max_multiplier: f64,
    ) -> Self {
        Self {
            surface_temp_k,
            geothermal_gradient_k_per_km,
            core_temp_k,
            foundry_oscillation_enabled: oscillation_enabled,
            foundry_oscillation_period_years: period_years,
            foundry_min_multiplier: min_multiplier,
            foundry_max_multiplier: max_multiplier,
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

    /// Generate pseudo-random offset based on cell ID
    pub fn generate_cell_phase_offset(&self, cell_id: u64) -> f64 {
        // Use a simple hash function to generate pseudo-random phase offset
        let mut hash = cell_id;
        hash = hash.wrapping_mul(0x9e3779b97f4a7c15); // Large prime multiplier
        hash = hash ^ (hash >> 30);
        hash = hash.wrapping_mul(0xbf58476d1ce4e5b9); // Another large prime
        hash = hash ^ (hash >> 27);

        // Convert to 0-1 range and scale to 0-2π for phase offset
        let normalized = (hash as f64) / (u64::MAX as f64);
        normalized * 2.0 * std::f64::consts::PI
    }

    /// Calculate oscillating foundry temperature multiplier
    pub fn calculate_foundry_multiplier(&self, current_time_years: f64, cell_id: u64) -> f64 {
        if !self.params.foundry_oscillation_enabled {
            return 1.0; // No oscillation, use base temperature
        }

        let phase_offset = self.generate_cell_phase_offset(cell_id);
        let angular_frequency = 2.0 * std::f64::consts::PI / self.params.foundry_oscillation_period_years;
        let phase = angular_frequency * current_time_years + phase_offset;

        // Sin function oscillates between -1 and 1, map to min_multiplier and max_multiplier
        let sin_value = phase.sin();
        let mid_point = (self.params.foundry_min_multiplier + self.params.foundry_max_multiplier) / 2.0;
        let amplitude = (self.params.foundry_max_multiplier - self.params.foundry_min_multiplier) / 2.0;

        mid_point + amplitude * sin_value
    }

    /// Enforce foundry temperature at the deepest layers with oscillating heat input
    fn enforce_foundry_temperature(&self, sim: &mut Simulation) {
        let base_foundry_heat_input_per_step = 1e20; // Base Joules per step
        let current_time_years = sim.step as f64 * sim.years_per_step as f64;

        for (cell_id, cell) in sim.cells.iter_mut() {
            // Calculate oscillating multiplier for this cell
            let cell_id_u64 = u64::from(*cell_id); // Convert H3Index to u64
            let multiplier = self.calculate_foundry_multiplier(current_time_years, cell_id_u64);
            let oscillating_heat_input = base_foundry_heat_input_per_step * multiplier;
            let oscillating_foundry_temp = self.params.core_temp_k * multiplier;

            // Find the deepest layer in this cell
            if let Some((deepest_current, deepest_next)) = cell.layers_t.last_mut() {
                // Add oscillating heat input to simulate variable foundry heat source
                deepest_current.add_energy(oscillating_heat_input);
                deepest_next.add_energy(oscillating_heat_input);

                // Ensure temperature matches oscillating foundry temperature
                deepest_current.energy_mass.set_temperature(oscillating_foundry_temp);
                deepest_next.energy_mass.set_temperature(oscillating_foundry_temp);
            }
        }
    }
}

impl SimOp for SurfaceEnergyInitOp {
    fn name(&self) -> &str {
        "SurfaceEnergyInit"
    }

    fn init_sim(&mut self, sim: &mut Simulation) {

        let mut _total_energy_applied = 0.0;
        let mut _layers_initialized = 0;
        let mut _foundry_layers_count = 0;

        for cell in sim.cells.values_mut() {
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
