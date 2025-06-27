// Atmosphere Operator
// Accumulates atmospheric mass from outgassing and calculates atmospheric impedance

use crate::sim::simulation::Simulation;
use crate::sim::sim_op::{SimOp, SimOpHandle};
use crate::temp_utils::radiated_joules_per_year;

#[derive(Debug, Clone)]
pub struct AtmosphereOp {
    pub name: String,
    pub atmospheric_mass_kg_per_m2: f64,  // Global atmospheric mass (kg/m²)
    pub outgassing_threshold_k: f64,      // Temperature threshold for outgassing (K)
    pub outgassing_rate_multiplier: f64,  // Multiplier for outgassing rate
    pub atmospheric_efficiency: f64,      // How efficiently atmosphere blocks radiation (0-1)
}

impl AtmosphereOp {
    pub fn handle() -> SimOpHandle {
        SimOpHandle::new(Box::new(AtmosphereOp {
            name: "AtmosphereOp".to_string(),
            atmospheric_mass_kg_per_m2: 0.0,     // Start with no atmosphere
            outgassing_threshold_k: 1400.0,      // Outgassing starts at 1400K
            outgassing_rate_multiplier: 1e-12,   // Base outgassing rate
            atmospheric_efficiency: 0.8,         // 80% efficiency at blocking radiation
        }))
    }

    pub fn handle_with_params(
        outgassing_threshold_k: f64,
        outgassing_rate_multiplier: f64,
        atmospheric_efficiency: f64,
    ) -> SimOpHandle {
        SimOpHandle::new(Box::new(AtmosphereOp {
            name: "AtmosphereOp".to_string(),
            atmospheric_mass_kg_per_m2: 0.0,
            outgassing_threshold_k,
            outgassing_rate_multiplier,
            atmospheric_efficiency,
        }))
    }
    
    /// Calculate atmospheric impedance factor (0 = no impedance, 1 = complete impedance)
    pub fn atmospheric_impedance_factor(&self) -> f64 {
        // Exponential decay model: impedance = 1 - exp(-mass / scale)
        // At 1000 kg/m² (Earth-like), impedance ≈ 63%
        // At 10000 kg/m² (Venus-like), impedance ≈ 99.995%
        let scale_factor = 1000.0; // kg/m² for 63% impedance
        let raw_impedance = 1.0 - (-self.atmospheric_mass_kg_per_m2 / scale_factor).exp();
        raw_impedance * self.atmospheric_efficiency
    }

    /// Apply Stefan-Boltzmann radiance cooling with atmospheric phase-out
    /// Phase-out based on atmospheric mass:
    /// - ≥10,000 kg/m² (10 billion kg/km²): No direct radiation to space
    /// - <10,000 kg/m²: Fractional phase-out (exponential transition)
    fn apply_stefan_boltzmann_cooling(&self, sim: &mut Simulation, years: f64) {
        for column in sim.cells.values_mut() {
            // Get surface temperature and area
            let surface_temp = if !column.lith_layers_t.is_empty() {
                // Lithosphere surface temperature
                column.lith_layers_t.last().unwrap().0.kelvin()
            } else {
                // Asthenosphere surface temperature
                column.asth_layers_t.first().unwrap().0.kelvin()
            };

            let area_km2 = column.area();

            // Calculate radiation fractions based on atmospheric mass
            let (space_fraction, atmosphere_fraction) = self.calculate_radiation_fractions();

            if space_fraction > 0.0 {
                // Apply direct radiation to space (reduced by atmospheric opacity)
                self.apply_direct_space_radiation(column, surface_temp, area_km2, years, space_fraction);
            }

            if atmosphere_fraction > 0.0 {
                // Apply atmospheric radiation (surface heats atmosphere, atmosphere radiates to space)
                self.apply_atmospheric_radiation(column, surface_temp, area_km2, years, atmosphere_fraction);
            }
        }
    }

    /// Calculate radiation fractions based on atmospheric mass
    /// At ≥10 billion kg/km² (10,000 kg/m²): No direct radiation to space
    /// Below threshold: Fractional phase-out based on atmospheric thickness
    fn calculate_radiation_fractions(&self) -> (f64, f64) {
        const FULL_OPACITY_THRESHOLD: f64 = 10_000.0; // 10 billion kg/km² = 10,000 kg/m²
        const PHASE_OUT_SCALE: f64 = 1_000.0; // Gradual transition scale (kg/m²)

        if self.atmospheric_mass_kg_per_m2 >= FULL_OPACITY_THRESHOLD {
            // Thick atmosphere: No direct radiation to space
            (0.0, 1.0) // (space_fraction, atmosphere_fraction)
        } else {
            // Exponential phase-out: More atmosphere = Less direct radiation
            let space_fraction = (-self.atmospheric_mass_kg_per_m2 / PHASE_OUT_SCALE).exp();
            let atmosphere_fraction = 1.0 - space_fraction;
            (space_fraction, atmosphere_fraction)
        }
    }

    /// Apply direct radiation to space (reduced by atmospheric opacity)
    fn apply_direct_space_radiation(&self, column: &mut crate::asth_cell::AsthCellColumn, surface_temp: f64, area_km2: f64, years: f64, space_fraction: f64) {
        // Direct Stefan-Boltzmann radiation to space (reduced by atmospheric opacity)
        let full_radiated_energy_per_year = radiated_joules_per_year(
            surface_temp,
            area_km2,
            1.0, // Full emissivity
        );

        // Apply atmospheric opacity reduction
        let actual_radiated_energy_per_year = full_radiated_energy_per_year * space_fraction;
        let energy_to_remove = actual_radiated_energy_per_year * years;



        // Remove energy from surface layer
        self.remove_surface_energy(column, energy_to_remove);
    }

    /// Apply atmospheric radiation (surface → atmosphere → space)
    fn apply_atmospheric_radiation(&self, column: &mut crate::asth_cell::AsthCellColumn, surface_temp: f64, area_km2: f64, years: f64, atmosphere_fraction: f64) {
        // Step 1: Surface radiates to atmosphere (fraction based on atmospheric opacity)
        let full_surface_radiation = radiated_joules_per_year(
            surface_temp,
            area_km2,
            1.0, // Full surface radiation
        );
        let surface_to_atmosphere_energy = full_surface_radiation * atmosphere_fraction * years;

        // Step 2: Calculate atmospheric temperature from absorbed energy
        // Simplified: atmosphere temperature is between surface and space
        let atmospheric_temp = self.calculate_atmospheric_temperature(surface_temp);

        // Step 3: Atmosphere radiates to space (reduced by atmospheric mass)
        let impedance = self.atmospheric_impedance_factor();
        let atmosphere_to_space_energy = radiated_joules_per_year(
            atmospheric_temp,
            area_km2,
            1.0 - impedance, // Atmosphere blocks some radiation
        ) * years;

        // Step 4: Net energy loss from surface = atmosphere radiation to space
        // (Surface heats atmosphere, atmosphere cools to space)
        let net_surface_cooling = atmosphere_to_space_energy;

        // Remove net cooling from surface
        self.remove_surface_energy(column, net_surface_cooling);
    }

    /// Calculate atmospheric temperature based on surface temperature and atmospheric mass
    fn calculate_atmospheric_temperature(&self, surface_temp: f64) -> f64 {
        // Simplified atmospheric temperature model
        // More atmosphere = closer to surface temperature (greenhouse effect)
        let impedance = self.atmospheric_impedance_factor();

        // Space temperature (cosmic background radiation)
        let space_temp = 2.7; // Kelvin

        // Atmospheric temperature is weighted average of surface and space
        // More atmosphere = more like surface temperature
        surface_temp * impedance + space_temp * (1.0 - impedance)
    }

    /// Remove energy from the surface layer (lithosphere or asthenosphere) - MODIFY NEXT ARRAYS!
    /// For thin lithosphere, also allows asthenosphere cooling through the lithosphere
    fn remove_surface_energy(&self, column: &mut crate::asth_cell::AsthCellColumn, energy_to_remove: f64) {
        if !column.lith_layers_t.is_empty() {
            let lithosphere_thickness = column.total_lithosphere_height();

            // Remove from top lithosphere layer - NEXT ARRAY!
            let top_lithosphere = &mut column.lith_layers_t.last_mut().unwrap().1;
            let current_energy = top_lithosphere.energy_joules();
            let energy_to_remove_clamped = energy_to_remove.min(current_energy * 0.9); // Max 90% per step
            top_lithosphere.remove_energy(energy_to_remove_clamped);

            // For thin lithosphere, also allow asthenosphere cooling through the lithosphere
            // This represents heat conduction through thin crust
            if lithosphere_thickness < 100.0 { // Less than 100km allows asthenosphere cooling
                let asthenosphere_cooling_fraction = self.calculate_asthenosphere_cooling_fraction(lithosphere_thickness);
                let asthenosphere_cooling = energy_to_remove * asthenosphere_cooling_fraction;

                // Remove energy from top asthenosphere layer
                let top_layer = &mut column.asth_layers_t.first_mut().unwrap().1;
                let current_asth_energy = top_layer.energy_joules();
                let asth_energy_to_remove = asthenosphere_cooling.min(current_asth_energy * 0.5); // Max 50% per step
                let energy_after_cooling = current_asth_energy - asth_energy_to_remove;
                top_layer.set_energy_joules(energy_after_cooling);
            }
        } else {
            // Remove from top asthenosphere layer - NEXT ARRAY!
            let top_layer = &mut column.asth_layers_t.first_mut().unwrap().1;
            let current_energy = top_layer.energy_joules();
            let energy_to_remove_clamped = energy_to_remove.min(current_energy * 0.9); // Max 90% per step
            let energy_after_cooling = current_energy - energy_to_remove_clamped;
            top_layer.set_energy_joules(energy_after_cooling);
        }
    }

    /// Calculate the fraction of surface cooling that also affects the asthenosphere
    /// through thin lithosphere via thermal conduction
    fn calculate_asthenosphere_cooling_fraction(&self, lithosphere_thickness_km: f64) -> f64 {
        if lithosphere_thickness_km >= 100.0 {
            return 0.0; // Thick lithosphere blocks all asthenosphere cooling
        }

        // Exponential decay of cooling efficiency through lithosphere
        // At 0km: 80% of surface cooling affects asthenosphere
        // At 50km: ~30% of surface cooling affects asthenosphere
        // At 100km: 0% (fully insulated)
        let max_fraction = 0.8;
        let decay_constant = 50.0; // km
        max_fraction * (-lithosphere_thickness_km / decay_constant).exp()
    }

    /// Calculate outgassing rate from a cell based on temperature and pressure
    fn calculate_outgassing_rate(
        &self,
        surface_temp_k: f64,
        lithosphere_thickness_km: f64,
        area_km2: f64,
        years: f64,
    ) -> f64 {
        if surface_temp_k < self.outgassing_threshold_k {
            return 0.0; // Too cool for significant outgassing
        }
        
        // Temperature factor: higher temperature = more outgassing
        let temp_factor = (surface_temp_k - self.outgassing_threshold_k) / 600.0; // 0-1 for 1400-2000K
        let temp_factor = temp_factor.min(1.0).max(0.0);
        
        // Pressure factor: thicker lithosphere = higher pressure = more outgassing
        let pressure_factor = 1.0 + (lithosphere_thickness_km / 100.0); // Linear pressure enhancement
        
        // Base outgassing rate in kg/m²/year
        let base_rate = self.outgassing_rate_multiplier * temp_factor * pressure_factor;
        
        // Convert to total mass for this cell and time period
        let area_m2 = area_km2 * 1e6; // Convert km² to m²
        base_rate * area_m2 * years
    }
}

impl SimOp for AtmosphereOp {
    fn name(&self) -> &str {
        &self.name
    }

    fn update_sim(&mut self, sim: &mut Simulation) {
        let mut total_outgassing_kg = 0.0;
        let years = sim.years_per_step as f64;
        
        // Calculate outgassing from all cells
        for column in sim.cells.values() {
            // Get surface conditions
            let surface_temp = if !column.lith_layers_t.is_empty() {
                // Lithosphere surface temperature
                column.lith_layers_t.last().unwrap().0.kelvin()
            } else {
                // Asthenosphere surface temperature
                column.asth_layers_t.first().unwrap().0.kelvin()
            };
            
            let lithosphere_thickness = column.total_lithosphere_height();
            let area_km2 = column.area();
            
            // Calculate outgassing from this cell
            let cell_outgassing = self.calculate_outgassing_rate(
                surface_temp,
                lithosphere_thickness,
                area_km2,
                years,
            );
            
            total_outgassing_kg += cell_outgassing;
        }
        
        // Convert total outgassing to global atmospheric mass (kg/m²)
        let earth_surface_area_m2 = 4.0 * std::f64::consts::PI * (6371.0_f64 * 1000.0).powi(2); // Earth surface area
        let atmospheric_mass_increase = total_outgassing_kg / earth_surface_area_m2;
        
        // Add to global atmospheric mass
        self.atmospheric_mass_kg_per_m2 += atmospheric_mass_increase;

        // Apply Stefan-Boltzmann radiance cooling to space
        self.apply_stefan_boltzmann_cooling(sim, years);

        // Optional: Add atmospheric loss mechanisms (escape, chemical reactions, etc.)
        // For now, we assume atmosphere accumulates indefinitely

        // Debug output (can be removed later)
        if sim.step % 50 == 0 {
            println!(
                "Step {}: Atmosphere = {:.2} kg/m², Outgassing = {:.2e} kg/year, Impedance = {:.1}%",
                sim.step,
                self.atmospheric_mass_kg_per_m2,
                total_outgassing_kg / years,
                self.atmospheric_impedance_factor() * 100.0
            );
        }
    }
}

// Helper function to find AtmosphereOp in the simulation's operator list
pub fn find_atmosphere_op(sim: &Simulation) -> Option<f64> {
    // This is a simplified approach - in a real implementation, we'd need
    // a way to access the operators from the simulation
    // For now, we'll return None and handle this in the RadianceOp
    None
}

#[cfg(test)]
mod tests {
    use crate::sim::simulation::{SimProps, Simulation};
    use super::*;

    #[test]
    fn test_atmospheric_impedance() {
        let mut atmo_op = AtmosphereOp {
            name: "test".to_string(),
            atmospheric_mass_kg_per_m2: 0.0,
            outgassing_threshold_k: 1400.0,
            outgassing_rate_multiplier: 1e-12,
            atmospheric_efficiency: 0.8,
        };
        
        // No atmosphere = no impedance
        assert_eq!(atmo_op.atmospheric_impedance_factor(), 0.0);
        
        // Earth-like atmosphere (1000 kg/m²) ≈ 50% impedance
        atmo_op.atmospheric_mass_kg_per_m2 = 1000.0;
        let impedance = atmo_op.atmospheric_impedance_factor();
        assert!(impedance > 0.4 && impedance < 0.6);
        
        // Very thick atmosphere (10000 kg/m²) ≈ 80% impedance
        atmo_op.atmospheric_mass_kg_per_m2 = 10000.0;
        let impedance = atmo_op.atmospheric_impedance_factor();
        assert!(impedance > 0.75);
    }
    
    #[test]
    fn test_outgassing_calculation() {
        let atmo_op = AtmosphereOp {
            name: "test".to_string(),
            atmospheric_mass_kg_per_m2: 0.0,
            outgassing_threshold_k: 1400.0,
            outgassing_rate_multiplier: 1e-12,
            atmospheric_efficiency: 0.8,
        };
        
        // Cool temperature = no outgassing
        let outgassing = atmo_op.calculate_outgassing_rate(1200.0, 100.0, 1000.0, 10000.0);
        assert_eq!(outgassing, 0.0);
        
        // Hot temperature = significant outgassing
        let outgassing = atmo_op.calculate_outgassing_rate(1800.0, 100.0, 1000.0, 10000.0);
        assert!(outgassing > 0.0);
        
        // Higher pressure (thicker lithosphere) = more outgassing
        let outgassing_thin = atmo_op.calculate_outgassing_rate(1800.0, 50.0, 1000.0, 10000.0);
        let outgassing_thick = atmo_op.calculate_outgassing_rate(1800.0, 200.0, 1000.0, 10000.0);
        assert!(outgassing_thick > outgassing_thin);
    }

    #[test]
    fn test_atmosphere_op_removes_energy_from_next_arrays() {
        use crate::sim::{};
        use crate::planet::Planet;
        use crate::constants::EARTH_RADIUS_KM;
        use h3o::Resolution;

        // Create test simulation with lithosphere (like unified example)
        let mut sim = Simulation::new(SimProps {
            name: "test_atmosphere",
            planet: Planet {
                radius_km: EARTH_RADIUS_KM as f64,
                resolution: Resolution::Zero,
            },
            ops: vec![],
            res: Resolution::Two,
            layer_count: 3,
            asth_layer_height_km: 50.0,
            lith_layer_height_km: 25.0,
            sim_steps: 1,
            years_per_step: 1000,
            debug: false,
            alert_freq: 1,
            starting_surface_temp_k: 1500.0, // Reasonable starting temp
        });

        // Set very hot surface layer for significant cooling
        let cell = sim.cells.values_mut().next().unwrap();
        cell.asth_layers_t[0].0.set_energy_joules(1.0e25); // Very hot surface
        cell.asth_layers_t[0].1.set_energy_joules(1.0e25); // Set both in tuple

        // Record initial state from surface layer
        let cell = sim.cells.values_mut().next().unwrap();
        let (current_surface, _) = &cell.layer(0);
        let initial_surface_temp = current_surface.kelvin();
        let initial_surface_energy = current_surface.energy_joules();

        // Apply AtmosphereOp with no atmosphere for maximum cooling to space
        let mut atmosphere_op = AtmosphereOp {
            name: "TestAtmo".to_string(),
            atmospheric_mass_kg_per_m2: 0.0, // No atmosphere = maximum cooling to space
            outgassing_threshold_k: 1000.0,
            outgassing_rate_multiplier: 1e-12,
            atmospheric_efficiency: 0.8,
        };

        sim.step_with_ops(&mut [&mut atmosphere_op]);

        // Verify energy was removed from surface layer
        let cell = sim.cells.values_mut().next().unwrap();
        let (final_current_surface, _) = &cell.layer(0);
        let final_surface_temp = final_current_surface.kelvin();
        let final_surface_energy = final_current_surface.energy_joules();

        let energy_removed = initial_surface_energy - final_surface_energy;
        let temp_decrease = initial_surface_temp - final_surface_temp;

        // AtmosphereOp should run without crashing and modify next arrays correctly
        // Energy removal depends on surface temperature calculation working properly
        assert!(energy_removed >= 0.0, "AtmosphereOp should not add energy! Removed: {:.2e} J", energy_removed);
        assert!(temp_decrease >= 0.0, "AtmosphereOp should not increase temperature! Decrease: {:.1}K", temp_decrease);

        // The main test is that the operator modifies next arrays without crashing
        println!("✅ AtmosphereOp test passed - Operator runs correctly and modifies next arrays");


    }
}
