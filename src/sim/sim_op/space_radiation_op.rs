use crate::sim::sim_op::SimOp;
use crate::energy_mass_composite::EnergyMassComposite;
use crate::global_thermal::global_h3_cell::GlobalH3Cell;
use crate::sim::simulation::Simulation;
use h3o::CellIndex;
use std::collections::HashMap;

/// Parameters for space radiation operation
#[derive(Debug, Clone)]
pub struct SpaceRadiationOpParams {
    /// Surface emissivity (0.0 to 1.0, typically 0.95 for rocky surfaces)
    pub emissivity: f64,
    /// Space temperature in Kelvin (typically ~2.7K cosmic background)
    pub space_temperature_k: f64,
    /// Density constant for opacity calculation (density * DENSITY_CONSTANT / height)
    pub density_constant: f64,
    /// Enable detailed reporting
    pub enable_reporting: bool,
}

impl Default for SpaceRadiationOpParams {
    fn default() -> Self {
        Self {
            emissivity: 0.95,
            space_temperature_k: 2.7, // Cosmic microwave background
            density_constant: 0.001, // Adjust this to control atmospheric opacity
            enable_reporting: false,
        }
    }
}

impl SpaceRadiationOpParams {
    /// Create parameters with custom emissivity
    pub fn with_emissivity(emissivity: f64) -> Self {
        Self {
            emissivity,
            ..Default::default()
        }
    }

    /// Create parameters with custom density constant for opacity calculation
    pub fn with_density_constant(density_constant: f64) -> Self {
        Self {
            density_constant,
            ..Default::default()
        }
    }

    /// Create parameters with reporting enabled
    pub fn with_reporting() -> Self {
        Self {
            enable_reporting: true,
            ..Default::default()
        }
    }
}

/// Space radiation operation for global thermal simulation
/// Radiates heat from surface layers to space using Stefan-Boltzmann law
pub struct SpaceRadiationOp {
    params: SpaceRadiationOpParams,
    total_radiated_energy: f64,
    cells_processed: usize,
    step_count: usize,
}

impl SpaceRadiationOp {
    pub fn new(params: SpaceRadiationOpParams) -> Self {
        Self {
            params,
            total_radiated_energy: 0.0,
            cells_processed: 0,
            step_count: 0,
        }
    }

    pub fn new_default() -> Self {
        Self::new(SpaceRadiationOpParams::default())
    }
    
    /// Apply space radiation to all cells in the simulation
    pub fn apply(&mut self, cells: &mut HashMap<CellIndex, GlobalH3Cell>, time_years: f64) {
        // Space radiation operation starting
        self.total_radiated_energy = 0.0;
        self.cells_processed = 0;

        for cell in cells.values_mut() {
            let radiated = self.apply_to_cell(cell, time_years);
            self.total_radiated_energy += radiated;
            self.cells_processed += 1;
        }
    }

    /// Apply space radiation to a single cell
    pub fn apply_to_cell(&self, cell: &mut GlobalH3Cell, time_years: f64) -> f64 {
        let mut total_radiated = 0.0;
        let surface_area_km2 = cell.surface_area_km2();

        // Find surface layer (first non-atmospheric layer)
        let mut surface_layer_index = None;
        for (i, (layer, _)) in cell.layers_t.iter().enumerate() {
            if !layer.is_atmospheric() {
                surface_layer_index = Some(i);
                break;
            }
        }

        if let Some(surface_index) = surface_layer_index {
            // Calculate simple opacity: sum of (density * DENSITY_CONSTANT / height) for layers 0 through surface
            let mut total_opacity: f64 = 0.0;

            for i in 0..surface_index {
                let (layer, _) = &cell.layers_t[i];
                let density = layer.energy_mass.density_kgm3();
                let height = layer.height_km;
                let layer_opacity = density * self.params.density_constant / height;
                total_opacity += layer_opacity;
            }

            // Minimum opacity of 1.0
            total_opacity = total_opacity.max(1.0);

            // Calculate transmission factor (0 to 1)
            let transmission_factor = (-total_opacity).exp();
            if transmission_factor > 0.0 {
                let effective_area_km2 = surface_area_km2 * transmission_factor;
                let radiated = self.radiate_layer_to_space(
                    &mut cell.layers_t[surface_index].0,
                    effective_area_km2,
                    time_years,
                );
                total_radiated += radiated;
            }
        }

        total_radiated
    }
    
    /// Apply Stefan-Boltzmann radiation to a single layer using density-based skin depth
    fn radiate_layer_to_space(
        &self,
        layer: &mut crate::global_thermal::thermal_layer::ThermalLayer,
        effective_area_km2: f64,
        time_years: f64,
    ) -> f64 {
        // Debug output removed

        // Use the proper density-based skin depth radiation method
        // This uses the improved radiation depth calculation based on material density
        let radiated = layer.energy_mass.radiate_to_space_with_skin_depth(
            effective_area_km2,
            time_years,
            1.0, // Full radiation factor (atmospheric attenuation handled separately)
        );

        // Debug output removed

        radiated
    }

    /// Report radiation results
    fn report_radiation_results(&self, _time_years: f64) {
        let _total_radiated = self.total_radiated_energy;
        let _cells_processed = self.cells_processed;
        let _avg_per_cell = self.average_radiated_per_cell();

        // Space radiation step completed
    }
    
    /// Get total energy radiated in the last operation
    pub fn total_radiated_energy(&self) -> f64 {
        self.total_radiated_energy
    }

    /// Get number of cells processed in the last operation
    pub fn cells_processed(&self) -> usize {
        self.cells_processed
    }

    /// Get average energy radiated per cell
    pub fn average_radiated_per_cell(&self) -> f64 {
        if self.cells_processed > 0 {
            self.total_radiated_energy / (self.cells_processed as f64)
        } else {
            0.0
        }
    }
}

impl SimOp for SpaceRadiationOp {
    fn name(&self) -> &str {
        "SpaceRadiation"
    }

    fn init_sim(&mut self, _sim: &mut Simulation) {
        // Space radiation initialized
    }

    fn update_sim(&mut self, sim: &mut Simulation) {
        self.step_count += 1;
        let time_years = sim.years_per_step as f64;

        // Apply radiation to all cells
        self.apply(&mut sim.cells, time_years);

        // Report if enabled
        if self.params.enable_reporting {
            self.report_radiation_results(time_years);
        }
    }
}

/// Convenience function to create and apply space radiation operation
pub fn apply_space_radiation(
    cells: &mut HashMap<CellIndex, GlobalH3Cell>,
    time_years: f64,
    params: Option<SpaceRadiationOpParams>,
) -> f64 {
    let mut op = SpaceRadiationOp::new(params.unwrap_or_default());
    op.apply(cells, time_years);
    op.total_radiated_energy()
}
