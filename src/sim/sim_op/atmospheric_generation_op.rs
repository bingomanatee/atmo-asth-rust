/// Atmospheric generation operation
/// Generates atmosphere specifically from MELTING LITHOSPHERE that produces gas.
/// When lithosphere layers melt (transition to liquid/gas), they outgas volatile materials
/// that are added to atmospheric layers with exponentially decreasing density (88% reduction per layer)

use crate::sim::sim_op::SimOp;
use crate::sim::simulation::Simulation;
use crate::energy_mass_composite::{EnergyMassComposite, MaterialPhase, MaterialCompositeType};

pub struct AtmosphericGenerationOp {
    pub apply_during_simulation: bool,
    pub outgassing_rate_multiplier: f64,  // Multiplier for outgassing rate from melting
    pub volume_decay_factor: f64,         // Exponential decay factor for atmospheric volumes
    pub density_decay_factor: f64,        // Exponential decay factor for atmospheric density (0.12 = 88% reduction per layer)
    pub depth_attenuation_factor: f64,    // How much deeper sources contribute less (0.8 = 20% reduction per layer depth)
    pub crystallization_rate: f64,        // Percentage of rising gas that crystallizes per layer (0.1 = 10% per layer)
    pub debug_output: bool,

    // Tracking state
    total_outgassed_mass: f64,
    total_redistributed_volume: f64,
    total_crystallized_mass: f64,
    step_count: usize,
}

impl AtmosphericGenerationOp {
    pub fn new() -> Self {
        Self {
            apply_during_simulation: true,
            outgassing_rate_multiplier: 0.01, // 1% of melted material becomes atmospheric
            volume_decay_factor: 0.7,         // Each layer up has 70% of the volume below
            density_decay_factor: 0.12,       // Each layer up has 12% of the density below (88% reduction)
            depth_attenuation_factor: 0.8,    // Deeper sources contribute 80% as much (20% reduction per layer)
            crystallization_rate: 0.1,        // 10% of rising gas crystallizes per layer
            debug_output: false,
            total_outgassed_mass: 0.0,
            total_redistributed_volume: 0.0,
            total_crystallized_mass: 0.0,
            step_count: 0,
        }
    }

    pub fn with_params(outgassing_rate: f64, volume_decay: f64, debug: bool) -> Self {
        Self {
            apply_during_simulation: true,
            outgassing_rate_multiplier: outgassing_rate,
            volume_decay_factor: volume_decay,
            density_decay_factor: 0.12,       // Each layer up has 12% of the density below (88% reduction)
            depth_attenuation_factor: 0.8,    // Deeper sources contribute 80% as much (20% reduction per layer)
            crystallization_rate: 0.1,        // 10% of rising gas crystallizes per layer
            debug_output: debug,
            total_outgassed_mass: 0.0,
            total_redistributed_volume: 0.0,
            total_crystallized_mass: 0.0,
            step_count: 0,
        }
    }

    pub fn with_full_params(outgassing_rate: f64, volume_decay: f64, density_decay: f64, debug: bool) -> Self {
        Self {
            apply_during_simulation: true,
            outgassing_rate_multiplier: outgassing_rate,
            volume_decay_factor: volume_decay,
            density_decay_factor: density_decay,
            depth_attenuation_factor: 0.8,    // Deeper sources contribute 80% as much (20% reduction per layer)
            crystallization_rate: 0.1,        // 10% of rising gas crystallizes per layer
            debug_output: debug,
            total_outgassed_mass: 0.0,
            total_redistributed_volume: 0.0,
            total_crystallized_mass: 0.0,
            step_count: 0,
        }
    }

    pub fn with_crystallization_params(outgassing_rate: f64, volume_decay: f64, density_decay: f64,
                                      depth_attenuation: f64, crystallization_rate: f64, debug: bool) -> Self {
        Self {
            apply_during_simulation: true,
            outgassing_rate_multiplier: outgassing_rate,
            volume_decay_factor: volume_decay,
            density_decay_factor: density_decay,
            depth_attenuation_factor: depth_attenuation,
            crystallization_rate,
            debug_output: debug,
            total_outgassed_mass: 0.0,
            total_redistributed_volume: 0.0,
            total_crystallized_mass: 0.0,
            step_count: 0,
        }
    }
    
    /// Check for melting lithosphere layers and generate atmospheric gas from outgassing
    fn process_outgassing(&mut self, cell: &mut crate::global_thermal::global_h3_cell::GlobalH3Cell) -> f64 {
        let mut total_outgassed = 0.0;

        // Find atmospheric and lithosphere layer ranges
        let mut atmo_end_index = 0;
        let mut lith_start_index = 0;
        let mut lith_end_index = 0;

        for (i, (current, _)) in cell.layers_t.iter().enumerate() {
            if current.is_atmospheric() {
                atmo_end_index = i + 1;
            } else if lith_start_index == 0 {
                lith_start_index = i;
            }
        }

        // Find end of lithosphere (before asthenosphere)
        for i in lith_start_index..cell.layers_t.len() {
            let (current, _) = &cell.layers_t[i];
            if current.is_surface_layer ||
               (i > lith_start_index && current.start_depth_km < cell.layers_t[i-1].0.start_depth_km + 50.0) {
                lith_end_index = i + 1;
            } else {
                break;
            }
        }

        // Check lithosphere layers for melting that generates gas
        for i in lith_start_index..lith_end_index {
            let (current, _) = &mut cell.layers_t[i];

            // Only process outgassing when lithosphere melts to liquid or gas phase
            let current_phase = current.energy_mass.phase();
            let material_type = current.energy_mass.material_composite_type();

            // Atmospheric generation occurs when solid lithosphere melts
            if current_phase == MaterialPhase::Liquid || current_phase == MaterialPhase::Gas {
                let layer_mass = current.energy_mass.mass_kg();
                let layer_temp = current.energy_mass.kelvin();

                // Calculate outgassing rate based on how much above melting point
                let melting_point = crate::material_composite::get_melting_point_k(&material_type);
                let temp_excess = (layer_temp - melting_point).max(0.0);

                // More outgassing at higher temperatures above melting point
                let base_outgas_rate = self.calculate_outgassing_rate(&material_type);
                let temp_multiplier = 1.0 + (temp_excess / 500.0); // Extra outgassing per 500K above melting
                let effective_outgas_rate = base_outgas_rate * temp_multiplier;

                // Apply depth attenuation: deeper layers contribute less
                let depth_from_surface = i - atmo_end_index; // Layers below atmosphere
                let depth_attenuation = self.depth_attenuation_factor.powi(depth_from_surface as i32);

                let raw_outgassed_mass = layer_mass * effective_outgas_rate * self.outgassing_rate_multiplier;
                let depth_attenuated_mass = raw_outgassed_mass * depth_attenuation;

                if depth_attenuated_mass > 0.0 {
                    total_outgassed += depth_attenuated_mass;

                    if self.debug_output {
                        println!("Lithosphere layer {} melting -> outgassing: {:.2e} kg (depth attenuation: {:.3}, phase: {:?}, temp: {:.0}K)",
                                i, depth_attenuated_mass, depth_attenuation, current_phase, layer_temp);
                    }
                }
            }
        }

        // Add outgassed material to atmospheric layers if any was generated
        if total_outgassed > 0.0 && atmo_end_index > 0 {
            self.add_atmospheric_material(cell, total_outgassed, atmo_end_index);
        }

        total_outgassed
    }
    
    /// Calculate outgassing rate based on material type
    fn calculate_outgassing_rate(&self, material_type: &MaterialCompositeType) -> f64 {
        match material_type {
            MaterialCompositeType::Silicate => 0.005,   // 0.5% outgassing rate
            MaterialCompositeType::Basaltic => 0.008,   // 0.8% outgassing rate (more volatile)
            MaterialCompositeType::Granitic => 0.012,   // 1.2% outgassing rate (most volatile)
            MaterialCompositeType::Metallic => 0.001,   // 0.1% outgassing rate (least volatile)
            _ => 0.003, // Default rate
        }
    }
    
    /// Add outgassed material to atmospheric layers with crystallization losses
    fn add_atmospheric_material(&mut self, cell: &mut crate::global_thermal::global_h3_cell::GlobalH3Cell,
                               total_mass: f64, atmo_layer_count: usize) {
        if atmo_layer_count == 0 {
            return;
        }

        // Start with total outgassed mass and apply crystallization as it rises through layers
        let mut remaining_mass = total_mass;
        let mut total_crystallized = 0.0;

        // Process from bottom atmospheric layer (closest to surface) upward
        for i in (0..atmo_layer_count).rev() {
            if remaining_mass <= 0.0 {
                break;
            }

            // Calculate crystallization loss for this layer
            let crystallized_mass = remaining_mass * self.crystallization_rate;
            remaining_mass -= crystallized_mass;
            total_crystallized += crystallized_mass;

            // Add remaining mass to this layer
            let mass_to_add = remaining_mass / (i + 1) as f64; // Distribute remaining among this and layers above

            let (current, _) = &mut cell.layers_t[i];

            // Add mass by increasing volume (atmospheric material is low density)
            let air_density = current.energy_mass.density_kgm3();
            if air_density > 0.0 && mass_to_add > 0.0 {
                let additional_volume_m3 = mass_to_add / air_density;
                let additional_volume_km3 = additional_volume_m3 / 1e9;

                // Scale the layer to include new volume
                let current_volume = current.energy_mass.volume();
                if current_volume > 0.0 {
                    let new_volume = current_volume + additional_volume_km3;
                    let scale_factor = new_volume / current_volume;

                    current.energy_mass.scale(scale_factor);

                    if self.debug_output {
                        println!("Layer {} added: {:.2e} kg, crystallized: {:.2e} kg, volume: {:.2} -> {:.2} km³",
                                i, mass_to_add, crystallized_mass, current_volume, new_volume);
                    }
                }
            }
        }

        self.total_crystallized_mass += total_crystallized;

        if self.debug_output && total_crystallized > 0.0 {
            println!("Total crystallization loss: {:.2e} kg ({:.1}% of outgassed material)",
                    total_crystallized, (total_crystallized / total_mass) * 100.0);
        }
    }
    
    /// Redistribute atmospheric material with exponential density and volume decay
    fn redistribute_atmosphere(&mut self, cell: &mut crate::global_thermal::global_h3_cell::GlobalH3Cell) -> f64 {
        let mut total_redistributed = 0.0;

        // Find atmospheric layers
        let mut atmo_layers: Vec<usize> = Vec::new();
        for (i, (current, _)) in cell.layers_t.iter().enumerate() {
            if current.is_atmospheric() {
                atmo_layers.push(i);
            } else {
                break; // Atmospheric layers are at the top
            }
        }

        if atmo_layers.len() < 2 {
            return 0.0; // Need at least 2 layers for redistribution
        }

        // Calculate total atmospheric mass
        let mut total_atmo_mass = 0.0;
        for &i in &atmo_layers {
            total_atmo_mass += cell.layers_t[i].0.energy_mass.mass_kg();
        }

        if total_atmo_mass <= 0.0 {
            return 0.0;
        }

        // Calculate target densities with exponential decay (bottom layer = highest density)
        // Each layer up has density_decay_factor (e.g., 0.12 = 12%) of the layer below
        let bottom_layer_idx = atmo_layers[atmo_layers.len()-1];
        let base_density = cell.layers_t[bottom_layer_idx].0.energy_mass.density_kgm3();

        let mut target_densities = Vec::new();
        let mut target_masses = Vec::new();
        let mut _total_target_mass = 0.0;

        for i in 0..atmo_layers.len() {
            // Higher layers (lower index) have exponentially lower density
            let layer_from_bottom = atmo_layers.len() - 1 - i;
            let target_density = base_density * self.density_decay_factor.powi(layer_from_bottom as i32);
            target_densities.push(target_density);

            // Calculate target mass based on layer volume and target density
            let layer_idx = atmo_layers[i];
            let layer_volume_m3 = cell.layers_t[layer_idx].0.energy_mass.volume() * 1e9; // km³ to m³
            let target_mass = target_density * layer_volume_m3;
            target_masses.push(target_mass);
            _total_target_mass += target_mass;
        }

        // Calculate base volume for volume decay before mutable borrows
        let base_volume = cell.layers_t[atmo_layers[atmo_layers.len()-1]].0.energy_mass.volume();

        // Redistribute mass according to exponential density profile
        for (idx, &layer_i) in atmo_layers.iter().enumerate() {
            let target_mass = target_masses[idx];
            let target_density = target_densities[idx];

            let (current, _) = &mut cell.layers_t[layer_i];
            let current_mass = current.energy_mass.mass_kg();
            let current_density = current.energy_mass.density_kgm3();

            if current_mass > 0.0 && target_mass > 0.0 {
                let mass_ratio = target_mass / current_mass;
                current.energy_mass.scale(mass_ratio);
                total_redistributed += (target_mass - current_mass).abs();

                if self.debug_output {
                    println!("Layer {} density: {:.3} -> {:.3} kg/m³ (mass: {:.2e} -> {:.2e} kg)",
                            layer_i, current_density, target_density, current_mass, target_mass);
                }

                // Remove excess volume that cannot be transported upward
                // Use exponential volume decay as well
                let layer_from_bottom = atmo_layers.len() - 1 - idx;
                let target_volume = base_volume * self.volume_decay_factor.powi(layer_from_bottom as i32);

                let new_volume = current.energy_mass.volume();
                if new_volume > target_volume * 1.1 { // Allow 10% tolerance
                    let volume_ratio = target_volume / new_volume;
                    current.energy_mass.scale(volume_ratio);

                    if self.debug_output {
                        println!("Removed excess volume from layer {}: {:.2} -> {:.2} km³",
                                layer_i, new_volume, target_volume);
                    }
                }
            }
        }

        total_redistributed
    }

    /// Globally redistribute atmospheric material equally across all H3 cells
    fn globally_redistribute_atmosphere(&mut self, sim: &mut Simulation) -> f64 {
        let mut total_redistributed = 0.0;

        if sim.cells.is_empty() {
            return 0.0;
        }

        // Step 1: Calculate total atmospheric mass and volumes across all cells
        let mut global_atmospheric_masses: Vec<f64> = Vec::new(); // Mass per atmospheric layer globally
        let mut cell_atmospheric_layers: Vec<Vec<usize>> = Vec::new(); // Atmospheric layer indices per cell
        let mut max_atmo_layers = 0;

        // Find atmospheric layers in each cell and calculate totals
        for cell in sim.cells.values() {
            let mut atmo_layers: Vec<usize> = Vec::new();
            for (i, (current, _)) in cell.layers_t.iter().enumerate() {
                if current.is_atmospheric() {
                    atmo_layers.push(i);
                } else {
                    break; // Atmospheric layers are at the top
                }
            }

            max_atmo_layers = max_atmo_layers.max(atmo_layers.len());
            cell_atmospheric_layers.push(atmo_layers);
        }

        // Initialize global mass totals for each atmospheric layer
        global_atmospheric_masses.resize(max_atmo_layers, 0.0);

        // Sum up total mass in each atmospheric layer across all cells
        for cell in sim.cells.values() {
            for (i, (current, _)) in cell.layers_t.iter().enumerate() {
                if current.is_atmospheric() && i < max_atmo_layers {
                    global_atmospheric_masses[i] += current.energy_mass.mass_kg();
                }
            }
        }

        // Step 2: Calculate target masses per cell with exponential density decay
        let cell_count = sim.cells.len() as f64;
        let mut target_masses_per_layer: Vec<f64> = Vec::new();

        for layer_idx in 0..max_atmo_layers {
            let total_mass_this_layer = global_atmospheric_masses[layer_idx];
            let target_mass_per_cell = total_mass_this_layer / cell_count;
            target_masses_per_layer.push(target_mass_per_cell);
        }

        // Step 3: Apply exponential density decay within each cell's atmospheric layers
        let mut cell_index = 0;
        for cell in sim.cells.values_mut() {
            if cell_index >= cell_atmospheric_layers.len() {
                break;
            }

            let atmo_layers = &cell_atmospheric_layers[cell_index];
            let atmo_layer_count = atmo_layers.len();
            cell_index += 1;

            if atmo_layer_count == 0 {
                continue;
            }

            // Calculate total target mass for this cell
            let mut total_target_mass_this_cell = 0.0;
            for layer_idx in 0..atmo_layer_count.min(max_atmo_layers) {
                total_target_mass_this_cell += target_masses_per_layer[layer_idx];
            }

            if total_target_mass_this_cell <= 0.0 {
                continue;
            }

            // Apply exponential density decay within this cell
            let mut density_weights: Vec<f64> = Vec::new();
            let mut total_density_weight = 0.0;

            for i in 0..atmo_layer_count {
                // Higher layers (lower index) have exponentially lower density
                let layer_from_bottom = atmo_layer_count - 1 - i;
                let density_weight = self.density_decay_factor.powi(layer_from_bottom as i32);
                density_weights.push(density_weight);
                total_density_weight += density_weight;
            }

            // Redistribute mass according to density profile
            for (idx, &layer_i) in atmo_layers.iter().enumerate() {
                if idx >= density_weights.len() {
                    break;
                }

                let density_weight = density_weights[idx];
                let target_mass = total_target_mass_this_cell * (density_weight / total_density_weight);

                // Get base volume before borrowing mutably
                let base_volume = if atmo_layer_count > 0 {
                    cell.layers_t[atmo_layers[atmo_layer_count-1]].0.energy_mass.volume()
                } else {
                    cell.layers_t[layer_i].0.energy_mass.volume()
                };

                let (current, _) = &mut cell.layers_t[layer_i];
                let current_mass = current.energy_mass.mass_kg();

                if current_mass > 0.0 && target_mass > 0.0 {
                    let mass_ratio = target_mass / current_mass;
                    current.energy_mass.scale(mass_ratio);
                    total_redistributed += (target_mass - current_mass).abs();

                    // Apply volume constraints
                    let layer_from_bottom = atmo_layer_count - 1 - idx;
                    let target_volume = base_volume * self.volume_decay_factor.powi(layer_from_bottom as i32);

                    let new_volume = current.energy_mass.volume();
                    if new_volume > target_volume * 1.1 { // Allow 10% tolerance
                        let volume_ratio = target_volume / new_volume;
                        current.energy_mass.scale(volume_ratio);

                        if self.debug_output && self.step_count % 500 == 0 {
                            println!("Cell layer {} volume constrained: {:.2} -> {:.2} km³",
                                    layer_i, new_volume, target_volume);
                        }
                    }
                }
            }
        }

        if self.debug_output && self.step_count % 500 == 0 {
            println!("Global atmospheric redistribution: {:.2e} kg across {} cells",
                    total_redistributed, sim.cells.len());
            for (i, &mass) in global_atmospheric_masses.iter().enumerate() {
                if mass > 0.0 {
                    println!("  Layer {}: {:.2e} kg total, {:.2e} kg per cell",
                            i, mass, mass / cell_count);
                }
            }
        }

        total_redistributed
    }
}

impl SimOp for AtmosphericGenerationOp {
    fn name(&self) -> &str {
        "AtmosphericGeneration"
    }

    fn init_sim(&mut self, _sim: &mut Simulation) {
        if self.debug_output {
            println!("AtmosphericGenerationOp initialized:");
            println!("  - Tracks melting lithosphere -> atmospheric gas generation");
            println!("  - Outgassing rate: {:.3}% of melted lithosphere mass", self.outgassing_rate_multiplier * 100.0);
            println!("  - Depth attenuation: {:.3} ({}% reduction per layer depth)",
                    self.depth_attenuation_factor, ((1.0 - self.depth_attenuation_factor) * 100.0) as i32);
            println!("  - Crystallization rate: {:.3} ({}% crystallizes per atmospheric layer)",
                    self.crystallization_rate, (self.crystallization_rate * 100.0) as i32);
            println!("  - Atmospheric density decay: {:.3} ({}% reduction per layer)",
                    self.density_decay_factor, ((1.0 - self.density_decay_factor) * 100.0) as i32);
            println!("  - Volume decay: {:.3} per layer", self.volume_decay_factor);
        }
    }

    fn update_sim(&mut self, sim: &mut Simulation) {
        if !self.apply_during_simulation {
            return;
        }

        self.step_count += 1;
        let mut step_outgassed = 0.0;

        // Step 1: Process outgassing from melting lithosphere in all cells
        for cell in sim.cells.values_mut() {
            let outgassed = self.process_outgassing(cell);
            step_outgassed += outgassed;
        }

        // Step 2: Globally redistribute all atmospheric material equally across all cells
        let step_redistributed = self.globally_redistribute_atmosphere(sim);

        self.total_outgassed_mass += step_outgassed;
        self.total_redistributed_volume += step_redistributed;

        if self.debug_output && self.step_count % 100 == 0 {
            println!("Step {}: Outgassed {:.2e} kg, Globally redistributed {:.2e} kg across {} cells",
                    self.step_count, step_outgassed, step_redistributed, sim.cells.len());
        }
    }

    fn after_sim(&mut self, _sim: &mut Simulation) {
        if self.debug_output {
            println!("AtmosphericGeneration complete:");
            println!("  - Total outgassed: {:.2e} kg", self.total_outgassed_mass);
            println!("  - Total crystallized: {:.2e} kg ({:.1}% loss)",
                    self.total_crystallized_mass,
                    if self.total_outgassed_mass > 0.0 {
                        (self.total_crystallized_mass / self.total_outgassed_mass) * 100.0
                    } else { 0.0 });
            println!("  - Net atmospheric addition: {:.2e} kg",
                    self.total_outgassed_mass - self.total_crystallized_mass);
            println!("  - Total redistributed: {:.2e} kg", self.total_redistributed_volume);
        }
    }
}
