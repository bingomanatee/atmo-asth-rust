/// Step-by-step melting event tracker with oscillating foundry
/// Now that simulation fields are public, we can track melting frequency and atmospheric mass changes at each step

use atmo_asth_rust::sim::simulation::{Simulation, SimProps};
use atmo_asth_rust::planet::Planet;
use atmo_asth_rust::global_thermal::global_h3_cell::{GlobalH3CellConfig, LayerConfig};
use atmo_asth_rust::energy_mass_composite::{MaterialCompositeType, EnergyMassComposite};
use atmo_asth_rust::material_composite::MaterialPhase;
use atmo_asth_rust::sim_op::{
    SurfaceEnergyInitOp,
    PressureAdjustmentOp,
    HeatRedistributionOp,
    TemperatureReportingOp,
    SpaceRadiationOp,
    AtmosphericGenerationOp,
};
use atmo_asth_rust::sim_op::surface_energy_init_op::SurfaceEnergyInitParams;
use atmo_asth_rust::sim_op::space_radiation_op::SpaceRadiationOpParams;
use atmo_asth_rust::sim_op::atmospheric_generation_op::CrystallizationParams;
use atmo_asth_rust::sim_op::SimOpHandle;
use h3o::Resolution;
use std::rc::Rc;

/// Structure to track melting events and atmospheric changes per step
#[derive(Debug, Clone)]
struct StepTracker {
    pub step: usize,
    pub time_years: f64,
    pub liquid_layers_count: usize,
    pub total_liquid_mass_kg: f64,
    pub atmospheric_mass_kg: f64,
    pub foundry_temp_range: (f64, f64), // (min, max)
    pub melting_depth_range: (f64, f64), // (shallowest, deepest)
    pub phase_transitions: usize, // New melting events this step
}

pub fn run_step_by_step_melting_tracker() {
    println!("🌋📊 Step-by-Step Melting Event Tracker with Oscillating Foundry");
    println!("{}", "=".repeat(75));
    println!("📊 Configuration:");
    println!("   - Base foundry temp: 1200K");
    println!("   - Oscillation range: 50% to 150% (600K to 1800K)");
    println!("   - Period: 500 years");
    println!("   - Enhanced atmospheric generation");
    println!("   - Step-by-step melting frequency tracking");
    println!("   - 20 steps × 1000 years = 20,000 years (40 oscillation cycles)");
    println!();

    // Create Earth planet with L2 resolution
    let planet = Planet::earth(Resolution::Two);

    // Create oscillating foundry parameters with reduced temperature range
    let surface_energy_params = SurfaceEnergyInitParams::with_foundry_oscillation(
        1200.0,  // surface_temp_k (reduced from 1500K)
        25.0,    // geothermal_gradient_k_per_km
        1200.0,  // core_temp_k (base foundry temperature, reduced from 1500K)
        true,    // oscillation_enabled
        500.0,   // period_years
        0.5,     // min_multiplier (50% - reduced from 25%)
        1.5,     // max_multiplier (150% - reduced from 175%)
    );

    // Enhanced atmospheric generation parameters with DEBUG ENABLED
    let crystallization_params = CrystallizationParams {
        outgassing_rate: 0.10,           // Increased to 10% for more outgassing
        volume_decay: 0.88,              // Standard atmospheric decay
        density_decay: 0.12,             // 88% reduction per layer
        depth_attenuation: 0.95,         // Increased to 95% contribution
        crystallization_rate: 0.02,      // Reduced to 2% for less loss
        debug: true,                     // ENABLE DEBUG OUTPUT!
    };

    // Create simulation properties
    let sim_props = SimProps {
        planet: planet.clone(),
        res: Resolution::Two,
        layer_count: 24,
        sim_steps: 20,
        years_per_step: 1000,
        name: "StepByStepMeltingTracker",
        debug: false,
        ops: vec![
            SimOpHandle::new(Box::new(SurfaceEnergyInitOp::new_with_params(surface_energy_params))),
            SimOpHandle::new(Box::new(PressureAdjustmentOp::new())),
            SimOpHandle::new(Box::new(SpaceRadiationOp::new(
                SpaceRadiationOpParams::with_reporting()
            ))),
            SimOpHandle::new(Box::new(HeatRedistributionOp::new())),
            SimOpHandle::new(Box::new(AtmosphericGenerationOp::with_crystallization_params(crystallization_params))),
            SimOpHandle::new(Box::new(TemperatureReportingOp::with_frequency(100.0))), // Only at end
        ],
    };

    // Create simulation
    let mut sim = Simulation::new(sim_props);
    
    // Configure cells
    let _planet_rc = Rc::new(planet);
    sim.make_cells(|cell_index, planet| {
        let layer_schedule = vec![
            LayerConfig {
                cell_type: MaterialCompositeType::Air,
                cell_count: 4,
                height_km: 20.0, // 80km total atmosphere
            },
            LayerConfig {
                cell_type: MaterialCompositeType::Silicate,
                cell_count: 10,
                height_km: 4.0, // 40km total lithosphere
            },
            LayerConfig {
                cell_type: MaterialCompositeType::Silicate,
                cell_count: 10,
                height_km: 8.0, // 80km total asthenosphere
            },
        ];
        
        GlobalH3CellConfig {
            h3_index: cell_index,
            planet,
            layer_schedule,
        }
    });

    // Track step-by-step changes
    let mut step_history = Vec::new();
    
    // Initialize simulation
    sim.step = 0;
    let mut ops = std::mem::take(&mut sim.ops);
    for op in &mut ops {
        op.init_sim(&mut sim);
    }
    sim.ops = ops;

    // Get initial state
    let initial_state = analyze_simulation_state(&sim, 0);
    step_history.push(initial_state.clone());
    
    println!("📊 Initial State:");
    print_step_analysis(&initial_state);

    // Run simulation step by step
    println!("\n🚀 Running step-by-step simulation...");
    
    for step in 0..sim.sim_steps {
        sim.step += 1;
        
        // Store previous state for comparison
        let prev_liquid_count = step_history.last().unwrap().liquid_layers_count;
        
        // Run simulation step
        let mut ops = std::mem::take(&mut sim.ops);
        for op in &mut ops {
            op.update_sim(&mut sim);
        }
        sim.ops = ops;
        
        // Advance to next state
        for cell in sim.cells.values_mut() {
            cell.commit_next_state();
        }
        
        // Analyze current state
        let current_state = analyze_simulation_state(&sim, (step + 1) as usize);
        let phase_transitions = if current_state.liquid_layers_count > prev_liquid_count {
            current_state.liquid_layers_count - prev_liquid_count
        } else {
            0
        };
        
        let mut current_state_with_transitions = current_state.clone();
        current_state_with_transitions.phase_transitions = phase_transitions;
        
        step_history.push(current_state_with_transitions.clone());
        
        // Report every 4 steps or when significant melting occurs
        if step % 4 == 0 || phase_transitions > 0 {
            println!("\n📊 Step {}: {:.0} years", step + 1, current_state_with_transitions.time_years);
            print_step_analysis(&current_state_with_transitions);
            
            if phase_transitions > 0 {
                println!("   🔥 NEW MELTING: {} layers transitioned to liquid!", phase_transitions);
            }
        }
    }

    // Final analysis
    analyze_melting_frequency_over_time(&step_history);
    analyze_atmospheric_correlation(&step_history);
    
    println!("\n🎉 Step-by-step melting tracker completed!");
}

fn analyze_simulation_state(sim: &Simulation, step: usize) -> StepTracker {
    let time_years = step as f64 * sim.years_per_step as f64;
    
    // Count liquid layers and calculate mass
    let mut liquid_count = 0;
    let mut total_liquid_mass = 0.0;
    let mut min_depth = f64::INFINITY;
    let mut max_depth = f64::NEG_INFINITY;
    
    for cell in sim.cells.values() {
        for (layer, _) in &cell.layers_t {
            if matches!(layer.phase(), MaterialPhase::Liquid) {
                liquid_count += 1;
                total_liquid_mass += layer.mass_kg();
                
                let depth = layer.start_depth_km + layer.height_km / 2.0;
                min_depth = min_depth.min(depth);
                max_depth = max_depth.max(depth);
            }
        }
    }
    
    if liquid_count == 0 {
        min_depth = 0.0;
        max_depth = 0.0;
    }
    
    // Calculate atmospheric mass
    let atmospheric_mass = calculate_total_atmospheric_mass(sim);
    
    // Calculate foundry temperature range
    let foundry_range = calculate_foundry_temperature_range(sim);
    
    StepTracker {
        step,
        time_years,
        liquid_layers_count: liquid_count,
        total_liquid_mass_kg: total_liquid_mass,
        atmospheric_mass_kg: atmospheric_mass,
        foundry_temp_range: foundry_range,
        melting_depth_range: (min_depth, max_depth),
        phase_transitions: 0, // Will be set by caller
    }
}

fn calculate_total_atmospheric_mass(sim: &Simulation) -> f64 {
    let mut total_mass = 0.0;
    
    for cell in sim.cells.values() {
        for i in 0..4.min(cell.layers_t.len()) {
            let (layer, _) = &cell.layers_t[i];
            if matches!(layer.energy_mass.material_composite_type(), MaterialCompositeType::Air) {
                total_mass += layer.mass_kg();
            }
        }
    }
    
    total_mass
}

fn calculate_foundry_temperature_range(sim: &Simulation) -> (f64, f64) {
    let mut min_temp = f64::INFINITY;
    let mut max_temp = f64::NEG_INFINITY;
    
    for cell in sim.cells.values() {
        if let Some((deepest_layer, _)) = cell.layers_t.last() {
            let temp = deepest_layer.temperature_k();
            min_temp = min_temp.min(temp);
            max_temp = max_temp.max(temp);
        }
    }
    
    if min_temp == f64::INFINITY {
        (0.0, 0.0)
    } else {
        (min_temp, max_temp)
    }
}

fn print_step_analysis(state: &StepTracker) {
    println!("   🔥 Liquid layers: {}", state.liquid_layers_count);
    println!("   🌬️  Atmospheric mass: {:.2e} kg", state.atmospheric_mass_kg);
    println!("   🌡️  Foundry range: {:.0}K - {:.0}K", 
             state.foundry_temp_range.0, state.foundry_temp_range.1);
    
    if state.melting_depth_range.1 > 0.0 {
        println!("   📏 Melting depth: {:.1}km - {:.1}km", 
                 state.melting_depth_range.0, state.melting_depth_range.1);
    }
}

fn analyze_melting_frequency_over_time(history: &[StepTracker]) {
    println!("\n🔥 MELTING FREQUENCY ANALYSIS:");
    
    let total_new_melting: usize = history.iter().map(|h| h.phase_transitions).sum();
    let max_liquid_layers = history.iter().map(|h| h.liquid_layers_count).max().unwrap_or(0);
    let final_liquid_layers = history.last().map(|h| h.liquid_layers_count).unwrap_or(0);
    
    println!("   Total new melting events: {}", total_new_melting);
    println!("   Peak liquid layers: {}", max_liquid_layers);
    println!("   Final liquid layers: {}", final_liquid_layers);
    
    // Find steps with significant melting
    let melting_steps: Vec<_> = history.iter()
        .filter(|h| h.phase_transitions > 0)
        .collect();
    
    if !melting_steps.is_empty() {
        println!("   Steps with new melting: {}", melting_steps.len());
        println!("   Major melting events:");
        for step in melting_steps.iter().take(5) {
            println!("     Year {:.0}: {} new liquid layers, foundry {:.0}K-{:.0}K", 
                     step.time_years, step.phase_transitions,
                     step.foundry_temp_range.0, step.foundry_temp_range.1);
        }
    }
}

fn analyze_atmospheric_correlation(history: &[StepTracker]) {
    println!("\n🌬️  ATMOSPHERIC-MELTING CORRELATION:");
    
    if history.len() < 2 {
        return;
    }
    
    let initial_atmo = history[0].atmospheric_mass_kg;
    let final_atmo = history.last().unwrap().atmospheric_mass_kg;
    let total_atmo_change = final_atmo - initial_atmo;
    
    println!("   Initial atmospheric mass: {:.2e} kg", initial_atmo);
    println!("   Final atmospheric mass: {:.2e} kg", final_atmo);
    println!("   Total atmospheric change: {:.2e} kg ({:+.3}%)", 
             total_atmo_change, (total_atmo_change / initial_atmo) * 100.0);
    
    // Find correlations between melting and atmospheric changes
    let mut positive_correlations = 0;
    let mut total_comparisons = 0;
    
    for i in 1..history.len() {
        let melting_change = history[i].liquid_layers_count as i32 - history[i-1].liquid_layers_count as i32;
        let atmo_change = history[i].atmospheric_mass_kg - history[i-1].atmospheric_mass_kg;
        
        if melting_change > 0 {
            total_comparisons += 1;
            if atmo_change > 0.0 {
                positive_correlations += 1;
            }
        }
    }
    
    if total_comparisons > 0 {
        let correlation_rate = (positive_correlations as f64 / total_comparisons as f64) * 100.0;
        println!("   Positive correlations (melting → atmosphere): {:.1}% ({}/{})", 
                 correlation_rate, positive_correlations, total_comparisons);
    }
}

fn main() {
    run_step_by_step_melting_tracker();
}
