/// Enhanced global test with atmospheric generation and oscillating foundry
/// Uses the most up-to-date simple_global_test configuration with atmospheric dynamics

use atmo_asth_rust::sim::simulation::{Simulation, SimProps};
use atmo_asth_rust::planet::Planet;
use atmo_asth_rust::global_thermal::global_h3_cell::{GlobalH3CellConfig, LayerConfig};
use atmo_asth_rust::energy_mass_composite::{MaterialCompositeType, EnergyMassComposite};
use atmo_asth_rust::material_composite::MaterialPhase;
use atmo_asth_rust::sim::sim_op::{
    SurfaceEnergyInitOp,
    PressureAdjustmentOp,
    HeatRedistributionOp,
    TemperatureReportingOp,
    SpaceRadiationOp,
    AtmosphericGenerationOp,
};
use atmo_asth_rust::sim::sim_op::surface_energy_init_op::SurfaceEnergyInitParams;
use atmo_asth_rust::sim::sim_op::space_radiation_op::SpaceRadiationOpParams;
use atmo_asth_rust::sim::sim_op::atmospheric_generation_op::CrystallizationParams;
use atmo_asth_rust::sim::sim_op::SimOpHandle;
use h3o::Resolution;
use std::rc::Rc;


/// Structure to track melting events and atmospheric changes
#[derive(Debug, Clone)]
struct MeltingEventTracker {
    pub step: usize,
    pub time_years: f64,
    pub liquid_layers_count: usize,
    pub total_liquid_mass_kg: f64,
    pub atmospheric_mass_kg: f64,
    pub foundry_temp_range: (f64, f64), // (min, max)
    pub melting_depth_range: (f64, f64), // (shallowest, deepest)
}

/// Structure to track phase transitions per step
#[derive(Debug, Clone)]
struct PhaseTransitionStats {
    pub solid_to_liquid: usize,
    pub liquid_to_solid: usize,
    pub liquid_to_gas: usize,
    pub gas_to_liquid: usize,
}

impl PhaseTransitionStats {
    fn new() -> Self {
        Self {
            solid_to_liquid: 0,
            liquid_to_solid: 0,
            liquid_to_gas: 0,
            gas_to_liquid: 0,
        }
    }

    fn total_melting_events(&self) -> usize {
        self.solid_to_liquid + self.liquid_to_gas
    }

    fn total_solidifying_events(&self) -> usize {
        self.liquid_to_solid + self.gas_to_liquid
    }
}

pub fn run_enhanced_global_atmo_test() {
    println!("🌋🌍 Enhanced Global Atmospheric Test with Oscillating Foundry");
    println!("{}", "=".repeat(70));
    println!("📊 Configuration:");
    println!("   - Base foundry temp: 1500K (from simple_global_test)");
    println!("   - Oscillation range: 25% to 175% (375K to 2625K)");
    println!("   - Period: 500 years");
    println!("   - Atmospheric generation enabled");
    println!("   - Enhanced outgassing parameters");
    println!("   - 20 steps × 1000 years = 20,000 years (40 oscillation cycles)");
    println!();

    // Create Earth planet with L2 resolution (same as simple_global_test)
    let planet = Planet::earth(Resolution::Two);

    // Create oscillating foundry parameters using simple_global_test base temperature
    let surface_energy_params = SurfaceEnergyInitParams::with_foundry_oscillation(
        1500.0,  // surface_temp_k (from simple_global_test)
        25.0,    // geothermal_gradient_k_per_km
        1500.0,  // core_temp_k (base foundry temperature from simple_global_test)
        true,    // oscillation_enabled
        500.0,   // period_years
        0.25,    // min_multiplier (25%)
        1.75,    // max_multiplier (175%)
    );

    // Enhanced atmospheric generation parameters for better visibility
    let crystallization_params = CrystallizationParams {
        outgassing_rate: 0.05,           // Increased from 0.01 to 5% for more outgassing
        volume_decay: 0.88,              // Standard atmospheric decay
        density_decay: 0.12,             // 88% reduction per layer
        depth_attenuation: 0.9,          // Increased from 0.8 to 90% contribution
        crystallization_rate: 0.05,      // Reduced from 0.1 to 5% for less loss
        debug: false,
    };

    // Create simulation properties with all operations (based on simple_global_test)
    let sim_props = SimProps {
        planet: planet.clone(),
        res: Resolution::Two,
        layer_count: 24, // Will be overridden by GlobalH3Cell configuration
        sim_steps: 20,   // 20 steps for longer observation
        years_per_step: 1000, // 1000 years per step (same as simple_global_test)
        name: "EnhancedGlobalAtmoTest",
        debug: false,
        ops: vec![
            SimOpHandle::new(Box::new(SurfaceEnergyInitOp::new_with_params(surface_energy_params))),
            SimOpHandle::new(Box::new(PressureAdjustmentOp::new())),
            SimOpHandle::new(Box::new(SpaceRadiationOp::new(
                SpaceRadiationOpParams::with_reporting()
            ))),
            SimOpHandle::new(Box::new(HeatRedistributionOp::new())),
            SimOpHandle::new(Box::new(AtmosphericGenerationOp::with_crystallization_params(crystallization_params))),
            SimOpHandle::new(Box::new(TemperatureReportingOp::with_frequency(20.0))),
        ],
    };

    // Create simulation
    let mut sim = Simulation::new(sim_props);
    
    // Configure cells with standard Earth-like layout (same as simple_global_test)
    let _planet_rc = Rc::new(planet);
    sim.make_cells(|cell_index, planet| {
        // Standard cell layout: 4×20km atmosphere + 10×4km lithosphere + 10×8km asthenosphere
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

    // Track atmospheric mass changes and melting events over time
    let mut _melting_event_history: Vec<MeltingEventTracker> = Vec::new();
    let mut atmospheric_mass_history: Vec<(f64, f64)> = Vec::new();

    // Get initial atmospheric mass and melting state
    let initial_atmo_mass = calculate_total_atmospheric_mass(&sim);
    let initial_melting_stats = analyze_melting_state(&sim);

    atmospheric_mass_history.push((0.0, initial_atmo_mass));
    println!("🌬️  Initial total atmospheric mass: {:.2e} kg", initial_atmo_mass);
    println!("🔥 Initial liquid layers: {}", initial_melting_stats.0);

    // Track initial melting state and foundry temperatures
    println!("\n🚀 Running simulation with oscillating foundry...");

    let initial_foundry_range = analyze_foundry_temperature_range(&sim, 0.0);
    println!("📊 Initial foundry range: {:.0}K - {:.0}K", initial_foundry_range.0, initial_foundry_range.1);

    // Run the full simulation
    sim.run();

    // Analyze final results
    let final_atmo_mass = calculate_total_atmospheric_mass(&sim);
    let final_melting_stats = analyze_melting_state(&sim);
    let final_time_years = sim.sim_steps as f64 * sim.years_per_step as f64;
    let final_foundry_range = analyze_foundry_temperature_range(&sim, final_time_years);

    let mass_change = final_atmo_mass - initial_atmo_mass;
    let mass_change_percent = (mass_change / initial_atmo_mass) * 100.0;

    println!("\n📊 FINAL ANALYSIS:");
    println!("🌬️  Atmospheric Mass:");
    println!("   Initial: {:.2e} kg", initial_atmo_mass);
    println!("   Final: {:.2e} kg", final_atmo_mass);
    println!("   Change: {:.2e} kg ({:+.2}%)", mass_change, mass_change_percent);

    println!("\n🔥 Melting Analysis:");
    println!("   Initial liquid layers: {}", initial_melting_stats.0);
    println!("   Final liquid layers: {} (Δ{:+})",
             final_melting_stats.0,
             final_melting_stats.0 as i32 - initial_melting_stats.0 as i32);

    if final_melting_stats.2.1 > 0.0 {
        println!("   Final melting depth: {:.1}km - {:.1}km",
                 final_melting_stats.2.0, final_melting_stats.2.1);
    }

    println!("\n🌡️  Foundry Temperature Analysis:");
    println!("   Initial range: {:.0}K - {:.0}K", initial_foundry_range.0, initial_foundry_range.1);
    println!("   Final range: {:.0}K - {:.0}K", final_foundry_range.0, final_foundry_range.1);
    println!("   Temperature span: {:.0}K", final_foundry_range.1 - final_foundry_range.0);

    // Analyze foundry temperature effects across cells
    analyze_foundry_effects(&sim);

    // Determine if oscillating foundry increased melting frequency
    let melting_increase = final_melting_stats.0 as i32 - initial_melting_stats.0 as i32;
    if melting_increase > 0 {
        println!("✅ Oscillating foundry increased melting frequency by {} layers!", melting_increase);
    } else if melting_increase < 0 {
        println!("📉 Melting decreased by {} layers", melting_increase.abs());
    } else {
        println!("➡️  No change in liquid layer count");
    }

    if mass_change.abs() > initial_atmo_mass * 0.001 {
        println!("✅ Significant atmospheric mass change detected!");
    } else {
        println!("⚠️  Minimal atmospheric mass change - may need parameter tuning");
    }

    println!("\n🎉 Enhanced global atmospheric test completed!");
}

fn calculate_total_atmospheric_mass(sim: &Simulation) -> f64 {
    let mut total_mass = 0.0;
    
    for cell in sim.cells.values() {
        // Count atmospheric layers (first 4 layers are atmosphere in our config)
        for i in 0..4.min(cell.layers_t.len()) {
            let (layer, _) = &cell.layers_t[i];
            if matches!(layer.energy_mass.material_composite_type(), MaterialCompositeType::Air) {
                total_mass += layer.mass_kg();
            }
        }
    }
    
    total_mass
}

fn analyze_foundry_effects(sim: &Simulation) {
    println!("\n🔥 Foundry Temperature Analysis:");
    println!("Examining foundry temperatures across different cells:");
    println!("   Cell ID          Foundry Temp(K)  Multiplier  Phase");
    println!("   ---------------- --------------- ---------- -------");
    
    let final_time_years = sim.sim_steps as f64 * sim.years_per_step as f64;
    let surface_op = SurfaceEnergyInitOp::new_with_params(
        SurfaceEnergyInitParams::with_foundry_oscillation(
            1500.0, 25.0, 1500.0, true, 500.0, 0.25, 1.75
        )
    );
    
    let mut foundry_temps = Vec::new();
    let mut cell_count = 0;
    
    for (cell_id, cell) in sim.cells.iter() {
        if cell_count >= 8 { break; } // Show first 8 cells
        
        if let Some((deepest_layer, _)) = cell.layers_t.last() {
            let foundry_temp = deepest_layer.temperature_k();
            let cell_id_u64 = u64::from(*cell_id);
            
            let multiplier = surface_op.calculate_foundry_multiplier(final_time_years, cell_id_u64);
            let phase_in_cycle = (final_time_years % 500.0) / 500.0;
            
            println!("   {:016x} {:15.1} {:10.3} {:7.3}", 
                     cell_id_u64, foundry_temp, multiplier, phase_in_cycle);
            
            foundry_temps.push(foundry_temp);
        }
        cell_count += 1;
    }
    
    if !foundry_temps.is_empty() {
        let min_temp = foundry_temps.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_temp = foundry_temps.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let avg_temp = foundry_temps.iter().sum::<f64>() / foundry_temps.len() as f64;
        
        println!("\n📊 Foundry Temperature Statistics:");
        println!("   Minimum: {:.1}K", min_temp);
        println!("   Maximum: {:.1}K", max_temp);
        println!("   Average: {:.1}K", avg_temp);
        println!("   Range: {:.1}K", max_temp - min_temp);
        
        let expected_min = 1500.0 * 0.25; // 375K
        let expected_max = 1500.0 * 1.75; // 2625K
        
        if (min_temp - expected_min).abs() < 100.0 && (max_temp - expected_max).abs() < 100.0 {
            println!("✅ Foundry oscillation working correctly!");
        } else {
            println!("⚠️  Foundry oscillation may need adjustment");
        }
    }
}

/// Analyze current melting state: (liquid_layer_count, total_liquid_mass, depth_range)
fn analyze_melting_state(sim: &Simulation) -> (usize, f64, (f64, f64)) {
    let mut liquid_count = 0;
    let mut total_liquid_mass = 0.0;
    let mut min_depth = f64::INFINITY;
    let mut max_depth = f64::NEG_INFINITY;

    for cell in sim.cells.values() {
        for (layer, _) in &cell.layers_t {
            if matches!(layer.phase(), MaterialPhase::Liquid) {
                liquid_count += 1;
                total_liquid_mass += layer.mass_kg();

                let depth = layer.start_depth_km + layer.height_km / 2.0; // Mid-depth
                min_depth = min_depth.min(depth);
                max_depth = max_depth.max(depth);
            }
        }
    }

    if liquid_count == 0 {
        min_depth = 0.0;
        max_depth = 0.0;
    }

    (liquid_count, total_liquid_mass, (min_depth, max_depth))
}



/// Analyze foundry temperature range across all cells
fn analyze_foundry_temperature_range(sim: &Simulation, _current_time_years: f64) -> (f64, f64) {
    let mut min_temp = f64::INFINITY;
    let mut max_temp = f64::NEG_INFINITY;

    for (_cell_id, cell) in sim.cells.iter() {
        if let Some((deepest_layer, _)) = cell.layers_t.last() {
            let foundry_temp = deepest_layer.temperature_k();
            min_temp = min_temp.min(foundry_temp);
            max_temp = max_temp.max(foundry_temp);
        }
    }

    if min_temp == f64::INFINITY {
        (0.0, 0.0)
    } else {
        (min_temp, max_temp)
    }
}



fn main() {
    run_enhanced_global_atmo_test();
}
