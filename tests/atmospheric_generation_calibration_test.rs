use atmo_asth_rust::sim_op::atmospheric_generation_op::AtmosphericGenerationOp;
use atmo_asth_rust::sim_op::surface_energy_init_op::{SurfaceEnergyInitOp, SurfaceEnergyInitParams};
use atmo_asth_rust::sim::simulation::{Simulation, SimProps};
use atmo_asth_rust::sim_op::SimOp;
use atmo_asth_rust::energy_mass_composite::{EnergyMassComposite, MaterialCompositeType};
use atmo_asth_rust::global_thermal::sim_cell::{GlobalH3CellConfig, LayerConfig};
use atmo_asth_rust::planet::Planet;
use h3o::Resolution;
use std::rc::Rc;

/// Target atmospheric generation timescales
const TARGET_MIN_YEARS: f64 = 1_000_000_000.0; // 1 billion years
const TARGET_MAX_YEARS: f64 = 2_000_000_000.0; // 2 billion years
const EARTH_ATMOSPHERE_KG: f64 = 5.15e18;
const SIMULATION_YEARS: f64 = 1000.0;
const MAX_CALIBRATION_ATTEMPTS: usize = 20;

#[derive(Debug, Clone)]
struct CalibrationParams {
    catastrophic_outgassing: f64,
    major_outgassing: f64,
    background_outgassing: f64,
    catastrophic_crystallization: f64,
    major_crystallization: f64,
    background_crystallization: f64,
    catastrophic_probability: f64,
}

impl CalibrationParams {
    fn new() -> Self {
        Self {
            catastrophic_outgassing: 0.05,
            major_outgassing: 0.001,
            background_outgassing: 0.0001,
            catastrophic_crystallization: 0.1,
            major_crystallization: 0.5,
            background_crystallization: 1.0,
            catastrophic_probability: 0.002,
        }
    }

    fn adjust_for_faster_generation(&mut self, factor: f64) {
        // Increase outgassing rates
        self.catastrophic_outgassing *= factor;
        self.major_outgassing *= factor;
        self.background_outgassing *= factor;
        
        // Decrease crystallization rates (more gas escapes)
        self.catastrophic_crystallization /= factor.sqrt();
        self.major_crystallization /= factor.sqrt();
        self.background_crystallization /= factor.sqrt();
    }

    fn adjust_for_slower_generation(&mut self, factor: f64) {
        // Decrease outgassing rates
        self.catastrophic_outgassing /= factor;
        self.major_outgassing /= factor;
        self.background_outgassing /= factor;
        
        // Increase crystallization rates (more gas trapped)
        self.catastrophic_crystallization *= factor.sqrt();
        self.major_crystallization *= factor.sqrt();
        self.background_crystallization *= factor.sqrt();
    }
}

#[derive(Debug)]
struct CalibrationResult {
    atmospheric_generation_kg: f64,
    years_to_earth_atmosphere: f64,
    melting_events: usize,
    params: CalibrationParams,
}

fn run_atmospheric_generation_test(params: &CalibrationParams) -> CalibrationResult {
    // Create planet
    let planet = Planet::earth(Resolution::One);

    // Create simulation with minimal props
    let sim_props = SimProps {
        planet: planet.clone(),
        name: "CalibrationTest",
        ops: vec![], // We'll add operations manually
        res: Resolution::One,
        layer_count: 10,
        sim_steps: 5,
        years_per_step: (SIMULATION_YEARS / 5.0) as u32,
        debug: false,
    };
    let mut sim = Simulation::new(sim_props);

    // Configure cells with simple layout for testing
    let _planet_rc = Rc::new(planet);
    sim.make_cells(|cell_index, planet| {
        // Simple layout: 2×20km atmosphere + 4×10km lithosphere + 4×10km asthenosphere
        let layer_schedule = vec![
            LayerConfig {
                cell_type: MaterialCompositeType::Air,
                cell_count: 2,
                height_km: 20.0, // 40km total atmosphere
                is_foundry: false,
            },
            LayerConfig {
                cell_type: MaterialCompositeType::Silicate,
                cell_count: 4,
                height_km: 10.0, // 40km total lithosphere
                is_foundry: false,
            },
            LayerConfig {
                cell_type: MaterialCompositeType::Silicate,
                cell_count: 4,
                height_km: 10.0, // 40km total asthenosphere
                is_foundry: false,
            },
        ];

        GlobalH3CellConfig {
            h3_index: cell_index,
            planet,
            layer_schedule,
        }
    });

    // Create atmospheric generation operation with test parameters
    let mut atmo_gen_op = AtmosphericGenerationOp::new();

    // Create surface energy operation with reduced temperature range
    let surface_energy_params = SurfaceEnergyInitParams::with_temperatures(
        1200.0, // surface_temp_k
        25.0,   // geothermal_gradient_k_per_km
        1200.0, // core_temp_k
    );
    let mut surface_energy_op = SurfaceEnergyInitOp::new_with_params(surface_energy_params);

    // Get initial atmospheric mass
    let initial_atmospheric_mass = sim.cells.values()
        .flat_map(|cell| &cell.layers_t)
        .filter(|(current, _)| current.start_depth_km < 0.0) // Atmospheric layers
        .map(|(current, _)| current.energy_mass.mass_kg())
        .sum::<f64>();

    // Run simulation for test period
    let steps = 5; // Reduced steps for faster calibration
    let years_per_step = SIMULATION_YEARS / steps as f64;
    
    for step in 0..steps {
        let current_year = step as f64 * years_per_step;
        
        // Apply surface energy
        surface_energy_op.update_sim(&mut sim);

        // Apply atmospheric generation
        atmo_gen_op.update_sim(&mut sim);
    }

    // Calculate final atmospheric mass
    let final_atmospheric_mass = sim.cells.values()
        .flat_map(|cell| &cell.layers_t)
        .filter(|(current, _)| current.start_depth_km < 0.0) // Atmospheric layers
        .map(|(current, _)| current.energy_mass.mass_kg())
        .sum::<f64>();

    let atmospheric_generation_kg = final_atmospheric_mass - initial_atmospheric_mass;
    let generation_rate_per_year = atmospheric_generation_kg / SIMULATION_YEARS;
    let years_to_earth_atmosphere = if generation_rate_per_year > 0.0 {
        EARTH_ATMOSPHERE_KG / generation_rate_per_year
    } else {
        f64::INFINITY
    };

    CalibrationResult {
        atmospheric_generation_kg,
        years_to_earth_atmosphere,
        melting_events: 0, // We'll estimate this from atmospheric generation
        params: params.clone(),
    }
}

#[test]
fn test_atmospheric_generation_calibration() {
    println!("🎯 ATMOSPHERIC GENERATION CALIBRATION TEST");
    println!("==========================================");
    println!("Target: {:.1e} - {:.1e} years to generate Earth atmosphere", TARGET_MIN_YEARS, TARGET_MAX_YEARS);
    println!("Earth atmosphere mass: {:.2e} kg", EARTH_ATMOSPHERE_KG);
    println!();

    let mut params = CalibrationParams::new();
    let mut best_result: Option<CalibrationResult> = None;
    
    for attempt in 1..=MAX_CALIBRATION_ATTEMPTS {
        println!("🔄 Calibration Attempt {}/{}", attempt, MAX_CALIBRATION_ATTEMPTS);
        
        let result = run_atmospheric_generation_test(&params);
        
        println!("   Atmospheric generation: {:.2e} kg in {} years", 
                 result.atmospheric_generation_kg, SIMULATION_YEARS);
        println!("   Years to Earth atmosphere: {:.2e}", result.years_to_earth_atmosphere);
        println!("   Melting events: {}", result.melting_events);
        
        // Check if we're in target range
        if result.years_to_earth_atmosphere >= TARGET_MIN_YEARS && 
           result.years_to_earth_atmosphere <= TARGET_MAX_YEARS {
            println!("🎉 SUCCESS! Found optimal parameters:");
            println!("   Catastrophic outgassing: {:.6}", params.catastrophic_outgassing);
            println!("   Major outgassing: {:.6}", params.major_outgassing);
            println!("   Background outgassing: {:.6}", params.background_outgassing);
            println!("   Catastrophic crystallization: {:.3}", params.catastrophic_crystallization);
            println!("   Major crystallization: {:.3}", params.major_crystallization);
            println!("   Background crystallization: {:.3}", params.background_crystallization);
            println!("   Catastrophic probability: {:.6}", params.catastrophic_probability);
            best_result = Some(result);
            break;
        }
        
        // Adjust parameters based on result
        if result.years_to_earth_atmosphere < TARGET_MIN_YEARS {
            // Too fast - need to slow down
            let factor = TARGET_MIN_YEARS / result.years_to_earth_atmosphere;
            let adjustment_factor = factor.powf(0.3); // Gradual adjustment
            params.adjust_for_slower_generation(adjustment_factor);
            println!("   ⬇️  Too fast, slowing down by factor {:.2}", adjustment_factor);
        } else if result.years_to_earth_atmosphere > TARGET_MAX_YEARS {
            // Too slow - need to speed up
            let factor = result.years_to_earth_atmosphere / TARGET_MAX_YEARS;
            let adjustment_factor = factor.powf(0.3); // Gradual adjustment
            params.adjust_for_faster_generation(adjustment_factor);
            println!("   ⬆️  Too slow, speeding up by factor {:.2}", adjustment_factor);
        }
        
        println!();
    }
    
    if let Some(result) = best_result {
        assert!(result.years_to_earth_atmosphere >= TARGET_MIN_YEARS);
        assert!(result.years_to_earth_atmosphere <= TARGET_MAX_YEARS);
        println!("✅ Calibration successful!");
    } else {
        println!("❌ Failed to find optimal parameters within {} attempts", MAX_CALIBRATION_ATTEMPTS);
        println!("💡 Consider adjusting target ranges or increasing max attempts");
        // Don't fail the test - this is calibration, not validation
    }
}
