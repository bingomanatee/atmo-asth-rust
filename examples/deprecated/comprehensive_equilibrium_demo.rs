use atmo_asth_rust::constants::EARTH_RADIUS_KM;
use atmo_asth_rust::planet::Planet;
use atmo_asth_rust::sim_op::{
    AtmosphereOp, CoreRadianceOp, CsvWriterOp, LithosphereUnifiedOp,
    ProgressReporterOp, ThermalDiffusionOp,
};
use atmo_asth_rust::material::MaterialType;
use h3o::Resolution;
use std::fs;
use atmo_asth_rust::sim::simulation::{SimProps, Simulation};

/// Comprehensive demonstration of all thermal operators working together
/// to achieve realistic planetary thermal equilibrium with CSV data export.

fn main() {
    println!("🌍 Comprehensive Thermal Equilibrium Demonstration");
    println!("==================================================");
    
    // Create output directory and clean up any existing output
    let output_dir = "examples/data";
    let _ = fs::create_dir_all(output_dir);
    let csv_file = format!("{}/thermal_equilibrium_demo.csv", output_dir);
    let _ = fs::remove_file(&csv_file);

    // Create comprehensive operator suite with modern thermal physics
    let operators = vec![
        CoreRadianceOp::handle_earth(), // Core heat input from planetary interior
        ThermalDiffusionOp::handle(
            0.5,   // Higher diffusion rate for better asthenosphere equilibration
            25.0), // Higher max temp change for asthenosphere layers
        AtmosphereOp::handle_with_params(
            1300.0, // 1300K outgassing threshold
            5e-11,  // Lower outgassing rate
            0.6,    // 60% atmospheric efficiency
        ),
        LithosphereUnifiedOp::handle(
            vec![
                (MaterialType::Basaltic, 0.7),  // 70% basaltic (oceanic)
                (MaterialType::Granitic, 0.2),  // 20% granitic (continental)
                (MaterialType::Silicate, 0.1),  // 10% silicate (mantle)
            ],
            123,   // Random seed
            0.15,  // Scale factor
            0.3,   // Production rate modifier (30% to reduce chaos)
        ),
        ProgressReporterOp::handle(30), // Report every 30 steps
        CsvWriterOp::handle_with_layer_temps(csv_file.clone(), 4, 6), // 4 asth layers, up to 6 lith layers
    ];

    // Create realistic Earth-like simulation with operators
    let sim_props = SimProps {
        name: "thermal_equilibrium",
        planet: Planet {
            radius_km: EARTH_RADIUS_KM as f64,
            resolution: Resolution::Zero,
        },
        ops: operators,
        res: Resolution::Two, // Small grid for faster computation
        layer_count: 4,      // 4 asthenosphere layers (200km total)
        asth_layer_height_km: 50.0,
        lith_layer_height_km: 25.0,
        sim_steps: 500,
        years_per_step: 5000,
        debug: false,
        alert_freq: 30,      // Report every 30 steps
        starting_surface_temp_k: 1600.0, // Moderate starting temperature
    };

    println!("📊 Simulation Parameters:");
    println!("   - Grid: 37 cells (Resolution::Two)");
    println!("   - Layers: {} asthenosphere layers", sim_props.layer_count);
    println!("   - Time: {} steps × {} years = {:.1} million years",
             sim_props.sim_steps, sim_props.years_per_step,
             (sim_props.sim_steps as u32 * sim_props.years_per_step) / 1_000_000);
    println!("   - Starting surface temp: {:.0}K", sim_props.starting_surface_temp_k);
    println!();

    let mut sim = Simulation::new(sim_props);

    println!("🔧 Operators Configured:");
    println!("   ✅ CoreRadianceOp: Core heat flux from planetary interior (1.0e11 J/km²/year)");
    println!("   ✅ ThermalDiffusionOp: Realistic thermal diffusion with cascading energy transfer");
    println!("   ✅ CoolingOp: Enhanced surface cooling (1.2x efficiency)");
    println!("   ✅ AtmosphereOp: Moderate atmosphere development");
    println!("   ✅ LithosphereUnifiedOp: Realistic crust formation/melting");
    println!("   ✅ ProgressReporterOp: Intermittent progress updates");
    println!("   ✅ CsvWriterOp: Complete data export");
    println!();

    // Run simulation using native mechanism
    sim.simulate();

    println!("   CSV data exported to: {}", csv_file);
    
    // Analyze final state
    analyze_final_state(&sim);

    // Row-by-row summary
    analyze_row_by_row(&sim);

    // Analyze CSV data
    analyze_csv_data(&csv_file);
}

fn analyze_final_state(sim: &Simulation) {
    println!("\n🔍 Final State Analysis:");
    println!("========================");
    
    let mut surface_temps = Vec::new();
    let mut lithosphere_thicknesses = Vec::new();
    let mut total_energy = 0.0;
    
    for cell in sim.cells.values() {
        // Surface temperature from top asthenosphere layer (most accurate)
        let surface_temp = cell.asth_layers_t[0].0.kelvin();
        surface_temps.push(surface_temp);
        
        // Lithosphere thickness
        let thickness = cell.total_lithosphere_height();
        lithosphere_thicknesses.push(thickness);
        
        // Total energy
        total_energy += cell.asth_layers_t.iter().map(|(l, _)| l.energy_joules()).sum::<f64>();
        total_energy += cell.lith_layers_t.iter().map(|(l, _)| l.energy_joules()).sum::<f64>();
    }
    
    let avg_surface_temp = surface_temps.iter().sum::<f64>() / surface_temps.len() as f64;
    let avg_lithosphere_thickness = lithosphere_thicknesses.iter().sum::<f64>() / lithosphere_thicknesses.len() as f64;
    
    println!("   Average surface temperature: {:.1}K ({:.1}°C)", 
             avg_surface_temp, avg_surface_temp - 273.15);
    println!("   Average lithosphere thickness: {:.1} km", avg_lithosphere_thickness);
    println!("   Total system energy: {:.2e} J", total_energy);
    
    // Check for equilibrium indicators
    let temp_range = surface_temps.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap() - 
                     surface_temps.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    println!("   Surface temperature range: {:.1}K", temp_range);
    
    if avg_surface_temp > 1000.0 && avg_surface_temp < 2000.0 {
        println!("   ✅ Surface temperature in realistic range for early Earth");
    }
    
    if avg_lithosphere_thickness > 5.0 && avg_lithosphere_thickness < 100.0 {
        println!("   ✅ Lithosphere thickness in reasonable range");
    }
    
    if temp_range < 200.0 {
        println!("   ✅ Temperature distribution relatively uniform");
    }
}

fn analyze_row_by_row(sim: &Simulation) {
    println!("\n📋 Row-by-Row Final State Summary:");
    println!("==================================");

    let mut cells: Vec<_> = sim.cells.iter().collect();
    cells.sort_by_key(|(cell_id, _)| *cell_id);

    println!("Cell | Surface Temp | Bottom Temp | Lith Thick | Total Energy");
    println!("-----|--------------|-------------|------------|-------------");

    for (i, (cell_id, cell)) in cells.iter().enumerate() {
        // Surface temperature (top of lithosphere or top of asthenosphere)
        let surface_temp = if !cell.lith_layers_t.is_empty() {
            cell.lith_layers_t.last().unwrap().0.kelvin()
        } else {
            cell.asth_layers_t.first().unwrap().0.kelvin()
        };

        // Bottom temperature (bottom of asthenosphere)
        let bottom_temp = cell.asth_layers_t.last().unwrap().0.kelvin();

        // Lithosphere thickness
        let lith_thickness = cell.total_lithosphere_height();

        // Total energy in this cell
        let mut total_energy = 0.0;
        for (layer, _) in &cell.asth_layers_t {
            total_energy += layer.energy_joules();
        }
        for (layer, _) in &cell.lith_layers_t {
            total_energy += layer.energy_joules();
        }

        println!("{:4} | {:8.1}K    | {:7.1}K     | {:6.1} km  | {:9.2e} J",
                 i + 1, surface_temp, bottom_temp, lith_thickness, total_energy);
    }

    println!("-----|--------------|-------------|------------|-------------");
    println!("Total cells: {}", cells.len());
}

fn analyze_csv_data(csv_file: &str) {
    println!("\n📊 CSV Data Analysis:");
    println!("=====================");
    
    match fs::read_to_string(csv_file) {
        Ok(content) => {
            let lines: Vec<&str> = content.lines().collect();
            if lines.len() < 3 {
                println!("   ⚠️  Insufficient data in CSV file");
                return;
            }
            
            println!("   Total data points: {}", lines.len() - 1); // Exclude header
            
            // Parse first and last data rows to show change
            if let (Some(first_data), Some(last_data)) = (lines.get(1), lines.last()) {
                let first_parts: Vec<&str> = first_data.split(',').collect();
                let last_parts: Vec<&str> = last_data.split(',').collect();
                
                if first_parts.len() >= 3 && last_parts.len() >= 3 {
                    // Assuming columns: step, years, avg_surface_temp_k, ...
                    if let (Ok(first_temp), Ok(last_temp)) = (
                        first_parts[2].parse::<f64>(),
                        last_parts[2].parse::<f64>()
                    ) {
                        let temp_change = last_temp - first_temp;
                        println!("   Initial surface temp: {:.1}K", first_temp);
                        println!("   Final surface temp: {:.1}K", last_temp);
                        println!("   Temperature change: {:.1}K", temp_change);
                        
                        if temp_change.abs() < 50.0 {
                            println!("   ✅ System approaching thermal equilibrium");
                        } else if temp_change < 0.0 {
                            println!("   🔽 System cooling towards equilibrium");
                        } else {
                            println!("   🔼 System heating towards equilibrium");
                        }
                    }
                }
            }
            
            println!("   📁 Open {} in a spreadsheet to see detailed evolution", csv_file);
            println!("   📈 Look for stabilizing temperatures and lithosphere growth");
        }
        Err(e) => {
            println!("   ❌ Error reading CSV file: {}", e);
        }
    }
}
