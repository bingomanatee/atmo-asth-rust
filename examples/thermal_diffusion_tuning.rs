use std::fs::File;
use std::io::Write;

/// Thermal Diffusion Parameter Tuning Loop
/// Tests A→A, A→L, and L→L energy transfers to achieve 1-2% target
fn main() {
    println!("🔬 Thermal Diffusion Parameter Tuning Loop");
    println!("==========================================");
    println!("🎯 Target: 1-2% energy transfer per neighbor");
    println!("📊 Testing: A→A, A→L, L→L material combinations");
    
    // Parameter ranges to test
    let conductivity_multipliers = vec![0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0];
    let pressure_baselines = vec![25.0, 30.0, 40.0, 50.0, 60.0, 75.0, 100.0];
    
    let mut best_params = (0.0, 0.0, f64::INFINITY); // (conductivity_mult, pressure_baseline, error)
    let mut results = Vec::new();
    
    println!("\n🔄 Starting parameter sweep...");
    
    for &conductivity_mult in &conductivity_multipliers {
        for &pressure_baseline in &pressure_baselines {
            
            // Test all three material combinations
            let aa_transfer = test_material_transfer(3.2, 3.2, conductivity_mult, pressure_baseline); // A→A
            let al_transfer = test_material_transfer(3.2, 2.5, conductivity_mult, pressure_baseline); // A→L  
            let ll_transfer = test_material_transfer(2.5, 2.5, conductivity_mult, pressure_baseline); // L→L
            
            let mean_transfer = (aa_transfer + al_transfer + ll_transfer) / 3.0;
            
            // Calculate error from target 1.5% (middle of 1-2% range)
            let error = (mean_transfer - 1.5).abs();
            
            println!("   C={:.2}, P={:.1}K → A→A:{:.2}%, A→L:{:.2}%, L→L:{:.2}% | Mean:{:.2}% (error:{:.3})", 
                     conductivity_mult, pressure_baseline, 
                     aa_transfer, al_transfer, ll_transfer, mean_transfer, error);
            
            results.push((conductivity_mult, pressure_baseline, aa_transfer, al_transfer, ll_transfer, mean_transfer, error));
            
            // Update best parameters if this is better
            if error < best_params.2 {
                best_params = (conductivity_mult, pressure_baseline, error);
                println!("   ✅ NEW BEST: {:.2}% mean transfer (target: 1.5%)", mean_transfer);
            }
        }
    }
    
    println!("\n🎯 OPTIMAL PARAMETERS FOUND:");
    println!("   Conductivity multiplier: {:.2}", best_params.0);
    println!("   Pressure baseline: {:.1}K", best_params.1);
    println!("   Final error: {:.3}%", best_params.2);
    
    // Export results to CSV
    export_results_to_csv(&results);
    
    println!("\n📈 Results exported to examples/data/thermal_tuning_results.csv");
    println!("   Use these parameters in ThermalDiffusionOp:");
    println!("   - Line 148: * {:.2} (instead of * 0.25)", best_params.0);
    println!("   - Line 152: / {:.1} (instead of / 50.0)", best_params.1);
}

/// Test energy transfer between two materials with given parameters
fn test_material_transfer(
    from_conductivity: f64, 
    to_conductivity: f64, 
    conductivity_mult: f64, 
    pressure_baseline: f64
) -> f64 {
    // Simulate typical conditions
    let temp_diff: f64 = 100.0; // 100K temperature difference
    let years: f64 = 100.0;     // 100-year time step
    let from_energy: f64 = 1e15; // 1 PJ of energy
    let thermal_capacity: f64 = 1e12; // 1 TJ/K thermal capacity
    
    // Apply the actual thermal diffusion formula from the code
    let conductivity_factor = (1.0 / from_conductivity * 1.0 / to_conductivity).powi(2) * conductivity_mult;
    let pressure_factor = (temp_diff.abs() / pressure_baseline).sqrt().clamp(0.666, 1.333);
    
    let energy_transfer = temp_diff * thermal_capacity * conductivity_factor * pressure_factor * years;
    let energy_throttle = 0.2; // ENERGY_THROTTLE constant
    let max_transfer = energy_transfer.abs().min(from_energy * energy_throttle);
    
    // Calculate percentage of energy transferred
    let transfer_percent = (max_transfer / from_energy) * 100.0;
    
    transfer_percent
}

/// Export tuning results to CSV for analysis
fn export_results_to_csv(results: &[(f64, f64, f64, f64, f64, f64, f64)]) {
    std::fs::create_dir_all("examples/data").expect("Failed to create data directory");
    
    let mut file = File::create("examples/data/thermal_tuning_results.csv")
        .expect("Failed to create CSV file");
    
    // Write header
    writeln!(file, "conductivity_mult,pressure_baseline,aa_transfer_pct,al_transfer_pct,ll_transfer_pct,mean_transfer_pct,error").unwrap();
    
    // Write data
    for &(c_mult, p_base, aa, al, ll, mean, error) in results {
        writeln!(file, "{:.3},{:.1},{:.3},{:.3},{:.3},{:.3},{:.3}", 
                 c_mult, p_base, aa, al, ll, mean, error).unwrap();
    }
    
    println!("📊 Exported {} parameter combinations to CSV", results.len());
}
