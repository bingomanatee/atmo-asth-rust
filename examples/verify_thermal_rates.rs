/// Quick verification of thermal transfer rates with 0.08 multiplier
fn main() {
    println!("🔬 Verifying Thermal Transfer Rates");
    println!("===================================");
    println!("🎯 Target: 0.5-1.5% per century (scientifically optimal)");
    
    let conductivity_mult = 0.06;
    let pressure_baseline = 100.0;
    
    println!("\n📊 Testing with parameters:");
    println!("   Conductivity multiplier: {:.2}", conductivity_mult);
    println!("   Pressure baseline: {:.1}K", pressure_baseline);
    
    // Test all three material combinations
    let aa_transfer = test_material_transfer(3.2, 3.2, conductivity_mult, pressure_baseline); // A→A
    let al_transfer = test_material_transfer(3.2, 2.5, conductivity_mult, pressure_baseline); // A→L  
    let ll_transfer = test_material_transfer(2.5, 2.5, conductivity_mult, pressure_baseline); // L→L
    
    let mean_transfer = (aa_transfer + al_transfer + ll_transfer) / 3.0;
    
    println!("\n🌡️  Transfer Rates:");
    println!("   A→A (Asthenosphere): {:.2}%", aa_transfer);
    println!("   A→L (Mixed):         {:.2}%", al_transfer);
    println!("   L→L (Lithosphere):   {:.2}%", ll_transfer);
    println!("   Mean transfer:       {:.2}%", mean_transfer);
    
    // Check if we're in the target range
    let in_range = aa_transfer >= 0.5 && aa_transfer <= 1.5 &&
                   al_transfer >= 0.5 && al_transfer <= 1.5 &&
                   ll_transfer >= 0.5 && ll_transfer <= 1.5;
    
    if in_range {
        println!("\n✅ PERFECT! All transfer rates in 0.5-1.5% scientifically optimal range");
    } else {
        println!("\n⚠️  Some rates outside optimal range. Adjusting...");
        
        // Try different multipliers to find the sweet spot
        println!("\n🔄 Fine-tuning multiplier:");
        for mult in [0.06, 0.07, 0.08, 0.09, 0.10] {
            let aa = test_material_transfer(3.2, 3.2, mult, pressure_baseline);
            let al = test_material_transfer(3.2, 2.5, mult, pressure_baseline);
            let ll = test_material_transfer(2.5, 2.5, mult, pressure_baseline);
            let mean = (aa + al + ll) / 3.0;
            
            let all_in_range = aa >= 0.5 && aa <= 1.5 && al >= 0.5 && al <= 1.5 && ll >= 0.5 && ll <= 1.5;
            let status = if all_in_range { "✅" } else { "  " };
            
            println!("   {:.2}: A→A:{:.2}%, A→L:{:.2}%, L→L:{:.2}% | Mean:{:.2}% {}", 
                     mult, aa, al, ll, mean, status);
        }
    }
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
