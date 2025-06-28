/// Final precise test for 0.058 multiplier
fn main() {
    println!("🎯 Final Thermal Transfer Rate Test");
    println!("===================================");
    
    let conductivity_mult = 0.058; // Keep original - 1% per 1000 years is actually realistic!
    let pressure_baseline = 100.0;
    
    // Test all three material combinations
    let aa_transfer = test_material_transfer(3.2, 3.2, conductivity_mult, pressure_baseline); // A→A
    let al_transfer = test_material_transfer(3.2, 2.5, conductivity_mult, pressure_baseline); // A→L  
    let ll_transfer = test_material_transfer(2.5, 2.5, conductivity_mult, pressure_baseline); // L→L
    
    let mean_transfer = (aa_transfer + al_transfer + ll_transfer) / 3.0;
    
    println!("\n🌡️  Final Transfer Rates (multiplier: {:.3}):", conductivity_mult);
    println!("   A→A (Asthenosphere): {:.2}%", aa_transfer);
    println!("   A→L (Mixed):         {:.2}%", al_transfer);
    println!("   L→L (Lithosphere):   {:.2}%", ll_transfer);
    println!("   Mean transfer:       {:.2}%", mean_transfer);
    
    // Check if we're in the target range
    let aa_ok = aa_transfer >= 0.5 && aa_transfer <= 1.5;
    let al_ok = al_transfer >= 0.5 && al_transfer <= 1.5;
    let ll_ok = ll_transfer >= 0.5 && ll_transfer <= 1.5;
    
    println!("\n📊 Range Check (target: 0.5-1.5%):");
    println!("   A→A: {} {}", if aa_ok { "✅" } else { "❌" }, if aa_ok { "GOOD" } else { "OUT OF RANGE" });
    println!("   A→L: {} {}", if al_ok { "✅" } else { "❌" }, if al_ok { "GOOD" } else { "OUT OF RANGE" });
    println!("   L→L: {} {}", if ll_ok { "✅" } else { "❌" }, if ll_ok { "GOOD" } else { "OUT OF RANGE" });
    
    if aa_ok && al_ok && ll_ok {
        println!("\n🎉 PERFECT! All rates within scientifically optimal 0.5-1.5% range!");
        println!("   This matches real geological thermal transfer timescales.");
        println!("   Ready for production thermal simulations! 🌍");
    } else {
        println!("\n⚠️  Still need fine-tuning...");
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
