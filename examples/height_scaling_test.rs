use atmo_asth_rust::energy_mass::{EnergyMass, StandardEnergyMass};
use atmo_asth_rust::material::MaterialType;

/// Test the height-based scaling for thermal transfer
fn main() {
    println!("🔥 Height-Based Thermal Transfer Scaling Test");
    println!("==============================================\n");

    // Test different layer heights
    let layer_heights = vec![1.0, 5.0, 10.0, 20.0, 50.0, 100.0];
    
    println!("Height Scaling Factors (base rates calibrated for 1km):");
    println!("-------------------------------------------------------");
    
    for &height_km in &layer_heights {
        let scale_factor = height_km.sqrt();
        println!("{:>6.0}km layer: {:.2}x scaling", height_km, scale_factor);
    }
    println!();

    // Test actual energy transfer with different heights
    println!("Energy Transfer Comparison:");
    println!("---------------------------");
    println!("Materials: Silicate (2000K) → Basaltic (1500K), 1000 years\n");

    let area_km2 = 85300.0; // Typical Resolution 2 cell area
    
    for &height_km in &layer_heights {
        let volume_km3 = area_km2 * height_km;
        
        let hot_silicate = create_sample_material(MaterialType::Silicate, 2000.0, volume_km3);
        let cold_basaltic = create_sample_material(MaterialType::Basaltic, 1500.0, volume_km3);
        
        // Base transfer (without height scaling)
        let base_transfer = hot_silicate.calculate_thermal_transfer(&cold_basaltic, 0.1, 1000.0);
        
        // Height-scaled transfer
        let height_scale = height_km.sqrt();
        let scaled_transfer = base_transfer * height_scale;
        
        println!("{:>6.0}km layer (volume: {:.1e} km³):", height_km, volume_km3);
        println!("  Base transfer:   {:.2e} J", base_transfer);
        println!("  Scaled transfer: {:.2e} J ({:.2}x)", scaled_transfer, height_scale);
        println!("  Per km³:         {:.2e} J/km³", scaled_transfer / volume_km3);
        println!();
    }

    // Compare with layer energy content
    println!("Transfer vs Layer Energy Content:");
    println!("---------------------------------");
    
    for &height_km in &[10.0, 50.0] {
        let volume_km3 = area_km2 * height_km;
        let hot_silicate = create_sample_material(MaterialType::Silicate, 2000.0, volume_km3);
        let cold_basaltic = create_sample_material(MaterialType::Basaltic, 1500.0, volume_km3);
        
        let base_transfer = hot_silicate.calculate_thermal_transfer(&cold_basaltic, 0.1, 1000.0);
        let scaled_transfer = base_transfer * height_km.sqrt();
        
        let layer_energy = hot_silicate.energy_joules();
        let transfer_percentage = (scaled_transfer / layer_energy) * 100.0;
        
        println!("{:>6.0}km layer:", height_km);
        println!("  Layer energy:    {:.2e} J", layer_energy);
        println!("  Transfer:        {:.2e} J", scaled_transfer);
        println!("  Transfer %:      {:.3}%", transfer_percentage);
        println!();
    }

    // Realistic transfer rates
    println!("Realistic Transfer Assessment:");
    println!("------------------------------");
    println!("For a typical thermal simulation step (1000 years):");
    println!();
    
    let typical_configs = vec![
        ("Small layers", 10.0),
        ("Medium layers", 20.0), 
        ("Large layers", 50.0),
    ];
    
    for (name, height_km) in typical_configs {
        let volume_km3 = area_km2 * height_km;
        let hot_silicate = create_sample_material(MaterialType::Silicate, 1800.0, volume_km3);
        let cold_basaltic = create_sample_material(MaterialType::Basaltic, 1600.0, volume_km3);
        
        let base_transfer = hot_silicate.calculate_thermal_transfer(&cold_basaltic, 0.1, 1000.0);
        let scaled_transfer = base_transfer * height_km.sqrt();
        
        let temp_change_k = scaled_transfer / (hot_silicate.volume_km3() * hot_silicate.density_kg_m3() * 1e9 * hot_silicate.specific_heat_capacity_j_per_kg_k());
        
        println!("{} ({:.0}km):", name, height_km);
        println!("  Energy transfer: {:.2e} J", scaled_transfer);
        println!("  Temperature change: {:.2}K", temp_change_k);
        println!("  Scaling factor: {:.2}x", height_km.sqrt());
        println!();
    }
}

/// Helper function to create a sample material with specified properties
fn create_sample_material(material_type: MaterialType, temp_k: f64, volume_km3: f64) -> StandardEnergyMass {
    StandardEnergyMass::new_with_material(material_type, temp_k, volume_km3)
}
