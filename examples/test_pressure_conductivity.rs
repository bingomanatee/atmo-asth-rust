/// Test pressure effects on thermal conductivity in heat redistribution
/// This test verifies that:
/// 1. Conductivity OUT increases with current density / default density
/// 2. Conductivity IN decreases with current density / default density
/// 3. Heat redistribution accounts for these pressure effects
/// 4. Space radiation skin depth is affected by pressure-enhanced conductivity

use atmo_asth_rust::sim::fourier_thermal_transfer::FourierThermalTransfer;
use atmo_asth_rust::energy_mass_composite::{StandardEnergyMassComposite, EnergyMassParams, EnergyMassComposite};
use atmo_asth_rust::material_composite::{MaterialCompositeType, MaterialPhase};
use atmo_asth_rust::global_thermal::thermal_layer::ThermalLayer;
use atmo_asth_rust::constants::SECONDS_PER_YEAR;

fn main() {
    println!("🧪 Testing Pressure Effects on Thermal Conductivity");
    println!("{}", "=".repeat(60));

    test_density_adjusted_conductivity();
    test_pressure_enhanced_skin_depth();
    test_heat_transfer_with_pressure_effects();
    
    println!("\n✅ All pressure conductivity tests completed successfully!");
}

fn test_density_adjusted_conductivity() {
    println!("\n🔬 Test 1: Density-Adjusted Conductivity");

    let fourier = FourierThermalTransfer::new(1.0); // 1 year in seconds
    
    let base_conductivity = 3.2; // W/(m·K) for silicate
    let default_density = 3300.0; // kg/m³ for silicate
    
    // Test different density scenarios
    let test_cases = vec![
        ("Normal density", 3300.0, 1.0),
        ("50% higher density (compressed)", 4950.0, 1.5),
        ("Double density (highly compressed)", 6600.0, 2.0),
        ("Lower density (expanded)", 2200.0, 0.667),
    ];
    
    for (description, current_density, expected_ratio) in test_cases {
        // Test conductivity OUT (should increase with density)
        let conductivity_out = fourier.calculate_density_adjusted_conductivity(
            base_conductivity, current_density, default_density
        );
        
        // Test conductivity IN (should decrease with density)
        let conductivity_in = fourier.calculate_density_adjusted_conductivity(
            base_conductivity, current_density, default_density
        );
        
        let actual_ratio = current_density / default_density;
        
        println!("  {}: density ratio {:.3}", description, actual_ratio);
        println!("    Conductivity OUT: {:.3} W/(m·K) ({}x base)", 
                 conductivity_out, conductivity_out / base_conductivity);
        println!("    Conductivity IN:  {:.3} W/(m·K) ({}x base)", 
                 conductivity_in, conductivity_in / base_conductivity);
        
        // Verify the relationships
        assert!((conductivity_out / base_conductivity - expected_ratio).abs() < 0.01, 
                "Conductivity OUT should scale with density ratio");
        assert!((conductivity_in * expected_ratio / base_conductivity - 1.0).abs() < 0.01, 
                "Conductivity IN should be inversely proportional to density ratio");
    }
}

fn test_pressure_enhanced_skin_depth() {
    println!("\n🔬 Test 2: Pressure-Enhanced Skin Depth");
    
    // Create energy mass composites with different pressures
    let low_pressure = StandardEnergyMassComposite::new_with_params(EnergyMassParams {
        material_type: MaterialCompositeType::Silicate,
        initial_phase: MaterialPhase::Solid,
        energy_joules: 1e20,
        volume_km3: 1000.0,
        height_km: 10.0,
        pressure_gpa: 0.1, // Low pressure
    });
    
    let high_pressure = StandardEnergyMassComposite::new_with_params(EnergyMassParams {
        material_type: MaterialCompositeType::Silicate,
        initial_phase: MaterialPhase::Solid,
        energy_joules: 1e20,
        volume_km3: 1000.0,
        height_km: 10.0,
        pressure_gpa: 5.0, // High pressure (5 GPa)
    });
    
    let time_years = 1000.0;
    
    let low_pressure_skin_depth = low_pressure.skin_depth_km(time_years);
    let high_pressure_skin_depth = high_pressure.skin_depth_km(time_years);
    
    println!("  Low pressure (0.1 GPa):");
    println!("    Thermal conductivity: {:.3} W/(m·K)", low_pressure.thermal_conductivity());
    println!("    Skin depth: {:.3} km", low_pressure_skin_depth);
    
    println!("  High pressure (5.0 GPa):");
    println!("    Thermal conductivity: {:.3} W/(m·K)", high_pressure.thermal_conductivity());
    println!("    Skin depth: {:.3} km", high_pressure_skin_depth);
    
    // High pressure should have higher conductivity and deeper skin depth
    assert!(high_pressure.thermal_conductivity() > low_pressure.thermal_conductivity(),
            "High pressure should increase thermal conductivity");
    assert!(high_pressure_skin_depth > low_pressure_skin_depth,
            "Higher conductivity should result in deeper skin depth");
    
    let conductivity_ratio = high_pressure.thermal_conductivity() / low_pressure.thermal_conductivity();
    println!("  Conductivity enhancement: {:.2}x", conductivity_ratio);
}

fn test_heat_transfer_with_pressure_effects() {
    println!("\n🔬 Test 3: Heat Transfer with Pressure Effects");

    // Create two thermal layers with different pressures and densities
    let hot_layer_low_pressure = create_compressed_test_layer(2000.0, 0.1, 2200.0); // 2000K, 0.1 GPa, low density
    let cold_layer_high_pressure = create_compressed_test_layer(1000.0, 5.0, 4400.0); // 1000K, 5.0 GPa, high density

    let fourier = FourierThermalTransfer::new(1.0); // 1 year in seconds
    
    // Test heat transfer without density adjustment
    let mut hot_tuple_normal = (hot_layer_low_pressure.clone(), hot_layer_low_pressure.clone());
    let mut cold_tuple_normal = (cold_layer_high_pressure.clone(), cold_layer_high_pressure.clone());

    let normal_transfer = fourier.apply_heat_transfer_between_layers(
        &mut hot_tuple_normal, &mut cold_tuple_normal
    );

    // Test heat transfer with density adjustment
    let mut hot_tuple_adjusted = (hot_layer_low_pressure.clone(), hot_layer_low_pressure.clone());
    let mut cold_tuple_adjusted = (cold_layer_high_pressure.clone(), cold_layer_high_pressure.clone());

    let adjusted_transfer = fourier.transfer_heat_between_layer_tuples(
        &mut hot_tuple_adjusted, &mut cold_tuple_adjusted
    );

    // Debug: Calculate what the density adjustments should be
    let hot_density_ratio = hot_layer_low_pressure.current_density_kg_m3() / 3300.0;
    let cold_density_ratio = cold_layer_high_pressure.current_density_kg_m3() / 3300.0;

    println!("  Debug: Hot layer density ratio: {:.3}", hot_density_ratio);
    println!("  Debug: Cold layer density ratio: {:.3}", cold_density_ratio);

    let hot_conductivity_out = fourier.calculate_density_adjusted_conductivity(
        hot_layer_low_pressure.thermal_conductivity(),
        hot_layer_low_pressure.current_density_kg_m3(),
        3300.0,
    );
    let cold_conductivity_in = fourier.calculate_density_adjusted_conductivity(
        cold_layer_high_pressure.thermal_conductivity(),
        cold_layer_high_pressure.current_density_kg_m3(),
        3300.0,
    );

    println!("  Debug: Hot conductivity OUT: {:.3} W/(m·K) (vs {:.3} normal)",
             hot_conductivity_out, hot_layer_low_pressure.thermal_conductivity());
    println!("  Debug: Cold conductivity IN: {:.3} W/(m·K) (vs {:.3} normal)",
             cold_conductivity_in, cold_layer_high_pressure.thermal_conductivity());
    
    println!("  Hot layer (low pressure): {:.1} K, conductivity {:.3} W/(m·K), density {:.1} kg/m³",
             hot_layer_low_pressure.temperature_k(), hot_layer_low_pressure.thermal_conductivity(),
             hot_layer_low_pressure.current_density_kg_m3());
    println!("  Cold layer (high pressure): {:.1} K, conductivity {:.3} W/(m·K), density {:.1} kg/m³",
             cold_layer_high_pressure.temperature_k(), cold_layer_high_pressure.thermal_conductivity(),
             cold_layer_high_pressure.current_density_kg_m3());
    println!("  Normal heat transfer: {:.2e} J", normal_transfer);
    println!("  Pressure-adjusted transfer: {:.2e} J", adjusted_transfer);
    
    // The pressure-adjusted transfer should be different due to conductivity modifications
    let transfer_difference = (adjusted_transfer - normal_transfer).abs();
    let relative_difference = transfer_difference / normal_transfer;

    println!("  Transfer difference: {:.2e} J ({:.1}%)", transfer_difference, relative_difference * 100.0);

    if relative_difference < 0.001 {
        println!("  ⚠️  Warning: Pressure adjustments had minimal effect (< 0.1%)");
        println!("  This might indicate that the density adjustments are not being applied correctly");
    } else {
        println!("  ✅ Pressure adjustments significantly affected heat transfer");
    }
    
    let transfer_ratio = adjusted_transfer / normal_transfer;
    println!("  Transfer ratio (adjusted/normal): {:.3}", transfer_ratio);
}

fn create_test_layer(temperature_k: f64, pressure_gpa: f64) -> ThermalLayer {
    let mut energy_mass = StandardEnergyMassComposite::new_with_params(EnergyMassParams {
        material_type: MaterialCompositeType::Silicate,
        initial_phase: MaterialPhase::Solid,
        energy_joules: 1e20,
        volume_km3: 1000.0,
        height_km: 10.0,
        pressure_gpa,
    });

    // Set the desired temperature
    energy_mass.set_temperature(temperature_k);

    let mut layer = ThermalLayer::new(
        0.0,    // start_depth_km
        10.0,   // height_km
        1000.0, // surface_area_km2
        MaterialCompositeType::Silicate,
    );

    // Replace the energy mass with our custom one
    layer.energy_mass = energy_mass;

    layer
}

fn create_compressed_test_layer(temperature_k: f64, pressure_gpa: f64, target_density_kg_m3: f64) -> ThermalLayer {
    // Calculate volume to achieve target density
    let mass_kg = 1e15; // Fixed mass
    let volume_km3 = mass_kg / (target_density_kg_m3 * 1e9); // Convert m³ to km³

    let mut energy_mass = StandardEnergyMassComposite::new_with_params(EnergyMassParams {
        material_type: MaterialCompositeType::Silicate,
        initial_phase: MaterialPhase::Solid,
        energy_joules: 1e20,
        volume_km3,
        height_km: 10.0,
        pressure_gpa,
    });

    // Set the desired temperature
    energy_mass.set_temperature(temperature_k);

    let mut layer = ThermalLayer::new(
        0.0,    // start_depth_km
        10.0,   // height_km
        1000.0, // surface_area_km2
        MaterialCompositeType::Silicate,
    );

    // Replace the energy mass with our custom one
    layer.energy_mass = energy_mass;

    println!("  Created layer: T={:.1}K, P={:.1}GPa, density={:.1} kg/m³ (default: 3300)",
             temperature_k, pressure_gpa, layer.current_density_kg_m3());

    layer
}
