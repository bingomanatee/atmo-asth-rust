use atmo_asth_rust::energy_mass::{EnergyMass, StandardEnergyMass};
use atmo_asth_rust::material::MaterialType;

/// Experiment to demonstrate thermal energy transfer between different materials
/// Shows how conductivity and temperature differences affect energy transfer rates
fn main() {
    println!("🌡️ Thermal Energy Transfer Experiment");
    println!("=====================================\n");

    // Test different material combinations
    let material_pairs = vec![
        (MaterialType::Basaltic, MaterialType::Granitic),
        (MaterialType::Basaltic, MaterialType::Silicate),
        (MaterialType::Granitic, MaterialType::Silicate),
        (MaterialType::Silicate, MaterialType::Basaltic),
        (MaterialType::Silicate, MaterialType::Granitic),
    ];

    // Test different time spans (years)
    let time_spans = vec![1.0, 10.0, 100.0, 1000.0, 10000.0];

    // Test different diffusion rates
    let diffusion_rates = vec![0.01, 0.1, 0.5, 1.0];

    println!("Material Thermal Conductivities:");
    for material in &[MaterialType::Basaltic, MaterialType::Granitic, MaterialType::Silicate, MaterialType::Metallic] {
        let sample = create_sample_material(*material, 1500.0, 100.0);
        println!("  {:?}: {:.2} W/(m·K)", material, sample.thermal_conductivity());
    }
    println!();

    // Experiment 1: Material pair comparison at fixed conditions
    println!("Experiment 1: Material Pair Comparison");
    println!("--------------------------------------");
    println!("Conditions: 1000 years, diffusion_rate=0.1, temp_diff=500K");
    println!("Hot material: 2000K, Cold material: 1500K\n");

    for (hot_material, cold_material) in &material_pairs {
        let hot_sample = create_sample_material(*hot_material, 2000.0, 100.0);
        let cold_sample = create_sample_material(*cold_material, 1500.0, 100.0);

        let energy_transfer = hot_sample.calculate_thermal_transfer(&cold_sample, 0.1, 1000.0);
        let transfer_percent = (energy_transfer / hot_sample.energy()) * 100.0;

        println!("{:?} → {:?}: {:.2e} J ({:.3}%)", hot_material, cold_material, energy_transfer, transfer_percent);
        println!("  Interface conductivity: {:.2} W/(m·K)",
            calculate_interface_conductivity(hot_sample.thermal_conductivity(), cold_sample.thermal_conductivity()));
    }
    println!();

    // Experiment 2: Time span effects
    println!("Experiment 2: Time Span Effects");
    println!("-------------------------------");
    println!("Materials: Basaltic (2000K) → Granitic (1500K), diffusion_rate=0.1\n");

    let hot_basalt = create_sample_material(MaterialType::Basaltic, 2000.0, 100.0);
    let cold_granite = create_sample_material(MaterialType::Granitic, 1500.0, 100.0);

    for &years in &time_spans {
        let energy_transfer = hot_basalt.calculate_thermal_transfer(&cold_granite, 0.1, years);
        let transfer_per_year = energy_transfer / years;
        let transfer_percent = (energy_transfer / hot_basalt.energy()) * 100.0;

        println!("{:>8.0} years: {:.2e} J total ({:.3}%), {:.2e} J/year",
            years, energy_transfer, transfer_percent, transfer_per_year);
    }
    println!();

    // Experiment 3: Surface vs Bulk Transfer Comparison
    println!("Experiment 3: Surface vs Bulk Transfer Comparison");
    println!("--------------------------------------------------");
    println!("Materials: Silicate (2200K) → Basaltic (1400K), 1000 years\n");

    let layer_thicknesses = vec![10.0, 20.0, 50.0, 100.0];

    for &thickness_km in &layer_thicknesses {
        let volume_km3 = 85300.0 * thickness_km; // Typical Resolution 2 cell area
        let hot_silicate = create_sample_material(MaterialType::Silicate, 2200.0, volume_km3);
        let cold_basaltic = create_sample_material(MaterialType::Basaltic, 1400.0, volume_km3);

        let surface_transfer = hot_silicate.calculate_thermal_transfer(&cold_basaltic, 0.1, 1000.0);
        let bulk_transfer = hot_silicate.calculate_bulk_thermal_transfer(&cold_basaltic, thickness_km, 1000.0);
        let ratio = if surface_transfer != 0.0 { bulk_transfer / surface_transfer } else { 0.0 };

        // Calculate energy percentages
        let hot_layer_energy = hot_silicate.energy();
        let cold_layer_energy = cold_basaltic.energy();
        let surface_percent_hot = (surface_transfer / hot_layer_energy) * 100.0;
        let bulk_percent_hot = (bulk_transfer / hot_layer_energy) * 100.0;

        println!("{:>6.0}km layer (volume: {:.1e} km³):", thickness_km, volume_km3);
        println!("  Hot layer energy: {:.2e} J", hot_layer_energy);
        println!("  Cold layer energy: {:.2e} J", cold_layer_energy);
        println!("  Surface-based: {:.2e} J ({:.3}% of hot layer)", surface_transfer, surface_percent_hot);
        println!("  Bulk-based:    {:.2e} J ({:.3}% of hot layer)", bulk_transfer, bulk_percent_hot);
        println!("  Bulk/Surface ratio: {:.1}x", ratio);

        // Show height scaling effect
        let height_scale = thickness_km.sqrt();
        let height_scaled_surface = surface_transfer * height_scale;
        let height_scaled_percent = (height_scaled_surface / hot_layer_energy) * 100.0;
        println!("  Height-scaled surface: {:.2e} J ({:.3}% of hot layer, {:.2}x)",
            height_scaled_surface, height_scaled_percent, height_scale);
        println!();
    }

    // Experiment 4: Temperature difference effects
    println!("Experiment 4: Temperature Difference Effects");
    println!("--------------------------------------------");
    println!("Materials: Basaltic → Granitic, 1000 years, diffusion_rate=0.1\n");

    let temp_differences = vec![100.0, 250.0, 500.0, 750.0, 1000.0];
    
    for &temp_diff in &temp_differences {
        let hot_temp = 1500.0 + temp_diff;
        let cold_temp = 1500.0;

        let hot_sample = create_sample_material(MaterialType::Basaltic, hot_temp, 100.0);
        let cold_sample = create_sample_material(MaterialType::Granitic, cold_temp, 100.0);

        let energy_transfer = hot_sample.calculate_thermal_transfer(&cold_sample, 0.1, 1000.0);
        let transfer_per_kelvin = energy_transfer / temp_diff;
        let transfer_percent = (energy_transfer / hot_sample.energy()) * 100.0;

        println!("{:>6.0}K diff: {:.2e} J ({:.4}%), {:.2e} J/K",
            temp_diff, energy_transfer, transfer_percent, transfer_per_kelvin);
    }
    println!();

    // Experiment 5: Conductivity vs Transfer Rate
    println!("Experiment 5: Conductivity Impact Analysis");
    println!("------------------------------------------");
    println!("Fixed conditions: 500K temp diff, 1000 years, diffusion_rate=0.1\n");

    let materials = vec![
        MaterialType::Granitic,
        MaterialType::Basaltic,
        MaterialType::Silicate,
        MaterialType::Metallic,
    ];

    for material in &materials {
        let hot_sample = create_sample_material(*material, 2000.0, 100.0);
        let cold_sample = create_sample_material(*material, 1500.0, 100.0);

        let energy_transfer = hot_sample.calculate_thermal_transfer(&cold_sample, 0.1, 1000.0);
        let conductivity = hot_sample.thermal_conductivity();
        let efficiency = energy_transfer / conductivity;
        let transfer_percent = (energy_transfer / hot_sample.energy()) * 100.0;

        println!("{:?}:", material);
        println!("  Conductivity: {:.2} W/(m·K)", conductivity);
        println!("  Energy transfer: {:.2e} J ({:.4}%)", energy_transfer, transfer_percent);
        println!("  Transfer efficiency: {:.2e} J per W/(m·K)", efficiency);
        println!();
    }

    // Experiment 6: Space Radiation Energy Loss
    println!("Experiment 6: Space Radiation Energy Loss");
    println!("------------------------------------------");
    println!("Testing radiate_to_space method with different temperatures\n");

    let temperatures = vec![1200.0, 1500.0, 1800.0, 2100.0];
    let area_km2 = 85300.0; // Typical Resolution 2 cell area
    let years = 1000.0;

    for &temp_k in &temperatures {
        let mut hot_layer = create_sample_material(MaterialType::Silicate, temp_k, 100.0);
        let initial_energy = hot_layer.energy();

        let radiated_energy = hot_layer.radiate_to_space(area_km2, years);
        let radiation_percent = (radiated_energy / initial_energy) * 100.0;
        let final_temp = hot_layer.kelvin();
        let temp_drop = temp_k - final_temp;

        println!("{:>6.0}K layer:", temp_k);
        println!("  Initial energy: {:.2e} J", initial_energy);
        println!("  Radiated energy: {:.2e} J ({:.4}%)", radiated_energy, radiation_percent);
        println!("  Temperature drop: {:.1}K (final: {:.1}K)", temp_drop, final_temp);
        println!();
    }

    // Compare radiation vs thermal transfer
    println!("Radiation vs Thermal Transfer Comparison:");
    println!("-----------------------------------------");

    let hot_layer = create_sample_material(MaterialType::Silicate, 2000.0, 100.0);
    let cold_layer = create_sample_material(MaterialType::Basaltic, 1500.0, 100.0);

    // Thermal transfer between layers
    let thermal_transfer = hot_layer.calculate_thermal_transfer(&cold_layer, 0.1, 1000.0);
    let thermal_percent = (thermal_transfer / hot_layer.energy()) * 100.0;

    // Space radiation from hot layer
    let mut hot_layer_copy = create_sample_material(MaterialType::Silicate, 2000.0, 100.0);
    let space_radiation = hot_layer_copy.radiate_to_space(area_km2, 1000.0);
    let radiation_percent_copy = (space_radiation / hot_layer.energy()) * 100.0;

    println!("Hot Silicate layer (2000K) over 1000 years:");
    println!("  Thermal transfer to cold layer: {:.2e} J ({:.4}%)", thermal_transfer, thermal_percent);
    println!("  Space radiation: {:.2e} J ({:.4}%)", space_radiation, radiation_percent_copy);
    println!("  Radiation/Transfer ratio: {:.1}x", space_radiation / thermal_transfer);
    println!();
}

/// Helper function to create a sample material with specified properties
fn create_sample_material(material_type: MaterialType, temp_k: f64, volume_km3: f64) -> StandardEnergyMass {
    let mut sample = StandardEnergyMass::new_with_material(material_type, temp_k, volume_km3);
    sample
}

/// Helper function to calculate interface conductivity (harmonic mean)
fn calculate_interface_conductivity(k1: f64, k2: f64) -> f64 {
    2.0 * k1 * k2 / (k1 + k2)
}
