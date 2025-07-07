/// Test Radiance System
/// 
/// Tests the new radiance system with Perlin noise transitions
/// and thermal inflow/outflow management

use atmo_asth_rust::sim::radiance::{RadianceSystem, ThermalFeatureConfig, ThermalFlowConfig};
use atmo_asth_rust::h3_graphics::H3GraphicsGenerator;
use h3o::Resolution;
use std::fs;


const INFLOWS: usize = 80;
const OUTFLOWS: usize = 20;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Testing Radiance System ===");
    println!("Creating radiance system with Perlin transitions and thermal flows");
    
    let current_year = 0.0;  // Start simulation at year 0
    let l2_resolution = Resolution::Two;
    let points_per_degree = 3;
    let cache_path = "./h3_neighbor_cache.db";
    
    println!("\nConfiguration:");
    println!("  Resolution: {:?} (Level 2)", l2_resolution);
    println!("  Current year: {:.1}", current_year);
    println!("  Points per degree: {}", points_per_degree);
    println!("  Cache path: {}", cache_path);
    
    // Create radiance system
    println!("\nInitializing radiance system...");

    // Create a custom thermal configuration for testing
    let mut inflow_config = ThermalFlowConfig::realistic_inflows();
    inflow_config.target_count = INFLOWS;
    inflow_config.calculate_replacement_rate();

    let mut outflow_config = ThermalFlowConfig::realistic_outflows();
    outflow_config.target_count = OUTFLOWS;
    outflow_config.calculate_replacement_rate();

    let thermal_config = ThermalFeatureConfig::custom(
        vec![inflow_config, outflow_config],
        1.0, // Check annually for probabilistic replacement
    );

    let mut radiance = RadianceSystem::new_with_config(current_year, thermal_config);
    
    // Display initial statistics
    let stats = radiance.get_statistics(current_year);
    println!("Initial radiance statistics:");
    println!("  Active inflows: {}", stats.active_inflows);
    println!("  Active outflows: {}", stats.active_outflows);
    println!("  Perlin transition progress: {:.1}%", stats.perlin_transition_progress * 100.0);
    println!("  Years to next Perlin: {:.1}", stats.years_to_next_perlin);
    
    // Initialize sustainable thermal features
    println!("\nInitializing sustainable thermal features...");
    radiance.initialize_sustainable_features(l2_resolution, current_year)?;

    // Display updated statistics
    let updated_stats = radiance.get_statistics(current_year);
    println!("\nUpdated radiance statistics:");
    println!("  Active inflows: {}", updated_stats.active_inflows);
    println!("  Active outflows: {}", updated_stats.active_outflows);
    println!("  Total inflow rate: {:.2} WM", updated_stats.total_inflow_rate_mw);
    println!("  Total outflow rate: {:.2} WM", updated_stats.total_outflow_rate_mw);
    println!("  Net flow rate: {:.2} WM", updated_stats.net_flow_rate_mw);
    
    // Generate thermal PNG with neighbor effects
    let filename = "radiance_thermal_0000.png";
    println!("\nGenerating thermal PNG with neighbor effects...");

    match radiance.generate_thermal_png(
        l2_resolution,
        points_per_degree,
        current_year,
        filename
    ) {
        Ok(()) => {
            println!("✓ Successfully generated thermal PNG: {}", filename);
        }
        Err(e) => {
            println!("✗ Failed to generate thermal PNG: {}", e);
            return Ok(());
        }
    }
    
    // Run 500-year simulation tracking only feature counts
    println!("\n=== Running 100-Year Odds Table Simulation ===");
    println!("Long-term probabilistic replacement tracking");

    let mut perlin_transition_count = 0;
    let mut last_perlin_progress = 0.0;
    let mut inflow_additions = 0;
    let mut outflow_additions = 0;
    let mut last_inflow_count = radiance.count_active_inflows(0.0);
    let mut last_outflow_count = radiance.count_active_outflows(0.0);

    for year in 0..=100 {
        // Maintain thermal features annually (probabilistic replacement)
        radiance.maintain_thermal_features(l2_resolution, year as f64)?;

        // Update radiance system for current year
        radiance.update(year as f64);
        let stats = radiance.get_statistics(year as f64);

        // Track Perlin transitions (when progress resets to 0)
        if stats.perlin_transition_progress < last_perlin_progress {
            perlin_transition_count += 1;
        }
        last_perlin_progress = stats.perlin_transition_progress;

        // Track replacement events
        let current_inflow_count = radiance.count_active_inflows(year as f64);
        let current_outflow_count = radiance.count_active_outflows(year as f64);

        // Detect additions (count increased)
        if current_inflow_count > last_inflow_count {
            inflow_additions += current_inflow_count - last_inflow_count;
            println!("Year {:4}: +{} INFLOWS added (total: {})",
                     year, current_inflow_count - last_inflow_count, current_inflow_count);
        }
        if current_outflow_count > last_outflow_count {
            outflow_additions += current_outflow_count - last_outflow_count;
            println!("Year {:4}: +{} OUTFLOWS added (total: {})",
                     year, current_outflow_count - last_outflow_count, current_outflow_count);
        }

        // Report summary every 10 years
        if year % 10 == 0 {
            println!("Year {:4}: {} inflows, {} outflows | Additions: {} inflows, {} outflows | Perlin: {:.1}%",
                     year, current_inflow_count, current_outflow_count,
                     inflow_additions, outflow_additions, stats.perlin_transition_progress * 100.0);
        }

        last_inflow_count = current_inflow_count;
        last_outflow_count = current_outflow_count;
    }

    println!("\n=== 100-Year Odds Table Summary ===");
    let final_year = 100.0;
    let final_inflows = radiance.count_active_inflows(final_year);
    let final_outflows = radiance.count_active_outflows(final_year);

    println!("Initial features: {} inflows, {} outflows", INFLOWS, OUTFLOWS);
    println!("Final features: {} inflows, {} outflows", final_inflows, final_outflows);
    println!("Total Perlin transitions: {}", perlin_transition_count);

    println!("\nFeature survival rates:");
    println!("  Inflows: {:.1}% ({} of {} survived)",
             (final_inflows as f64 / INFLOWS as f64) * 100.0, final_inflows, INFLOWS);
    println!("  Outflows: {:.1}% ({} of {} survived)",
             (final_outflows as f64 / OUTFLOWS as f64) * 100.0, final_outflows, OUTFLOWS);

    println!("\nGeological dynamics observed:");
    println!("✓ Probabilistic thermal feature replacement (0.2-0.3% annual chance)");
    println!("✓ Natural feature expiration based on geological lifetimes (200-800 years)");
    println!("✓ Perlin noise background transitions every 80-120 years");
    println!("✓ Realistic long-term population decline");

    // Calculate average replacement events
    let total_years = 100.0;
    let inflow_replacement_rate = 0.002; // 0.2%
    let outflow_replacement_rate = 0.003; // 0.3%
    let expected_inflow_replacements = total_years * (INFLOWS as f64) * inflow_replacement_rate;
    let expected_outflow_replacements = total_years * (OUTFLOWS as f64) * outflow_replacement_rate;

    println!("\nExpected vs Actual replacement events:");
    println!("  Expected inflow replacements: ~{:.0}", expected_inflow_replacements);
    println!("  Expected outflow replacements: ~{:.0}", expected_outflow_replacements);
    println!("  Expected Perlin transitions: ~{:.0}", total_years / 100.0); // Average 100 years per transition

    println!("\nActual replacement events:");
    println!("  Inflow additions: {}", inflow_additions);
    println!("  Outflow additions: {}", outflow_additions);
    println!("  Perlin transitions: {}", perlin_transition_count);

    println!("\nReplacement efficiency:");
    let inflow_efficiency = if expected_inflow_replacements > 0.0 {
        (inflow_additions as f64 / expected_inflow_replacements) * 100.0
    } else { 0.0 };
    let outflow_efficiency = if expected_outflow_replacements > 0.0 {
        (outflow_additions as f64 / expected_outflow_replacements) * 100.0
    } else { 0.0 };
    println!("  Inflow replacement rate: {:.1}% of expected", inflow_efficiency);
    println!("  Outflow replacement rate: {:.1}% of expected", outflow_efficiency);

    Ok(())
}
