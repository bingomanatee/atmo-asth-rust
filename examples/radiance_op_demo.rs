/// Demonstration of the RadianceOp integration with the radiance system
/// Shows how radiance energy is applied to the bottom asthenosphere layers

use atmo_asth_rust::planet::Planet;
use atmo_asth_rust::global_thermal::global_h3_cell::{GlobalH3Cell, GlobalH3CellConfig};
use atmo_asth_rust::sim_op::radiance_op::{RadianceOp, RadianceOpParams};
use atmo_asth_rust::sim::radiance::RadianceSystem;
use atmo_asth_rust::h3_utils::H3Utils;
use atmo_asth_rust::energy_mass_composite::EnergyMassComposite;
use h3o::{Resolution, CellIndex};
use std::rc::Rc;
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌋 RadianceOp Demo: Integrating Radiance System with Asthenosphere Energy");
    println!("{}", "=".repeat(70));

    // Create Earth planet with L2 resolution
    let planet = Rc::new(Planet::earth(Resolution::Two));
    println!("🌍 Created Earth planet with radius: {:.1} km", planet.radius_km);

    // Create a few test cells
    let mut cells = HashMap::new();
    let cell_indices: Vec<CellIndex> = CellIndex::base_cells()
        .take(3) // Just take first 3 base cells for demo
        .flat_map(|base| base.children(Resolution::Two).take(2)) // 2 children each
        .collect();

    println!("📍 Creating {} test cells...", cell_indices.len());

    for cell_index in &cell_indices {
        let config = GlobalH3CellConfig::new_earth_like(*cell_index, planet.clone());
        let cell = GlobalH3Cell::new_with_schedule(*cell_index, planet.clone(), &config.layer_schedule);
        cells.insert(*cell_index, cell);
    }

    // Create radiance system with realistic baseline configuration
    let mut radiance_system = RadianceSystem::new(0.0);
    
    // Initialize with sustainable thermal features
    println!("🔥 Initializing radiance system with thermal features...");
    radiance_system.initialize_sustainable_features(Resolution::Two, 0.0)?;

    // Display radiance system statistics
    let stats = radiance_system.get_statistics(0.0);
    println!("📊 Radiance System Statistics:");
    println!("   - Active inflows: {}", stats.active_inflows);
    println!("   - Active outflows: {}", stats.active_outflows);
    println!("   - Total inflow rate: {:.2} MW", stats.total_inflow_rate_mw);
    println!("   - Total outflow rate: {:.2} MW", stats.total_outflow_rate_mw);
    println!("   - Net flow rate: {:.2} MW", stats.net_flow_rate_mw);
    println!();

    // Create radiance operation with custom parameters
    let radiance_params = RadianceOpParams {
        base_core_radiance_j_per_km2_per_year: 2.52e12, // Earth's core radiance
        radiance_system_multiplier: 1.0, // Full radiance system contribution
        enable_reporting: true,
    };

    let mut radiance_op = RadianceOp::new(radiance_params, radiance_system);

    // Record initial energies of bottom layers
    println!("🔍 Recording initial bottom layer energies...");
    let mut initial_energies = HashMap::new();
    for (cell_index, cell) in &cells {
        // Find bottom asthenosphere layer
        if let Some(bottom_layer_index) = find_bottom_asthenosphere_layer(cell) {
            let initial_energy = cell.layers_t[bottom_layer_index].0.energy_mass.energy();
            initial_energies.insert(*cell_index, initial_energy);
            println!("   Cell {:?}: {:.2e} J", cell_index, initial_energy);
        }
    }
    println!();

    // Apply radiance operation for 1000 years
    let time_years = 1000.0;
    let current_year = 0.0;
    
    println!("⚡ Applying RadianceOp for {} years...", time_years);
    radiance_op.apply(&mut cells, time_years, current_year);

    println!("📈 RadianceOp Results:");
    println!("   - Total energy added: {:.2e} J", radiance_op.total_energy_added());
    println!("   - Cells processed: {}", radiance_op.cells_processed());
    println!();

    // Compare final energies
    println!("📊 Energy Changes in Bottom Asthenosphere Layers:");
    println!("{:<20} {:<15} {:<15} {:<15} {:<10}", "Cell Index", "Initial (J)", "Final (J)", "Added (J)", "% Increase");
    println!("{}", "-".repeat(85));

    for (cell_index, cell) in &cells {
        if let Some(bottom_layer_index) = find_bottom_asthenosphere_layer(cell) {
            let initial_energy = initial_energies[cell_index];
            let final_energy = cell.layers_t[bottom_layer_index].1.energy_mass.energy();
            let energy_added = final_energy - initial_energy;
            let percent_increase = (energy_added / initial_energy) * 100.0;

            println!("{:<20} {:<15.2e} {:<15.2e} {:<15.2e} {:<10.2}%", 
                format!("{:?}", cell_index)[..16].to_string(),
                initial_energy,
                final_energy,
                energy_added,
                percent_increase
            );
        }
    }
    println!();

    // Show radiance system contribution breakdown
    println!("🔬 Radiance System Analysis:");
    for (cell_index, _cell) in &cells {
        let lat_lng = h3o::LatLng::from(*cell_index);
        let lat = lat_lng.lat_radians().to_degrees();
        let lng = lat_lng.lng_radians().to_degrees();

        // Get neighbors for thermal flow calculations
        let neighbors = H3Utils::neighbors_for(*cell_index);

        // Calculate radiance system energy per km² per year
        let radiance_energy_per_km2_per_year = radiance_op.radiance_system().calculate_cell_energy_with_neighbors(
            *cell_index, current_year, &neighbors
        );
        
        // Calculate surface area
        let surface_area_km2 = H3Utils::cell_area(planet.resolution, planet.radius_km);
        
        // Calculate total radiance system contribution
        let radiance_contribution = radiance_energy_per_km2_per_year * surface_area_km2 * time_years;
        
        // Calculate base core radiance
        let base_core_radiance = 2.52e12 * surface_area_km2 * time_years;
        
        println!("   Cell {:?}:", format!("{:?}", cell_index)[..16].to_string());
        println!("     - Lat/Lng: {:.2}°, {:.2}°", lat, lng);
        println!("     - Surface area: {:.1} km²", surface_area_km2);
        println!("     - Base core radiance: {:.2e} J", base_core_radiance);
        println!("     - Radiance system: {:.2e} J", radiance_contribution);
        println!("     - Total: {:.2e} J", base_core_radiance + radiance_contribution);
        println!();
    }

    println!("✅ RadianceOp demo completed successfully!");
    println!("💡 The radiance system successfully integrates Perlin noise and thermal flows");
    println!("   to provide realistic upward energy for asthenosphere root layers.");

    Ok(())
}

/// Find the index of the bottom asthenosphere layer (deepest non-atmospheric layer)
fn find_bottom_asthenosphere_layer(cell: &GlobalH3Cell) -> Option<usize> {
    // Find the deepest layer that is not atmospheric (depth >= 0)
    for (index, (current_layer, _)) in cell.layers_t.iter().enumerate().rev() {
        if current_layer.start_depth_km >= 0.0 {
            return Some(index);
        }
    }
    None
}
