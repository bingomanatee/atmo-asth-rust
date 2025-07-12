/// Heat Plume Operation v2 - Entity-Based System
/// 
/// This operation manages heat plumes as separate spanning entities that transport
/// energy and material from foundry layers to the surface. Each plume persists
/// across simulation steps and has baked-in effects on the layers it traverses.

use crate::sim_op::SimOp;
use crate::sim::simulation::Simulation;
use std::any::Any;
use crate::global_thermal::sim_cell::SimCell;
use crate::global_thermal::heat_plume::PlumeStats;
use crate::energy_mass_composite::EnergyMassComposite;
use rayon::prelude::*;
use h3o::CellIndex;

pub struct HeatPlumeOp {
    /// Temperature threshold for plume initiation (K)
    plume_threshold_temp: f64,
    
    /// Enable detailed plume reporting
    enable_reporting: bool,
    
    /// Performance statistics
    performance_stats: PlumePerformanceStats,
    
    /// Global plume statistics for monitoring
    global_plume_stats: GlobalPlumeStats,
}

#[derive(Default)]
struct PlumePerformanceStats {
    total_plumes: usize,
    active_plumes: usize,
    new_plumes_created: usize,
    plumes_dissipated: usize,
    total_energy_transported: f64,
    total_material_transported: f64,
    cells_with_plumes: usize,
    computation_time_ms: f64,
}

#[derive(Default)]
struct GlobalPlumeStats {
    hottest_plume_temp: f64,
    highest_plume_energy: f64,
    deepest_plume_origin: usize,
    shallowest_plume_position: usize,
    average_plume_velocity: f64,
    total_plume_energy: f64,
}

impl HeatPlumeOp {
    pub fn new() -> Self {
        Self {
            plume_threshold_temp: 1800.0, // 1800K threshold for plume initiation
            enable_reporting: false,
            performance_stats: PlumePerformanceStats::default(),
            global_plume_stats: GlobalPlumeStats::default(),
        }
    }
    
    pub fn with_threshold(threshold_temp: f64) -> Self {
        Self {
            plume_threshold_temp: threshold_temp,
            enable_reporting: false,
            performance_stats: PlumePerformanceStats::default(),
            global_plume_stats: GlobalPlumeStats::default(),
        }
    }
    
    pub fn with_reporting(mut self, enabled: bool) -> Self {
        self.enable_reporting = enabled;
        self
    }
    
    /// Process all plumes in a single cell
    fn process_cell_plumes(&self, cell: &mut SimCell, years_per_step: u32) -> CellPlumeProcessingResult {
        let mut result = CellPlumeProcessingResult::default();
        
        // Update existing plumes
        let plumes_before = cell.plumes.plumes.len();
        cell.plumes.update_all_plumes(&cell.layers_t, years_per_step);
        let plumes_after = cell.plumes.plumes.len();
        
        result.plumes_dissipated = plumes_before.saturating_sub(plumes_after);
        
        // Check for new plume creation from foundry layers
        let plumes_before_creation = cell.plumes.plumes.len();
        cell.plumes.check_and_create_plumes(&cell.layers_t, self.plume_threshold_temp);
        let plumes_after_creation = cell.plumes.plumes.len();
        
        result.new_plumes_created = plumes_after_creation.saturating_sub(plumes_before_creation);
        result.active_plumes = plumes_after_creation;
        
        // Apply plume effects to layers
        let energy_deposits = cell.plumes.get_total_energy_deposits();
        let material_deposits = cell.plumes.get_total_material_deposits();
        
        // Bake plume effects into layer next states
        for (layer_idx, energy_deposit) in energy_deposits {
            if let Some((_, next_layer)) = cell.layers_t.get_mut(layer_idx) {
                next_layer.energy_mass.energy_joules += energy_deposit;
                result.total_energy_transported += energy_deposit;
            }
        }
        
        for (layer_idx, material_deposit) in material_deposits {
            if let Some((_, next_layer)) = cell.layers_t.get_mut(layer_idx) {
                // Add material mass to the layer
                let current_mass = next_layer.mass_kg();
                let new_mass = current_mass + material_deposit;
                
                // Update energy to maintain temperature while adding mass
                let current_temp = next_layer.temperature_k();
                let specific_heat = next_layer.energy_mass.specific_heat_j_kg_k();
                let new_energy = new_mass * specific_heat * current_temp;
                
                next_layer.energy_mass.energy_joules = new_energy;
                result.total_material_transported += material_deposit;
            }
        }
        
        // Collect plume statistics
        result.plume_stats = cell.plumes.get_all_stats();
        
        result
    }
    
    /// Update global plume statistics
    fn update_global_stats(&mut self, cell_results: &[CellPlumeProcessingResult]) {
        self.global_plume_stats = GlobalPlumeStats::default();
        
        for result in cell_results {
            for plume_stat in &result.plume_stats {
                // Update hottest plume
                if plume_stat.temperature_k > self.global_plume_stats.hottest_plume_temp {
                    self.global_plume_stats.hottest_plume_temp = plume_stat.temperature_k;
                }
                
                // Update highest energy plume
                if plume_stat.energy_joules > self.global_plume_stats.highest_plume_energy {
                    self.global_plume_stats.highest_plume_energy = plume_stat.energy_joules;
                }
                
                // Update deepest plume origin
                if plume_stat.current_layer > self.global_plume_stats.deepest_plume_origin {
                    self.global_plume_stats.deepest_plume_origin = plume_stat.current_layer;
                }
                
                // Update shallowest plume position
                if self.global_plume_stats.shallowest_plume_position == 0 || 
                   plume_stat.current_layer < self.global_plume_stats.shallowest_plume_position {
                    self.global_plume_stats.shallowest_plume_position = plume_stat.current_layer;
                }
                
                // Accumulate for averages
                self.global_plume_stats.average_plume_velocity += plume_stat.velocity_m_s;
                self.global_plume_stats.total_plume_energy += plume_stat.energy_joules;
            }
        }
        
        // Calculate averages
        let total_plumes = self.performance_stats.active_plumes;
        if total_plumes > 0 {
            self.global_plume_stats.average_plume_velocity /= total_plumes as f64;
        }
    }
    
    /// Generate detailed plume report
    fn generate_plume_report(&self, step: i32) {
        if !self.enable_reporting {
            return;
        }
        
        println!("🌋 === Heat Plume Report (Step {}) ===", step);
        println!("📊 Performance Stats:");
        println!("   - Total plumes: {}", self.performance_stats.total_plumes);
        println!("   - Active plumes: {}", self.performance_stats.active_plumes);
        println!("   - New plumes created: {}", self.performance_stats.new_plumes_created);
        println!("   - Plumes dissipated: {}", self.performance_stats.plumes_dissipated);
        println!("   - Cells with plumes: {}", self.performance_stats.cells_with_plumes);
        println!("   - Total energy transported: {:.2e} J", self.performance_stats.total_energy_transported);
        println!("   - Total material transported: {:.2e} kg", self.performance_stats.total_material_transported);
        println!("   - Computation time: {:.2} ms", self.performance_stats.computation_time_ms);
        
        println!("🌡️  Global Plume Stats:");
        println!("   - Hottest plume temperature: {:.1}K", self.global_plume_stats.hottest_plume_temp);
        println!("   - Highest plume energy: {:.2e} J", self.global_plume_stats.highest_plume_energy);
        println!("   - Deepest plume origin: Layer {}", self.global_plume_stats.deepest_plume_origin);
        println!("   - Shallowest plume position: Layer {}", self.global_plume_stats.shallowest_plume_position);
        println!("   - Average plume velocity: {:.2} m/s", self.global_plume_stats.average_plume_velocity);
        println!("   - Total plume energy: {:.2e} J", self.global_plume_stats.total_plume_energy);
        println!();
    }
}

#[derive(Default)]
struct CellPlumeProcessingResult {
    active_plumes: usize,
    new_plumes_created: usize,
    plumes_dissipated: usize,
    total_energy_transported: f64,
    total_material_transported: f64,
    plume_stats: Vec<PlumeStats>,
}

impl SimOp for HeatPlumeOp {
    fn name(&self) -> &str {
        "HeatPlumeV2"
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
    
    fn init_sim(&mut self, _sim: &mut Simulation) {
        println!("🌋 Heat Plume Operation v2 initialized");
        println!("   - Plume threshold temperature: {:.1}K", self.plume_threshold_temp);
        println!("   - Detailed reporting: {}", self.enable_reporting);
        println!("   - Entity-based plume system active");
    }
    
    fn update_sim(&mut self, sim: &mut Simulation) {
        let start_time = std::time::Instant::now();
        
        // Reset performance stats
        self.performance_stats = PlumePerformanceStats::default();
        
        // Convert to vector for parallel processing
        let mut cell_data: Vec<(CellIndex, SimCell)> = sim.cells.drain().collect();
        
        // Process cells in parallel
        let cell_results: Vec<CellPlumeProcessingResult> = cell_data.par_iter_mut()
            .map(|(_, cell)| {
                self.process_cell_plumes(cell, sim.years_per_step)
            })
            .collect();
        
        // Reconstruct HashMap
        sim.cells = cell_data.into_iter().collect();
        
        // Aggregate results
        for result in &cell_results {
            self.performance_stats.total_plumes += result.active_plumes;
            self.performance_stats.active_plumes += result.active_plumes;
            self.performance_stats.new_plumes_created += result.new_plumes_created;
            self.performance_stats.plumes_dissipated += result.plumes_dissipated;
            self.performance_stats.total_energy_transported += result.total_energy_transported;
            self.performance_stats.total_material_transported += result.total_material_transported;
            
            if result.active_plumes > 0 {
                self.performance_stats.cells_with_plumes += 1;
            }
        }
        
        // Update global statistics
        self.update_global_stats(&cell_results);
        
        // Record computation time
        self.performance_stats.computation_time_ms = start_time.elapsed().as_millis() as f64;
        
        // Generate periodic reports
        if sim.step % 20 == 0 {
            self.generate_plume_report(sim.step);
        }
        
        // Log summary every 100 steps with aggressive heat dumping info
        if sim.step % 100 == 0 && self.performance_stats.active_plumes > 0 {
            let avg_energy_per_plume = if self.performance_stats.active_plumes > 0 {
                self.performance_stats.total_energy_transported / self.performance_stats.active_plumes as f64
            } else {
                0.0
            };
            
            println!("🌋 Heat Plume Summary (Step {}): {} active plumes, {:.2e}J transported ({:.2e}J/plume avg), {:.1}ms", 
                sim.step,
                self.performance_stats.active_plumes,
                self.performance_stats.total_energy_transported,
                avg_energy_per_plume,
                self.performance_stats.computation_time_ms);
                
            // Show hottest plume info for temperature differential context
            if self.global_plume_stats.hottest_plume_temp > 0.0 {
                println!("🔥 Hottest plume: {:.1}K, Highest energy: {:.2e}J, Deepest origin: Layer {}", 
                    self.global_plume_stats.hottest_plume_temp,
                    self.global_plume_stats.highest_plume_energy,
                    self.global_plume_stats.deepest_plume_origin);
            }
        }
    }
    
    fn after_sim(&mut self, _sim: &mut Simulation) {
        println!("🌋 Heat Plume Operation v2 completed");
        println!("   - Final active plumes: {}", self.performance_stats.active_plumes);
        println!("   - Total energy transported: {:.2e} J", self.performance_stats.total_energy_transported);
        println!("   - Total material transported: {:.2e} kg", self.performance_stats.total_material_transported);
        println!("   - Cells with plumes: {}", self.performance_stats.cells_with_plumes);
    }
}

impl Default for HeatPlumeOp {
    fn default() -> Self {
        Self::new()
    }
}