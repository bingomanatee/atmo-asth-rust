// Example demonstrating the new unified simulation operation architecture
// This shows how to create operations that use all three lifecycle methods

use atmo_asth_rust::sim::{Simulation, SimProps};
use atmo_asth_rust::sim::sim_op::{SimOp, SimOpHandle};
use atmo_asth_rust::planet::Planet;
use atmo_asth_rust::constants::{ASTHENOSPHERE_SURFACE_START_TEMP_K, EARTH_RADIUS_KM};
use h3o::Resolution;

/// Example operation that demonstrates all three lifecycle methods
pub struct ExampleLifecycleOp {
    pub name: String,
    pub init_count: usize,
    pub update_count: usize,
    pub total_energy_start: f64,
    pub total_energy_end: f64,
}

impl ExampleLifecycleOp {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            init_count: 0,
            update_count: 0,
            total_energy_start: 0.0,
            total_energy_end: 0.0,
        }
    }

    pub fn handle(name: &str) -> SimOpHandle {
        SimOpHandle::new(Box::new(Self::new(name)))
    }
}

impl SimOp for ExampleLifecycleOp {
    fn init_sim(&mut self, sim: &mut Simulation) {
        self.init_count += 1;
        
        // Calculate initial total energy
        self.total_energy_start = sim.cells.values()
            .flat_map(|column| &column.layers)
            .map(|layer| layer.energy_joules())
            .sum();
            
        println!("🚀 [{}] Initializing simulation with {} cells", 
                 self.name, sim.cells.len());
        println!("   Initial total energy: {:.2e} J", self.total_energy_start);
    }

    fn update_sim(&mut self, sim: &mut Simulation) {
        self.update_count += 1;
        
        // Example: Apply a small cooling effect each step
        for column in sim.cells.values_mut() {
            for layer in &mut column.layers_next {
                let current_energy = layer.energy_joules();
                layer.set_energy_joules(current_energy * 0.999); // 0.1% cooling per step
            }
        }
        
        if self.update_count % 100 == 0 {
            let current_energy: f64 = sim.cells.values()
                .flat_map(|column| &column.layers_next)
                .map(|layer| layer.energy_joules())
                .sum();
            println!("   [{}] Step {}: Current energy: {:.2e} J", 
                     self.name, self.update_count, current_energy);
        }
    }

    fn after_sim(&mut self, sim: &mut Simulation) {
        // Calculate final total energy
        self.total_energy_end = sim.cells.values()
            .flat_map(|column| &column.layers)
            .map(|layer| layer.energy_joules())
            .sum();
            
        let energy_loss = self.total_energy_start - self.total_energy_end;
        let energy_loss_percent = (energy_loss / self.total_energy_start) * 100.0;
        
        println!("🏁 [{}] Simulation completed!", self.name);
        println!("   Total steps: {}", self.update_count);
        println!("   Final total energy: {:.2e} J", self.total_energy_end);
        println!("   Energy lost: {:.2e} J ({:.2}%)", energy_loss, energy_loss_percent);
    }
}

/// Another example operation that only uses update_sim
pub struct SimpleHeatingOp {
    pub heating_factor: f64,
}

impl SimpleHeatingOp {
    pub fn new(heating_factor: f64) -> Self {
        Self { heating_factor }
    }

    pub fn handle(heating_factor: f64) -> SimOpHandle {
        SimOpHandle::new(Box::new(Self::new(heating_factor)))
    }
}

impl SimOp for SimpleHeatingOp {
    // Only implementing update_sim - init_sim and after_sim use default (do nothing)
    fn update_sim(&mut self, sim: &mut Simulation) {
        for column in sim.cells.values_mut() {
            for layer in &mut column.layers_next {
                let current_energy = layer.energy_joules();
                layer.set_energy_joules(current_energy * self.heating_factor);
            }
        }
    }
}

fn main() {
    println!("🌍 Unified Simulation Operations Example");
    println!("========================================");
    
    // Create a simulation with multiple operations that use different lifecycle methods
    let mut sim = Simulation::new(SimProps {
        name: "unified_ops_demo",
        planet: Planet {
            radius_km: EARTH_RADIUS_KM as f64,
            resolution: Resolution::One,
        },
        ops: vec![
            ExampleLifecycleOp::handle("Energy Monitor"),
            SimpleHeatingOp::handle(1.001), // 0.1% heating per step
            ExampleLifecycleOp::handle("Secondary Monitor"),
        ],
        res: Resolution::Two,
        layer_count: 4,
        layer_height: 10.0,
        layer_height_km: 10.0,
        sim_steps: 500,
        years_per_step: 1_000_000,
        debug: false,
        alert_freq: 100,
        starting_surface_temp_k: ASTHENOSPHERE_SURFACE_START_TEMP_K
    });

    println!("\n🔄 Running simulation...");
    sim.simulate();
    
    println!("\n✅ Example completed!");
    println!("\nKey Benefits of the Unified Architecture:");
    println!("• Single array of operations instead of separate init/update/end arrays");
    println!("• Each operation can implement any combination of lifecycle methods");
    println!("• Default implementations mean you only implement what you need");
    println!("• Cleaner, more flexible simulation architecture");
}
