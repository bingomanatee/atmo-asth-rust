use std::fs::File;
use std::io::Write;

/// Lithosphere smoothing experiment
/// Shows how chunky temperature blocks (like our current layer system) smooth into realistic gradients
fn main() {
    println!("🔬 Lithosphere Cliff-to-Slope Smoothing Experiment");
    println!("==================================================");

    // Create chunky temperature blocks that simulate our current layer system problem
    let mut experiment = LithosphereSmoothing::new();
    experiment.create_varied_lithosphere_profile();

    println!("📊 Initial Setup:");
    println!("   {} small buckets (1km each)", experiment.small_buckets.len());
    println!("   Target: {} large buckets (4km each)", experiment.small_buckets.len() / 4);
    
    experiment.print_initial_profile();
    
    // Run smoothing simulation
    println!("\n🌡️  Running thermal smoothing simulation...");
    let years_per_step = 50.0;  // Shorter steps to see gradual changes
    let total_years = 2000.0;   // 2000 years total
    let steps = (total_years / years_per_step) as usize;
    
    experiment.run_smoothing_with_export(steps, years_per_step);
    
    println!("\n📈 Results exported to examples/data/thermal_experiment.csv");
    println!("   Each row shows how chunky temperature blocks evolve into smooth gradients");
    println!("   Watch artificial cliffs transform into realistic slopes!");
}

struct LithosphereSmoothing {
    small_buckets: Vec<ThermalBucket>,
    initial_buckets: Vec<ThermalBucket>,
}

#[derive(Debug, Clone)]
struct ThermalBucket {
    depth_km: f64,
    temperature_k: f64,
    thermal_conductivity: f64,
    density: f64,
    specific_heat: f64,
}

impl LithosphereSmoothing {
    fn new() -> Self {
        Self {
            small_buckets: Vec::new(),
            initial_buckets: Vec::new(),
        }
    }
    
    fn create_varied_lithosphere_profile(&mut self) {
        // Create 80 lithosphere buckets that simulate our current "chunky layer" problem
        // Start with big temperature blocks (like our current layer system) to show smoothing

        // Define 4 major "layers" with uniform temperatures (like our current system)
        let layer_definitions = [
            (0..20, 350.0),    // Surface layer: 0-20km at 350K
            (20..40, 800.0),   // Upper lithosphere: 20-40km at 800K
            (40..60, 1200.0),  // Mid lithosphere: 40-60km at 1200K
            (60..80, 1500.0),  // Deep lithosphere: 60-80km at 1500K
        ];

        for i in 0..80 {
            let depth = i as f64;

            // Find which "layer" this bucket belongs to and assign uniform temperature
            let mut layer_temp = 350.0; // Default
            for (range, temp) in &layer_definitions {
                if range.contains(&i) {
                    layer_temp = *temp;
                    break;
                }
            }

            // Add some small random variation within each layer (±5K) to simulate real material
            let small_variation = match i % 7 {
                0 => -3.0,
                1 => 2.0,
                2 => -1.0,
                3 => 4.0,
                4 => -2.0,
                5 => 1.0,
                _ => 0.0,
            };

            let final_temp = layer_temp + small_variation;

            self.small_buckets.push(ThermalBucket {
                depth_km: depth,
                temperature_k: final_temp,
                thermal_conductivity: 2.5, // Standard lithosphere conductivity
                density: 2900.0,
                specific_heat: 1000.0,
            });
        }

        self.initial_buckets = self.small_buckets.clone();

        println!("   Temperature range: {:.1}K to {:.1}K",
                 self.small_buckets.iter().map(|b| b.temperature_k).fold(f64::INFINITY, f64::min),
                 self.small_buckets.iter().map(|b| b.temperature_k).fold(f64::NEG_INFINITY, f64::max));

        // Show the chunky layer structure
        println!("   Layer structure (chunky blocks):");
        println!("     0-20km: ~350K (Surface layer)");
        println!("     20-40km: ~800K (Upper lithosphere)");
        println!("     40-60km: ~1200K (Mid lithosphere)");
        println!("     60-80km: ~1500K (Deep lithosphere)");
    }
    
    fn print_initial_profile(&self) {
        println!("\n🌡️  Initial Temperature Profile (every 10km):");
        for (i, bucket) in self.small_buckets.iter().enumerate() {
            if i % 10 == 0 {
                println!("   {}km: {:.1}K", bucket.depth_km, bucket.temperature_k);
            }
        }
    }
    
    fn run_smoothing_with_export(&mut self, steps: usize, years_per_step: f64) {
        let mut file = File::create("examples/data/thermal_experiment.csv").expect("Could not create file");
        
        // Write header - show every 4th bucket (4km spacing) to keep manageable
        write!(file, "step,years").unwrap();
        for i in (0..self.small_buckets.len()).step_by(4) {
            write!(file, ",temp_{}km", i).unwrap();
        }
        writeln!(file).unwrap();
        
        // Export initial state
        self.export_smoothing_row(&mut file, 0, 0.0);
        
        // Run smoothing simulation
        for step in 0..steps {
            self.smoothing_step(years_per_step);
            
            // Export every 2 steps to see gradual changes
            if step % 2 == 0 || step == steps - 1 {
                let years = (step + 1) as f64 * years_per_step;
                self.export_smoothing_row(&mut file, step + 1, years);
            }
            
            if step % 10 == 0 {
                let temp_range = self.calculate_temperature_range();
                let variation = self.calculate_temperature_variation();
                println!("   Step {}: Range={:.1}K, Variation={:.1}K", step, temp_range, variation);
            }
        }
        
        // Print final analysis
        self.print_final_analysis();
    }
    
    fn smoothing_step(&mut self, years: f64) {
        let mut new_temps = vec![0.0; self.small_buckets.len()];
        
        // Calculate thermal diffusion between adjacent buckets
        for i in 0..self.small_buckets.len() {
            let current_bucket = &self.small_buckets[i];
            let mut energy_change = 0.0;
            
            // Diffusion with bucket above (if exists)
            if i > 0 {
                let above_bucket = &self.small_buckets[i - 1];
                let temp_diff = above_bucket.temperature_k - current_bucket.temperature_k;
                let thermal_transfer = self.calculate_thermal_transfer(above_bucket, current_bucket, temp_diff, years);
                energy_change += thermal_transfer;
            }
            
            // Diffusion with bucket below (if exists)
            if i < self.small_buckets.len() - 1 {
                let below_bucket = &self.small_buckets[i + 1];
                let temp_diff = below_bucket.temperature_k - current_bucket.temperature_k;
                let thermal_transfer = self.calculate_thermal_transfer(below_bucket, current_bucket, temp_diff, years);
                energy_change += thermal_transfer;
            }
            
            // Convert energy change to temperature change
            let mass = current_bucket.density * 1000.0 * 1000.0 * 1.0; // 1km³ in kg
            let thermal_capacity = mass * current_bucket.specific_heat;
            let temp_change = energy_change / thermal_capacity;
            
            new_temps[i] = current_bucket.temperature_k + temp_change;
        }
        
        // Apply new temperatures
        for (i, &new_temp) in new_temps.iter().enumerate() {
            self.small_buckets[i].temperature_k = new_temp;
        }
    }
    
    fn calculate_thermal_transfer(&self, from_bucket: &ThermalBucket, to_bucket: &ThermalBucket, temp_diff: f64, years: f64) -> f64 {
        let avg_conductivity = (from_bucket.thermal_conductivity + to_bucket.thermal_conductivity) / 2.0;
        let distance = 1000.0; // 1km in meters
        let area = 1000.0 * 1000.0; // 1km² in m²
        let seconds = years * 365.25 * 24.0 * 3600.0;
        
        let heat_flux = avg_conductivity * temp_diff * area / distance;
        heat_flux * seconds * 0.005 // Scale down for stability and realistic rates
    }
    
    fn export_smoothing_row(&self, file: &mut File, step: usize, years: f64) {
        write!(file, "{},{:.0}", step, years).unwrap();
        
        // Export every 4th bucket (4km spacing)
        for i in (0..self.small_buckets.len()).step_by(4) {
            write!(file, ",{:.1}", self.small_buckets[i].temperature_k).unwrap();
        }
        writeln!(file).unwrap();
    }
    
    fn calculate_temperature_range(&self) -> f64 {
        let min_temp = self.small_buckets.iter().map(|b| b.temperature_k).fold(f64::INFINITY, f64::min);
        let max_temp = self.small_buckets.iter().map(|b| b.temperature_k).fold(f64::NEG_INFINITY, f64::max);
        max_temp - min_temp
    }
    
    fn calculate_temperature_variation(&self) -> f64 {
        // Calculate average temperature difference between adjacent buckets
        let mut total_diff = 0.0;
        for i in 1..self.small_buckets.len() {
            total_diff += (self.small_buckets[i].temperature_k - self.small_buckets[i-1].temperature_k).abs();
        }
        total_diff / (self.small_buckets.len() - 1) as f64
    }
    
    fn print_final_analysis(&self) {
        println!("\n📊 Final Analysis:");
        println!("   Initial temperature range: {:.1}K", 
                 self.initial_buckets.iter().map(|b| b.temperature_k).fold(f64::NEG_INFINITY, f64::max) - 
                 self.initial_buckets.iter().map(|b| b.temperature_k).fold(f64::INFINITY, f64::min));
        println!("   Final temperature range: {:.1}K", self.calculate_temperature_range());
        
        let initial_variation = self.calculate_initial_variation();
        let final_variation = self.calculate_temperature_variation();
        println!("   Initial avg variation: {:.1}K per km", initial_variation);
        println!("   Final avg variation: {:.1}K per km", final_variation);
        println!("   Smoothing factor: {:.1}x", initial_variation / final_variation);
        
        println!("\n🌡️  Final Temperature Profile (every 10km):");
        for (i, bucket) in self.small_buckets.iter().enumerate() {
            if i % 10 == 0 {
                let initial_temp = self.initial_buckets[i].temperature_k;
                let change = bucket.temperature_k - initial_temp;
                println!("   {}km: {:.1}K (Δ{:+.1}K)", bucket.depth_km, bucket.temperature_k, change);
            }
        }
    }
    
    fn calculate_initial_variation(&self) -> f64 {
        let mut total_diff = 0.0;
        for i in 1..self.initial_buckets.len() {
            total_diff += (self.initial_buckets[i].temperature_k - self.initial_buckets[i-1].temperature_k).abs();
        }
        total_diff / (self.initial_buckets.len() - 1) as f64
    }
}
