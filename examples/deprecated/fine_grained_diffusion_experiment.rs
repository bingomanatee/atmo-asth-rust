use std::fs::File;
use std::io::Write;

/// Fine-grained thermal diffusion experiment
/// Takes a sample state from our simulation and runs detailed 1km bucket diffusion
fn main() {
    println!("🔬 Fine-Grained Thermal Diffusion Experiment");
    println!("============================================");
    
    // Sample data from CSV row 100 (step 490000, 2.45M years)
    // Surface: 427.5K, Asth: [1614.5, 1618.1, 1620.1], Lith: [389.2, 9.4, 26.8, 7.7, 9.2, 27.3]
    let sample_state = SampleState {
        surface_temp_k: 427.5,
        asthenosphere_temps: vec![1614.5, 1618.1, 1620.1],
        lithosphere_temps: vec![389.2, 9.4, 26.8, 7.7, 9.2, 27.3],
        lithosphere_heights: vec![50.0, 20.0, 30.0, 15.0, 18.0, 35.0], // Estimated heights
        asthenosphere_height_per_layer: 50.0,
    };
    
    println!("📊 Sample State (Step 490000):");
    println!("   Surface: {:.1}K", sample_state.surface_temp_k);
    println!("   Asthenosphere: {:?}K", sample_state.asthenosphere_temps);
    println!("   Lithosphere: {:?}K", sample_state.lithosphere_temps);
    println!("   Lithosphere Heights: {:?}km", sample_state.lithosphere_heights);
    
    // Create fine-grained buckets (1km each)
    let mut experiment = FineGrainedDiffusion::new(sample_state);
    
    println!("\n🧪 Creating 1km thermal buckets...");
    experiment.create_buckets();
    
    println!("   Total buckets: {}", experiment.buckets.len());
    println!("   Depth range: 0km to {}km", experiment.total_depth_km);
    
    // Run diffusion simulation with step-by-step export
    println!("\n🌡️  Running fine-grained thermal diffusion...");
    let years_per_step = 100.0; // 100-year substeps
    let total_years = 5000.0;   // 5000 years total
    let steps = (total_years / years_per_step) as usize;

    experiment.run_diffusion_with_evolution_export(steps, years_per_step);

    // Also export final state comparison
    experiment.export_results("examples/data/thermal_final.csv");

    println!("\n📈 Results exported:");
    println!("   - examples/data/thermal_experiment.csv (thermal evolution over time)");
    println!("   - examples/data/thermal_final.csv (initial vs final comparison)");
    println!("   Each column represents 4km-averaged temperature");
    println!("   Look for gradual slope formation between cliffs and plains!");
}

#[derive(Debug, Clone)]
struct SampleState {
    surface_temp_k: f64,
    asthenosphere_temps: Vec<f64>,
    lithosphere_temps: Vec<f64>,
    lithosphere_heights: Vec<f64>,
    asthenosphere_height_per_layer: f64,
}

#[derive(Debug, Clone)]
struct ThermalBucket {
    depth_km: f64,
    height_km: f64,
    temperature_k: f64,
    material_type: BucketMaterial,
    thermal_conductivity: f64,
    density: f64,
    specific_heat: f64,
}

#[derive(Debug, Clone)]
enum BucketMaterial {
    Surface,
    Lithosphere,
    Asthenosphere,
}

struct FineGrainedDiffusion {
    sample_state: SampleState,
    buckets: Vec<ThermalBucket>,
    total_depth_km: f64,
    initial_buckets: Vec<ThermalBucket>, // Store initial state for comparison
}

impl FineGrainedDiffusion {
    fn new(sample_state: SampleState) -> Self {
        Self {
            sample_state,
            buckets: Vec::new(),
            total_depth_km: 0.0,
            initial_buckets: Vec::new(),
        }
    }
    
    fn create_buckets(&mut self) {
        let mut current_depth = 0.0;
        
        // Surface bucket (1km)
        self.buckets.push(ThermalBucket {
            depth_km: current_depth,
            height_km: 1.0,
            temperature_k: self.sample_state.surface_temp_k,
            material_type: BucketMaterial::Surface,
            thermal_conductivity: 2.0, // Surface material
            density: 2500.0,
            specific_heat: 1000.0,
        });
        current_depth += 1.0;
        
        // Lithosphere buckets (1km each)
        for (i, &layer_height) in self.sample_state.lithosphere_heights.iter().enumerate() {
            let layer_temp = self.sample_state.lithosphere_temps[i];
            let buckets_in_layer = (layer_height as usize).max(1);
            
            for bucket_idx in 0..buckets_in_layer {
                self.buckets.push(ThermalBucket {
                    depth_km: current_depth,
                    height_km: 1.0,
                    temperature_k: layer_temp, // Start with uniform temperature per layer
                    material_type: BucketMaterial::Lithosphere,
                    thermal_conductivity: 2.5, // Lithosphere conductivity
                    density: 2900.0,
                    specific_heat: 1000.0,
                });
                current_depth += 1.0;
            }
        }
        
        // Asthenosphere buckets (1km each)
        for (i, &layer_temp) in self.sample_state.asthenosphere_temps.iter().enumerate() {
            let buckets_in_layer = (self.sample_state.asthenosphere_height_per_layer as usize).max(1);
            
            for bucket_idx in 0..buckets_in_layer {
                self.buckets.push(ThermalBucket {
                    depth_km: current_depth,
                    height_km: 1.0,
                    temperature_k: layer_temp,
                    material_type: BucketMaterial::Asthenosphere,
                    thermal_conductivity: 2.5, // Asthenosphere conductivity (could be higher for convection)
                    density: 3200.0,
                    specific_heat: 1300.0,
                });
                current_depth += 1.0;
            }
        }
        
        self.total_depth_km = current_depth;
        self.initial_buckets = self.buckets.clone(); // Store initial state
        
        println!("   Lithosphere buckets: {}", 
                 self.buckets.iter().filter(|b| matches!(b.material_type, BucketMaterial::Lithosphere)).count());
        println!("   Asthenosphere buckets: {}", 
                 self.buckets.iter().filter(|b| matches!(b.material_type, BucketMaterial::Asthenosphere)).count());
    }
    
    fn run_diffusion(&mut self, steps: usize, years_per_step: f64) {
        for step in 0..steps {
            self.diffusion_step(years_per_step);

            if step % 10 == 0 {
                let surface_temp = self.buckets[0].temperature_k;
                let deep_temp = self.buckets.last().unwrap().temperature_k;
                println!("   Step {}: Surface={:.1}K, Deep={:.1}K", step, surface_temp, deep_temp);
            }
        }
    }

    fn run_diffusion_with_evolution_export(&mut self, steps: usize, years_per_step: f64) {
        // Create evolution CSV with step-by-step thermal profiles
        let mut file = File::create("../data/thermal_experiment.csv").expect("Could not create file");

        // Write header with depth columns (every 4km)
        write!(file, "step,years").unwrap();
        let mut depth = 0.0;
        while depth < self.total_depth_km {
            write!(file, ",temp_{}km", depth as i32).unwrap();
            depth += 4.0;
        }
        writeln!(file).unwrap();

        // Export initial state
        self.export_evolution_row(&mut file, 0, 0.0);

        // Run diffusion and export every few steps
        for step in 0..steps {
            self.diffusion_step(years_per_step);

            // Export every 5 steps to keep file manageable
            if step % 5 == 0 || step == steps - 1 {
                let years = (step + 1) as f64 * years_per_step;
                self.export_evolution_row(&mut file, step + 1, years);
            }

            if step % 10 == 0 {
                let surface_temp = self.buckets[0].temperature_k;
                let deep_temp = self.buckets.last().unwrap().temperature_k;
                println!("   Step {}: Surface={:.1}K, Deep={:.1}K", step, surface_temp, deep_temp);
            }
        }
    }

    fn export_evolution_row(&self, file: &mut File, step: usize, years: f64) {
        write!(file, "{},{:.0}", step, years).unwrap();

        // Export 4km-averaged temperatures
        let mut i = 0;
        while i < self.buckets.len() {
            let end_idx = (i + 4).min(self.buckets.len());
            let chunk = &self.buckets[i..end_idx];
            let avg_temp = chunk.iter().map(|b| b.temperature_k).sum::<f64>() / chunk.len() as f64;
            write!(file, ",{:.1}", avg_temp).unwrap();
            i += 4;
        }
        writeln!(file).unwrap();
    }
    
    fn diffusion_step(&mut self, years: f64) {
        let mut new_temps = vec![0.0; self.buckets.len()];
        
        // Calculate thermal diffusion between adjacent buckets
        for i in 0..self.buckets.len() {
            let current_bucket = &self.buckets[i];
            let mut energy_change = 0.0;
            
            // Diffusion with bucket above (if exists)
            if i > 0 {
                let above_bucket = &self.buckets[i - 1];
                let temp_diff = above_bucket.temperature_k - current_bucket.temperature_k;
                let thermal_transfer = self.calculate_thermal_transfer(above_bucket, current_bucket, temp_diff, years);
                energy_change += thermal_transfer;
            }
            
            // Diffusion with bucket below (if exists)
            if i < self.buckets.len() - 1 {
                let below_bucket = &self.buckets[i + 1];
                let temp_diff = below_bucket.temperature_k - current_bucket.temperature_k;
                let thermal_transfer = self.calculate_thermal_transfer(below_bucket, current_bucket, temp_diff, years);
                energy_change += thermal_transfer;
            }
            
            // Convert energy change to temperature change
            let mass = current_bucket.density * 1000.0 * 1000.0 * current_bucket.height_km; // kg
            let thermal_capacity = mass * current_bucket.specific_heat;
            let temp_change = energy_change / thermal_capacity;
            
            new_temps[i] = current_bucket.temperature_k + temp_change;
        }
        
        // Apply new temperatures
        for (i, &new_temp) in new_temps.iter().enumerate() {
            self.buckets[i].temperature_k = new_temp;
        }
    }
    
    fn calculate_thermal_transfer(&self, from_bucket: &ThermalBucket, to_bucket: &ThermalBucket, temp_diff: f64, years: f64) -> f64 {
        // Simple thermal diffusion calculation
        let avg_conductivity = (from_bucket.thermal_conductivity + to_bucket.thermal_conductivity) / 2.0;
        let distance = 1000.0; // 1km in meters
        let area = 1000.0 * 1000.0; // 1km² in m²
        let seconds = years * 365.25 * 24.0 * 3600.0;
        
        // Enhanced diffusion for asthenosphere (convection simulation)
        let conductivity_multiplier = match (&from_bucket.material_type, &to_bucket.material_type) {
            (BucketMaterial::Asthenosphere, BucketMaterial::Asthenosphere) => 3.0, // Moderate convective mixing
            _ => 1.0,
        };
        
        let heat_flux = avg_conductivity * conductivity_multiplier * temp_diff * area / distance;
        heat_flux * seconds * 0.01 // Scale down for stability
    }
    
    fn export_results(&self, filename: &str) {
        let mut file = File::create(filename).expect("Could not create file");

        // Aggregate every 4 buckets into one column for manageable data size
        writeln!(file, "depth_4km,initial_temp_k,final_temp_k,temp_change_k,material_type").unwrap();

        let mut i = 0;
        while i < self.buckets.len() {
            let end_idx = (i + 4).min(self.buckets.len());
            let chunk = &self.buckets[i..end_idx];
            let initial_chunk = &self.initial_buckets[i..end_idx];

            // Average temperatures over 4km chunk
            let avg_depth = chunk.iter().map(|b| b.depth_km).sum::<f64>() / chunk.len() as f64;
            let avg_initial_temp = initial_chunk.iter().map(|b| b.temperature_k).sum::<f64>() / chunk.len() as f64;
            let avg_final_temp = chunk.iter().map(|b| b.temperature_k).sum::<f64>() / chunk.len() as f64;
            let temp_change = avg_final_temp - avg_initial_temp;

            // Use the most common material type in the chunk
            let material_str = match &chunk[0].material_type {
                BucketMaterial::Surface => "Surface",
                BucketMaterial::Lithosphere => "Lithosphere",
                BucketMaterial::Asthenosphere => "Asthenosphere",
            };

            writeln!(file, "{:.1},{:.1},{:.1},{:.1},{}",
                     avg_depth, avg_initial_temp, avg_final_temp, temp_change, material_str).unwrap();

            i += 4;
        }

        println!("   Temperature range: {:.1}K to {:.1}K",
                 self.buckets.iter().map(|b| b.temperature_k).fold(f64::INFINITY, f64::min),
                 self.buckets.iter().map(|b| b.temperature_k).fold(f64::NEG_INFINITY, f64::max));
        println!("   Exported {} 4km-averaged data points", (self.buckets.len() + 3) / 4);
    }
}
