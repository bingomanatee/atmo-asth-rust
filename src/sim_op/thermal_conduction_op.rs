use crate::constants::KM_TO_M;
use crate::energy_mass_composite::EnergyMassComposite;
use crate::fourier_thermal_transfer::FourierThermalTransfer;
use crate::global_thermal::sim_cell::SimCell;
use crate::h3_utils::H3Utils;
use crate::sim::simulation::Simulation;
/// Thermal conduction operation extending HeatRedistributionOp
///
/// Implements proper Fourier thermal conduction considering ALL thermal neighbors:
/// - Vertical neighbors (layers within same cell)  
/// - Lateral neighbors (corresponding layers in adjacent cells)
///
/// Heat flows naturally based on temperature gradients rather than artificial budget splitting
use crate::sim_op::SimOp;
use crate::thermal_pressure_cache::ThermalPressureCache;
use h3o::CellIndex;
use rayon::prelude::*;
use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;

/// Parameters for thermal conduction
#[derive(Debug, Clone)]
pub struct ThermalConductionParams {
    /// Include lateral (horizontal) heat transfer between neighboring cells
    pub enable_lateral_conduction: bool,
    /// Lateral thermal conductivity scaling factor (relative to material conductivity)
    pub lateral_conductivity_factor: f64,
    /// Temperature difference threshold (K) - ignore smaller differences
    pub temp_diff_threshold_k: f64,
    /// Enable detailed reporting for debugging
    pub enable_reporting: bool,
}

impl Default for ThermalConductionParams {
    fn default() -> Self {
        Self {
            enable_lateral_conduction: true,
            lateral_conductivity_factor: 0.5, // Lateral conductivity is 50% of material conductivity
            temp_diff_threshold_k: 1.0,       // 1K threshold for energy transfer
            enable_reporting: false,
        }
    }
}

/// Thermal neighbor representing a potential heat transfer path
#[derive(Debug, Clone, Debug, Clone)]
enum NeighborType {
    VerticalLayer {
        cell_index: CellIndex,
        layer_index: usize,
    },
    LateralCell {
        cell_index: CellIndex,
        layer_index: usize,
    },
}

/// Cached neighbor topology for performance optimization
#[derive(Debug, Clone)]
struct NeighborTopology {
    /// Map from cell index to its H3 neighbors
    lateral_neighbors: HashMap<CellIndex, Vec<CellIndex>>,
    /// Map from cell pair to distance in meters
    cell_distances: HashMap<(CellIndex, CellIndex), f64>,
}

impl NeighborTopology {
    fn new() -> Self {
        Self {
            lateral_neighbors: HashMap::new(),
            cell_distances: HashMap::new(),
        }
    }

    fn get_or_compute_neighbors(&mut self, cell_index: CellIndex) -> &Vec<CellIndex> {
        self.lateral_neighbors
            .entry(cell_index)
            .or_insert_with(|| H3Utils::neighbors_for(cell_index))
    }

    fn get_or_compute_distance(
        &mut self,
        cell_a: CellIndex,
        cell_b: CellIndex,
        planet_radius_km: f64,
    ) -> f64 {
        let key = if cell_a < cell_b {
            (cell_a, cell_b)
        } else {
            (cell_b, cell_a)
        };
        *self
            .cell_distances
            .entry(key)
            .or_insert_with(|| H3Utils::cell_distance_m(cell_a, cell_b, planet_radius_km))
    }
}

/// Pre-calculated geometric properties for thermal transfers at a specific layer depth
#[derive(Debug, Clone)]
struct LayerGeometry {
    /// Typical cell area at this depth (km²)
    cell_area_km2: f64,
    /// Typical distance between adjacent cells at this depth (m)
    lateral_distance_m: f64,
    /// Lateral transfer area (1/6 of cell area for hexagonal face) (m²)
    lateral_transfer_area_m2: f64,
    /// Vertical transfer area (full cell area) (m²)
    vertical_transfer_area_m2: f64,
}

impl LayerGeometry {
    fn new(resolution: h3o::Resolution, planet_radius_km: f64) -> Self {
        // Calculate typical cell area at this resolution and planet radius
        let cell_area_km2 = H3Utils::cell_area(resolution, planet_radius_km);

        // Calculate typical distance between adjacent cells
        // For H3, the distance between adjacent cell centers is approximately:
        // distance ≈ sqrt(2 * area / (3 * sqrt(3))) * 2
        // But we can use a more accurate approximation for H3 hexagons
        let lateral_distance_m = (cell_area_km2 * 2.0 / 3.0_f64.sqrt()).sqrt() * 1000.0;

        // Calculate transfer areas
        let lateral_transfer_area_m2 = cell_area_km2 * 1e6 / 6.0; // 1/6 of cell area for hex face
        let vertical_transfer_area_m2 = cell_area_km2 * 1e6; // Full cell area

        Self {
            cell_area_km2,
            lateral_distance_m,
            lateral_transfer_area_m2,
            vertical_transfer_area_m2,
        }
    }
}

/// Thermal conduction operation
pub struct ThermalConductionOp {
    params: ThermalConductionParams,
    fourier_transfer: Option<FourierThermalTransfer>,
    apply_during_simulation: bool,
    debug_output: bool,
    current_step: usize,
    total_vertical_energy_transferred: f64,
    total_lateral_energy_transferred: f64,
 }

struct PotentialTransfer {
    neighbor: NeighborType,
    potential_transfer_j: f64,
    cell_index: CellIndex,
    layer_index: usize,
}

struct Transfer {
    neighbor: NeighborType,
    transfer_j: f64,
    cell_index: CellIndex,
    layer_index: usize,
}

struct TransferSource {
    source_temp: f64,
    source_energy: f64,
    cell_index: CellIndex,
    layer_index: usize,
}

impl ThermalConductionOp {
    pub fn new() -> Self {
        Self {
            params: ThermalConductionParams::default(),
            fourier_transfer: None,
            apply_during_simulation: true,
            debug_output: false,
            current_step: 0,
            total_vertical_energy_transferred: 0.0,
            total_lateral_energy_transferred: 0.0,
        }
    }

    pub fn new_with_params(params: ThermalConductionParams) -> Self {
        Self {
            params,
            fourier_transfer: None,
            apply_during_simulation: true,
            debug_output: false,
            current_step: 0,
            total_vertical_energy_transferred: 0.0,
            total_lateral_energy_transferred: 0.0,
        }
    }

    pub fn with_debug(mut self) -> Self {
        self.debug_output = true;
        self
    }

    /// Calculate cell radius from surface area
    fn calculate_cell_radius_km(&self, surface_area_km2: f64) -> f64 {
        (surface_area_km2 / std::f64::consts::PI).sqrt()
    }

    /// Identify all thermal neighbors for a given layer in a cell with caching
    fn cooler_neighbors(
        &mut self,
        source: &TransferSource,
        source_cell: &SimCell,
        all_cells: &HashMap<CellIndex, SimCell>,
    ) -> Vec<NeighborType> {
        let mut neighbors = Vec::new();

        // Get source layer properties
        if source.layer_index >= source_cell.layers_t.len() {
            return neighbors;
        }

        let (source_layer, _) = &source_cell.layers_t[source.layer_index];

        // Skip atmospheric layers as heat sources
        if source_layer.is_atmospheric() {
            return neighbors;
        }

        // 1. VERTICAL NEIGHBORS (layers above and below in same cell)

        // Layer below (deeper)
        for offset in [-1, 1].iter() {
            let index = source.layer_index + offset;
            if index < 0 || index >= source_cell.layers_t.len() {
                continue;
            }
            let (neighbor_layer, _) = &source_cell.layers_t[index];

            // Skip atmospheric neighbor layers
            if !neighbor_layer.is_atmospheric() {
                let neighbor_temp = neighbor_layer.temperature_k();
                let temp_diff = (source.source_temp - neighbor_temp).abs();

                if temp_diff > self.params.temp_diff_threshold_k
                    && source.source_temp > neighbor_temp
                {
                    neighbors.push(NeighborType::VerticalLayer {
                        cell_index: source.cell_index,
                        layer_index: index,
                    });
                }
            }
        }

        // 2. LATERAL NEIGHBORS (corresponding layers in adjacent cells)
        if self.params.enable_lateral_conduction {
            for neighbor_cell_index in H3Utils::neighbors_for(source.cell_index) {
                if let Some(neighbor_cell) = all_cells.get(&neighbor_cell_index) {
                    // Check if neighbor cell has the corresponding layer
                    if source.layer_index < neighbor_cell.layers_t.len() {
                        let (neighbor_layer, _) = &neighbor_cell.layers_t[source.layer_index];

                        // Skip atmospheric neighbor layers
                        if neighbor_layer.energy_mass.material_composite_type()
                            != crate::energy_mass_composite::MaterialCompositeType::Air
                        {
                            let neighbor_temp = neighbor_layer.temperature_k();
                            let temp_diff = (source.source_temp - neighbor_temp).abs();

                            if temp_diff > self.params.temp_diff_threshold_k
                                && neighbor_temp < source.source_temp
                            {
                                // Store lateral neighbor info (geometry calculated in physics method)
                                neighbors.push(NeighborType::LateralCell {
                                    cell_index: neighbor_cell_index,
                                    layer_index: source.layer_index,
                                });
                            }
                        }
                    }
                }
            }
        }

        neighbors
    }

    /// Calculate Fourier thermal transfer between a source layer and its thermal neighbor
    /// Uses the sophisticated FourierThermalTransfer class for accurate physics
    fn calculate_fourier_transfer_between_neighbors(
        &mut self,
        cells: &HashMap<CellIndex, SimCell>,
        source_cell_index: CellIndex,
        source_layer_index: usize,
        neighbor: &NeighborType,
        sim: &Simulation,
    ) -> f64 {
        // Get the source cell and layer
        let source_cell = match cells.get(&source_cell_index) {
            Some(cell) => cell,
            None => return 0.0,
        };

        if source_layer_index >= source_cell.layers_t.len() {
            return 0.0;
        }

        let source_layer_tuple = &source_cell.layers_t[source_layer_index];

        // Get the target cell and layer based on neighbor type
        let (target_cell_index, target_layer_index) = match neighbor {
            NeighborType::VerticalLayer {
                cell_index,
                layer_index,
            } => (cell_index, layer_index),
            NeighborType::LateralCell {
                cell_index,
                layer_index,
            } => (cell_index, layer_index),
        };

        let target_cell = match cells.get(&target_cell_index) {
            Some(cell) => cell,
            None => return 0.0,
        };

        if target_layer_index >= *target_cell.layers_t.len() {
            return 0.0;
        }

        let target_layer_tuple = &target_cell.layers_t[target_layer_index];
        let (temp_source, _) = source_layer_tuple;
        let (temp_target, _) = target_layer_tuple;
        // Use FourierThermalTransfer with proper physics-based calculations
        if let Some(ref fourier) = self.fourier_transfer {
            match neighbor {
                NeighborType::VerticalLayer { .. } => fourier.calculate_potential_heat_flow_simple(
                    &temp_source,
                    &temp_target,
                    source_cell.surface_area_km2(),
                    source_cell.height_km() + target_cell.height_km(),
                    sim.years_per_step as f64,
                ),
                NeighborType::LateralCell { .. } => {
                    // For lateral transfers, use cached distance calculation
                    let distance_km = target_cell.cell_radius() * 2.0;
                    let area_m2 = source_layer_tuple.0.surface_area_km2 * KM_TO_M.powi(2) / 6.0; // Hex face area in m²
                    let effective_conductivity =
                        source_layer_tuple.0.energy_mass.thermal_conductivity()
                            * self.params.lateral_conductivity_factor;

                    fourier.calculate_potential_heat_flow_simple(
                        &temp_source,
                        &temp_target,
                        area_m2,
                        distance_km,
                        effective_conductivity,
                    )
                }
            }
        } else {
            0.0
        }
    }

    /// Calculate Fourier thermal transfer without mutable access (for parallel processing)
    fn calculate_fourier_transfer_immutable(
        &self,
        mut sim: &Simulation,
        source: &TransferSource,
        neighbor: &NeighborType,
    ) -> f64 {
        // Get the source cell and layer
        let source_cell = match sim.cells.get(&source.cell_index) {
            Some(cell) => cell,
            None => return 0.0,
        };

        if source.layer_index >= source_cell.layers_t.len() {
            return 0.0;
        }
        let source_layer_tuple = &source_cell.layers_t[source.layer_index];

        // Get the target cell and layer based on neighbor type
        let (target_cell_index, target_layer_index) = match neighbor {
            NeighborType::VerticalLayer {
                cell_index,
                layer_index,
            } => (cell_index, layer_index),
            NeighborType::LateralCell {
                cell_index,
                layer_index,
            } => (cell_index, layer_index),
        };

        let target_cell = match sim.cells.get(&target_cell_index) {
            Some(cell) => cell,
            None => return 0.0,
        };

        if *target_layer_index >= target_cell.layers_t.len() {
            return 0.0;
        }

        let target_layer_tuple = &target_cell.layers_t[target_layer_index];

        // Use FourierThermalTransfer with proper physics-based calculations
        if let Some(ref fourier) = self.fourier_transfer {
            match neighbor {
                NeighborType::VerticalLayer { .. } => fourier.calculate_potential_heat_flow_simple(
                    &source_layer_tuple.0,
                    &target_layer_tuple.0,
                    source_cell.surface_area_km2(),
                    source_cell.height_km() + target_cell.height_km(),
                    sim.years_per_step as f64,
                ),
                NeighborType::LateralCell { .. } => fourier.calculate_potential_heat_flow_simple(
                    &source_layer_tuple.0,
                    &target_layer_tuple.0,
                    source_cell.lateral_area(),
                    2.0 * source_cell.cell_radius(),
                    sim.years_per_step as f64,
                ),
            }
        } else {
            0.0
        }
    }

    /// Apply unified thermal conduction to all cells with parallel processing
    /// Only models downstream flows (hot->cold) with overbalance protection
    fn apply_unified_thermal_conduction(&mut self, sim: &mut Simulation) {
        use std::sync::Mutex;

        // Reset transfer counters
        self.total_vertical_energy_transferred = 0.0;
        self.total_lateral_energy_transferred = 0.0;

        // Collect all source layers that need processing
        let mut source_layers: Vec<TransferSource> = Vec::new();
        // Thread-safe collections for parallel processing
        let energy_transfers = Mutex::new(Vec::new());

        for (cell_index, cell) in sim.cells.iter() {
            cell.layers_t
                .iter()
                .enumerate()
                .for_each(|(layer_index, (source_layer, _))| {
                    if !source_layer.is_atmospheric() {
                        let source_temp = source_layer.temperature_k();
                        let source_energy = source_layer.energy_mass.energy();
                        source_layers.push(TransferSource {
                            source_temp,
                            source_energy,
                            cell_index: *cell_index,
                            layer_index,
                        });
                    }
                });
        }

        // Process source layers in parallel
        source_layers.par_iter().for_each(|source| {
            let cell = match sim.cells.get(&source.cell_index) {
                Some(cell) => cell,
                None => return,
            };

            // Find all COOLER thermal neighbors (downstream flow only)
            // Note: We can't use &mut self in parallel, so we'll compute neighbors directly
            let cooler_neighbors = self.cooler_neighbors(source, cell, sim.borrow());
            if cooler_neighbors.is_empty() {
                return; // No cooler neighbors to transfer to
            }

            // Calculate total potential energy outflow to all cooler neighbors
            let mut total_potential_outflow = 0.0;
            let mut neighbor_transfers = Vec::new();

            for neighbor in &cooler_neighbors {
                // Create pair key for caching (always put lexicographically smaller first)

                // Calculate energy transfer using immutable method
                let energy_transfer_j =
                    self.calculate_fourier_transfer_immutable(sim.borrow(), source, neighbor);

                total_potential_outflow += energy_transfer_j;
                neighbor_transfers.push(PotentialTransfer {
                    cell_index: source.cell_index,
                    layer_index: source.layer_index,
                    neighbor: neighbor.clone(),
                    potential_transfer_j: energy_transfer_j,
                });
            }

            // Apply overbalance protection: limit total outflow to half the source energy
            let max_total_outflow = source.source_energy * 0.5;
            let outflow_scale_factor = if total_potential_outflow > max_total_outflow {
                max_total_outflow / total_potential_outflow
            } else {
                1.0
            };

            // Apply scaled transfers to prevent overbalancing
            let mut actual_total_outflow = 0.0;

            for transfer in neighbor_transfers {
                let scaled_transfer = transfer.potential_transfer_j * outflow_scale_factor;
                actual_total_outflow += scaled_transfer;

                // Record the transfer for atomic application
                let mut transfers = energy_transfers.lock().unwrap();
                transfers.push(Transfer {
                    cell_index: transfer.cell_index,
                    layer_index: transfer.layer_index,
                    transfer_j: -scaled_transfer,
                    neighbor: transfer.neighbor,
                });
            }
        });

        // Extract results from thread-safe containers
        let final_energy_transfers = energy_transfers.into_inner().unwrap();

        // Apply all energy transfers atomically
        for transfer in final_energy_transfers {
            let source = match sim.cells.get_mut(&transfer.cell_index) {
                Some(cell) => match cell.layer_tuple_mut(transfer.layer_index) {
                    Some((_, next)) => next,
                    None => continue,
                },
                None => continue,
            };

            let target = match transfer.neighbor {
                NeighborType::VerticalLayer {
                    cell_index,
                    layer_index,
                } => match sim.cells.get_mut(&cell_index) {
                    Some(cell) => match cell.layer_tuple_mut(layer_index) {
                        Some((_, next)) => next,
                        None => continue,
                    },
                    None => continue,
                },
                NeighborType::LateralCell {
                    cell_index,
                    layer_index,
                } => match sim.cells.get_mut(&cell_index) {
                    Some(cell) => match cell.layer_tuple_mut(layer_index) {
                        Some((_, next)) => next,
                        None => continue,
                    },
                    None => continue,
                },
            };

            if transfer.transfer_j > 0.0 {
                source.remove_energy(transfer.transfer_j);
                target.add_energy(transfer.transfer_j);
            }
        }
    }
}

impl SimOp for ThermalConductionOp {
    fn name(&self) -> &str {
        "ThermalConduction"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn init_sim(&mut self, sim: &mut Simulation) {
        self.current_step = 0;
        self.total_vertical_energy_transferred = 0.0;
        self.total_lateral_energy_transferred = 0.0;

        // Initialize Fourier thermal transfer (for potential future use)
        // The cache is initialized automatically in the constructor
        self.fourier_transfer = Some(FourierThermalTransfer::new(sim.years_per_step as f64));

        // Pre-cache neighbor topology for common cells to optimize later lookups
        // This is optional and will be filled as needed during simulation
        self.neighbor_topology = NeighborTopology::new();

        // Initialize geometry cache based on simulation parameters
        let max_layers = if let Some(first_cell) = sim.cells.values().next() {
            first_cell.layers_t.len()
        } else {
            30 // Default maximum layers
        };

        self.geometry_cache = Some(GeometryCache::new(
            sim.resolution,
            sim.planet.radius_km as f64,
            max_layers,
        ));

        if self.params.enable_reporting {
            println!("🌡️  Thermal conduction initialized with advanced optimizations");
            println!(
                "   - Lateral conduction: {}",
                if self.params.enable_lateral_conduction {
                    "enabled"
                } else {
                    "disabled"
                }
            );
            println!(
                "   - Lateral conductivity factor: {:.1}%",
                self.params.lateral_conductivity_factor * 100.0
            );
            println!(
                "   - Temperature threshold: {:.1}K",
                self.params.temp_diff_threshold_k
            );
            println!("   - Neighbor caching: enabled");
            println!("   - Parallel processing: enabled");
            println!("   - Geometry caching: enabled ({} layers)", max_layers);
            println!("   - Planetary-scale approximation: enabled");
        }
    }

    fn update_sim(&mut self, sim: &mut Simulation) {
        self.apply_unified_thermal_conduction(sim);
    }
}
