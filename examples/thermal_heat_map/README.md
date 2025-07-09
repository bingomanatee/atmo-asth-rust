# Thermal Heat Map Example

This example demonstrates PNG heat map export functionality for the global thermal simulation.

## Overview

Based on `global_thermal_radiance_integrated.rs`, this example adds PNG heat map visualization that exports temperature data as color-coded images every simulation step.

## Features

- **Real-time PNG Export**: Generates heat map images every simulation step
- **3 PPD Resolution**: Uses 3 pixels per degree (1080x540 pixel images)
- **Temperature Color Coding**: 
  - 0K = Black
  - 1000K = Red  
  - 1500K = Yellow
  - 2000K = White
- **H3 Graphics Integration**: Uses existing `h3o_png.rs` graphics system
- **Mass-Weighted Temperatures**: Calculates average temperature per cell weighted by mass
- **Filtered Data**: Excludes foundry layers (>500km depth) and atmospheric layers

## Files Generated

- `heat_map_step_XXXX.png` - PNG images for each simulation step
- Images show thermal patterns across the planet's surface
- Latitude/longitude grid overlay for geographic reference

## Usage

```bash
cargo run --example global_thermal_radiance_with_heat_map
```

## Technical Details

- **Resolution**: H3 Level 2 (5,882 cells globally)
- **Image Size**: 1080x540 pixels (3 pixels per degree)
- **Export Format**: PNG with RGB color mapping
- **Simulation**: 50 steps at 5,000 years per step (250,000 years total)
- **Graphics System**: Uses `H3GraphicsGenerator::generate_coherent_cells_with_colors_png()`

## Output

The simulation generates:
1. Console output showing thermal evolution
2. PNG heat map files in `examples/thermal_heat_map/`
3. Real-time progress updates during export

Each PNG file shows the global thermal state at that simulation step, allowing visualization of how thermal patterns evolve over geological time scales.