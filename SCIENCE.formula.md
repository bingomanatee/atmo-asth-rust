
## Radiative Exchange Between Surface, Atmosphere, and Space

#### Surface to Surface
```text
heat_flux_surface_to_surface =
    view_factor                      // fraction [0–1]
  * surface1_emissivity              // [0–1]
  * surface2_emissivity              // [0–1]
  * stefan_boltzmann_constant        // 5.670374419e-8 W/(m²·K⁴)
  * (temperature_surface1^4 – temperature_surface2^4) // K⁴
```

#### Surface to Atmosphere
```text
heat_flux_surface_to_atmosphere =
    surface_emissivity                // [0–1]
  * stefan_boltzmann_constant        // W/(m²·K⁴)
  * (surface_temperature)^4          // K⁴

# Optional atmospheric filtering:
heat_flux_surface_to_atmosphere =
    atmospheric_transmissivity        // [0–1]
  * surface_emissivity                // [0–1]
  * stefan_boltzmann_constant        // W/(m²·K⁴)
  * (surface_temperature)^4          // K⁴
```

#### Atmosphere to Space
```text
heat_flux_atmosphere_to_space =
    atmosphere_emissivity             // [0–1]
  * stefan_boltzmann_constant        // W/(m²·K⁴)
  * (effective_atmosphere_temperature)^4 // K⁴
```

#### Surface to Space (Vacuum)
```text
heat_flux_surface_to_space =
    surface_emissivity                // [0–1]
  * stefan_boltzmann_constant        // W/(m²·K⁴)
  * (surface_temperature)^4          // K⁴
```

## Radiating Skin Depth
- Only the topmost portion of solid material exchanges radiation each timestep. Compute **skin depth**:
  ```text
  skin_depth = sqrt(
      thermal_diffusivity            // m²/s
    * time_step                       // s
  )                                    // m
  ```
- **thermal_diffusivity_of_material** = conductivity / (density * specific_heat_capacity)
- Carve that depth out of the top layer(s) to form a radiating slab, sum their heat capacities, and apply radiative energy change to that slab.


## Conduction Between Layers

note - Fourier's Law includes distance between layer centers - this is the standout scenario in which thickness
matters; all the other formula only care about surface area.

- **Heat flux between two layers** (Fourier’s law):
  ```text
  heat_flux = -effective_conductivity                          // W/(m·K)
              * (temperature_of_upper_layer - temperature_of_lower_layer)  // K
              / distance_between_layer_centers                  // m
  ```
  where:
    - **effective_conductivity** = (conductivity_upper + conductivity_lower) / 2  (W/m·K)
    - **distance_between_layer_centers** = sum of half the thicknesses of each layer (m)

- **Energy transferred** over one timestep:
  ```text
  energy_transferred = heat_flux   // W/m²
                     * layer_area  // m²
                     * time_step   // s
  ```

- **Temperature update** for each layer:
  ```text
  temperature_change = energy_transferred  // J
                     / heat_capacity_of_layer // J/K
  new_temperature   = old_temperature + temperature_change // K
  ```

- **Stability check** for explicit conduction:
  ```text
  max_time_step = (distance_between_layers)^2             // m²
                / (2 * thermal_diffusivity_of_material)   // m²/s
  ```
  where **thermal_diffusivity_of_material** = conductivity / (density * specific_heat_capacity) (m²/s).

## Thermal Expansion
```text
density = referenceDensity × (1 − thermalExpansivity × ( currentTemperature − referenceTemperature ))

```

## Raileigh number 
```text

RayleighNumber = ( Gravity 
                   × ThermalExpansivity 
                   × TemperatureContrast 
                   × LayerWidth³ ) 
                 / ( KinematicViscosity 
                  yes.    × ThermalDiffusivity )

KinematicViscosity = DynamicViscosity / Density

boundaryLayerThickness = LayerWidth
                         × ( RayleighNumber )^(–1/4)
            
          Using your 200 km depth and typical mantle numbers:

Quantity	Value
LayerWidth	200 000 m
Gravity	9.8 m/s²
ThermalExpansivity	1×10⁻⁵ 1/K
TemperatureContrast	500 K
DynamicViscosity	1×10²¹ Pa·scan 
Density	3300 kg/m³
ThermalDiffusivity	1×10⁻⁶ m²/s

KinematicViscosity = 1×10²¹ Pa·s ÷ 3300 kg/m³ ≈ 3×10¹⁷ m²/s

RayleighNumber ≈ 9.8×10⁻⁵×500×(2×10⁵)³ ÷ (3×10¹⁷×1×10⁻⁶) ≈ 1.3×10³

boundaryLayerThickness = 2×10⁵ m × (1.3×10³)^(–¼) ≈ 3.3×10⁴ m ≈ 33 km

So you’d expect plume diameters on the order of 30 km in that setup.  
                         
```
## Choosing Between Conduction and Radiation

Use conduction and radiation in complementary contexts based on where and how heat is moving:

- **Deep and internal layers (kilometers below the surface):**
    - Heat moves primarily by **conduction** through solid rock or molten material.
    - Apply the **Fourier’s law** formulas to every pair of adjacent layers throughout the column.
    - Conduction captures the gradual transfer of heat from the core upward, through thick strata.

- **Surface and boundary interfaces:**
    - Heat leaves the solid planet either by **radiation** (to space or between exposed surfaces) or by **exchange with the atmosphere**.
    - Use the **radiative exchange** formulas only at the topmost radiating skin depth (or between two solid surfaces that directly face each other).

- **When to choose one over the other:**
    - If your layer-to-layer step is fully buried (no direct line of sight to space or another surface), always use **conduction**.
    - If you are modeling heat loss or gain at an exposed boundary (surface→space or surface→atmosphere) or between two surfaces (e.g. two rock outcrops), use **radiation**.
    - Do **not** replace conduction with radiation in the deep interior—radiation through solid rock is negligible compared to conduction.

- **Practical hybrid workflow:**
    1. **Core injection and internal diffusion:** apply conduction ops throughout all subsurface chunks.
    2. **Surface skin radiation:** carve out the radiating slab and apply radiative formulas at that interface only.
    3. **Atmospheric coupling:** apply surface↔atmosphere radiative (and optionally convective) exchange next, then atmosphere→space radiation.

By keeping conduction for all buried layers and radiation only at exposed boundaries, your simulation remains both physically accurate and numerically stable.
This summary replaces Greek symbols with descriptive variable names for clarity. Adjust the variables to match your simulation's naming conventions and units.