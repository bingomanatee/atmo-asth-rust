# Evolving Energy-Mass Models: From Hardcoded Constants to JSON-Driven Material Physics

## Introduction

In the development of complex thermal simulation systems, one of the most challenging aspects is accurately modeling how materials behave under different temperature and energy conditions. This article chronicles the evolution of an energy-mass model from a simple hardcoded system to a sophisticated JSON-driven material physics engine, highlighting the algorithmic challenges and architectural decisions that emerged along the way.

## Some more context 

All of this was a prelude to a weather system I wanted to model for Earth 
or hypothetically other worlds with other properties,
in a way that coincided with scientific law. 
To develop a tectonic plate simulation we wanted to simulate a moving layer of magma (asthenosphere) 
_and_ the generation of the surface layer of material (lithosphere) through which plates move. 
My naive assumption was plates skim around on the lithosphere on momentum like ice hockey pucks; 
in truth they are dragged along propelled by high energy flows of magma beneath them.
Furthere there are two basic types of plates - smaller high density oceanic plates that move lower 
and in fact cover the earth and add kinetic energy to the continental plates. 
(and in fact a mid range set of plates between these) 

Despite being enamored of the lateral motion of plates we then had to simulate the three dimensional movement of energy
and material not only across the surface but up and down towards or away from the earths core in upwelling and downwelling.
These are far lareger than simple or clusters of volcanic eruptions - these are large persistent wellings of material hundreds or thousands of miles across driven by energy surges or the descent of cooled and or more dense materials into the oceans core. 

## Too Cool for School

This meant modeling the Lithosphere; an that meant the end of setting up a nice clean set of layers an actually getting into
what created - or didn't create - or even melted -- the solid material on which we live. This meant diving deep into the
material properties that make up the Earth itself. 

As it turns out everything tangible has three fundamental qualities, aside from the identity of the material it is made from:

1. Mass (kg)
2. Temperature (Kelvin)
3. Volume (km^3)

However there are many ways to interpret these:

1. energy (Joules)
2. volume (km^3)
3. density (kg/km3)

Temperature (Kelvin) can be derived from energy and volume; density can be considered a constant tied to material properties. 
That also makes mass a derivable property driven by density and volume. Another important property was the _specific heat_;
this is the __energy required to raise 1 kg of the material by 1 degree Kelvin__.

The overall model was that energy would be transferred across layers, ultimately being cooled at the surface of the 
planet first by the icy void of space, then by the atmosphere and oceans above. Given that there is no absolute rule as
to which properties were "hard" and which were derived the following starting point was set: 

these properties would be 
* energy
* volume

density for the moment would be kept static with the material properties, and temperature would be derived. Setting
temperature would change the energy due to specific heat. 

## The Initial Challenge: Hardcoded Material Properties

The journey began with a straightforward approach - hardcoded material constants embedded directly in the code:

```rust
const MANTLE_DENSITY_KGM3: f64 = 3300.0;
const SPECIFIC_HEAT_CAPACITY_MANTLE_J_PER_KG_K: f64 = 1200.0;
```

While this approach worked for initial prototyping,
it quickly became clear that real-world geological simulations required:

1. **Multiple material types** (silicate, basalt, peridotite, ice, air)
2. **Accurate phase transition modeling** (solid ↔ liquid ↔ gas)
3. **Temperature-dependent properties**
4. **Configurable material parameters** for different simulation scenarios

the latter was the kicker - material has different densities like specific heat and a very important statistic 
_conductivity_ -- the affinity for sharing energy between layers of different levels, a critical feature for
this system mostly developed for transmission of energy through layers. 

Further: Lithosphere and Asthenosphere layers were not so different - one wa a molten version of the other,
or a cooled version wherever you start off. So tracking when one becomes the other required me to  understand
when - and how - this occurs. 

## Algorithm Evolution: Three Generations of Material State Conversion

### Generation 1: Simple Temperature Thresholds

The first iteration used basic temperature thresholds:

```rust
fn resolve_phase_simple(temperature: f64) -> MaterialPhase {
    if temperature > 1600.0 {
        MaterialPhase::Liquid
    } else {
        MaterialPhase::Solid
    }
}
```

The problems with this is that flicking the switch from solid to liquid also changed the specific heat - which changed
the temperature. The other thing was that simply turning a lump of earth to magma instantly at the change of one degree
was just too easy; and in fact in the real world you have to pump a high amount of energy into a system to melt it. 
Finding that quantity that had to be invvested was important; also, this process worked in reverse as well - cooling 
magma into crust meant taking a chunk of energy out of the equation. 

This taking in and putting back a chunk of energy was not only necessary to prevent instant magma but to reflect the 
fact that you hvae to invest a significant hunk of joules to change state - so its appropriate to stick that energy
somewhere - and that somewhere we called the "phase bank". 

### Generation 2: Complex Energy Banking System

The second generation introduced sophisticated energy banking to handle gradual phase transitions:

```rust
struct EnergyBank {
    banked_energy: f64,
    transition_range_min: f64,
    transition_range_max: f64,
    target_phase: MaterialPhase,
}

impl EnergyBank {
    fn add_energy(&mut self, energy: f64) -> f64 {
        // Complex logic for banking energy during transitions
        let transition_cost = self.calculate_transition_cost();
        if self.banked_energy + energy >= transition_cost {
            // Complete transition
            let excess = (self.banked_energy + energy) - transition_cost;
            self.banked_energy = 0.0;
            excess
        } else {
            // Bank the energy
            self.banked_energy += energy;
            0.0
        }
    }
}
```

**Features:**
- Gradual energy accumulation during phase transitions
- Temperature ranges for transitions (e.g., 1600K - 3200K for silicate melting)
- Energy conservation through banking mechanism

This proved complex and error prone. As the share of energy sloughed out to the bank increased as you passed from 
minimum to maximum, it wasn't easy to just plug in a set of numbers and get your response; it required incremental 
investment to tack the escalating portion of the energy stuck into the bank. 

further doing so did not in fact increase its scientific accuracy. Aside from having the temperature do a graceful 
pause, our numbers required a tremendous investment of energy to change state - an unrealisticly large one at that,
and one that was difficult to compute.

### Generation 3: Simplified Latent Heat Model

The final iteration adopted a simplified but physically accurate approach: we have a single conversion point which was a 
full stop, and a better and more scientific set of material characteristics to determine exactly how much energy to sluff off 
to the bank; this was not only easier to model, but the transition point was hard coded and much easier to comppute. Further,
phase (solid/liquid/gas) was determined exclusively from temerature, as it was just a cutoff-based value" 

```rust
fn resolve_phase_from_temperature(
    material_type: MaterialCompositeType, 
    temperature: f64
) -> MaterialPhase {
    let profile = get_material_profile(material_type);
    
    if temperature >= profile.melt_temp {
        if temperature >= profile.boil_temp {
            MaterialPhase::Gas
        } else {
            MaterialPhase::Liquid
        }
    } else {
        MaterialPhase::Solid
    }
}

fn apply_latent_heat_transition(
    &mut self, 
    target_phase: MaterialPhase
) -> f64 {
    if target_phase == self.phase {
        return 0.0;
    }
    
    let profile = self.material_composite_profile();
    let latent_heat = match (self.phase, target_phase) {
        (MaterialPhase::Solid, MaterialPhase::Liquid) => profile.latent_heat_fusion,
        (MaterialPhase::Liquid, MaterialPhase::Solid) => -profile.latent_heat_fusion,
        (MaterialPhase::Liquid, MaterialPhase::Gas) => profile.latent_heat_vaporization,
        (MaterialPhase::Gas, MaterialPhase::Liquid) => -profile.latent_heat_vaporization,
        _ => 0.0,
    };
    
    let mass_kg = self.mass_kg();
    let energy_change = mass_kg * latent_heat;
    
    self.energy_joules += energy_change;
    self.phase = target_phase;
    
    energy_change
}
```

## The JSON Revolution: Data-Driven Material Properties

Along side that we put off the material properties into JSON instead of a hard coded hash map. this cleaned up the code
quite a bit. Also instead of putting conversion temperatures into the root we put it into the material/phase definition
reducing teh quantity of tracked material definition objects.  

This allows for data models to be swapped out in the face of scientific breaktrhoughs.

```json
{
  "basalt": {
    "solid": {
      "density_kg_m3": 3000.0,
      "specific_heat_capacity_j_per_kg_k": 840.0,
      "thermal_conductivity_w_m_k": 1.8,
      "thermal_transmission_r0_min": 1.0,
      "thermal_transmission_r0_max": 5.0,
      "melt_temp": 1473.0,
      "latent_heat_fusion": 400000.0,
      "boil_temp": 2900.0,
      "latent_heat_vapor": 2000000.0
    },
    "liquid": {
      "density_kg_m3": 2850.0,
      "specific_heat_capacity_j_per_kg_k": 1100.0,
      "thermal_conductivity_w_m_k": 1.2,
      "thermal_transmission_r0_min": 0.5,
      "thermal_transmission_r0_max": 3.0,
      "melt_temp": 1473.0,
      "latent_heat_fusion": 400000.0,
      "boil_temp": 2900.0,
      "latent_heat_vapor": 2000000.0
    },
    "gas": {
      "density_kg_m3": 0.1,
      "specific_heat_capacity_j_per_kg_k": 1200.0,
      "thermal_conductivity_w_m_k": 0.1,
      "thermal_transmission_r0_min": 0.1,
      "thermal_transmission_r0_max": 1.0,
      "melt_temp": 1473.0,
      "latent_heat_fusion": 400000.0,
      "boil_temp": 2900.0,
      "latent_heat_vapor": 2000000.0
    }
  },
  "granite": {
    "solid": {
      "density_kg_m3": 2700.0,
      "specific_heat_capacity_j_per_kg_k": 790.0,
      "thermal_conductivity_w_m_k": 2.5,
      "thermal_transmission_r0_min": 1.0,
      "thermal_transmission_r0_max": 5.0,
      "melt_temp": 1215.0,
      "latent_heat_fusion": 225000.0,
      "boil_temp": 3000.0,
      "latent_heat_vapor": 1750000.0
    },
    "...": "and so on"
  }
}
```
The new data set also has "fusion" temperatures - this is the amount of energy per square kg it takes (in Joules) to 
change phase for a given material. This meaant the conversion price was a simple product of this property and 
the materials fusion const

## Constructor Pattern Evolution

The material system evolution also drove changes in object construction patterns:

## Testing Strategy: From Hardcoded to Data-Driven

The evolution required a complete overhaul of the testing strategy:

### Before: Hardcoded Test Values
```rust
#[test]
fn test_energy_calculation() {
    let energy = 100.0 * 1e9 * 3300.0 * 1200.0 * 1500.0; // Hardcoded calculation
    // Test with hardcoded energy...
}
```

### After: JSON-Calibrated Tests
```rust
#[test]
fn test_temperature_energy_roundtrip() {
    let target_temp = 1673.15;
    let energy_mass = StandardEnergyMassComposite::new_with_temperature(
        MaterialCompositeType::Silicate,
        target_temp,
        1000.0,
        10.0,
    );
    
    assert_abs_diff_eq!(energy_mass.temperature(), target_temp, epsilon = 0.01);
    assert_eq!(energy_mass.phase, MaterialPhase::Liquid); // 1673K > 1600K melting point
}
```

## Performance Considerations

The evolution also addressed performance concerns:

1. **Lookup Optimization**: Material properties cached in static arrays indexed by enum values
2. **Memory Efficiency**: Eliminated complex banking structures in favor of direct calculations
3. **Computational Simplicity**: Reduced phase transition logic from O(n) banking operations to O(1) lookups

## Lessons Learned

### 1. Start Simple, Evolve Gradually
The complex energy banking system was an over-engineering solution to a problem that could be solved more elegantly with direct latent heat calculations.

### 2. Data-Driven Design Wins
Moving material properties to JSON configuration provided flexibility that hardcoded constants never could, enabling rapid iteration and scientific validation.

### 3. Constructor Patterns Matter
Temperature-based constructors proved more intuitive and less error-prone than energy-based constructors, as temperature is the natural parameter scientists think in terms of.

### 4. Test Evolution is Critical
As the underlying algorithms evolved, tests had to evolve from validating hardcoded behavior to validating scientifically accurate behavior.

## Future Directions

The current system opens several avenues for future development:

1. **Pressure-Dependent Properties**: Extending the JSON schema to include pressure effects
2. **Composite Materials**: Modeling mixtures of different material types
3. **Dynamic Property Loading**: Runtime material property updates for different planetary scenarios
4. **Machine Learning Integration**: Using ML to predict material behavior in extreme conditions

## Conclusion

The evolution from hardcoded material constants to a sophisticated JSON-driven material physics engine demonstrates the importance of iterative design in scientific computing. Each generation of the algorithm solved real problems but also revealed new challenges, ultimately leading to a system that balances scientific accuracy, computational efficiency, and developer ergonomics.

The key insight is that in scientific simulation, the data model is often more important than the algorithms themselves. By making material properties first-class, configurable entities, we created a system that can grow and adapt as our understanding of planetary thermal dynamics evolves.

This journey illustrates a broader principle in scientific software development: start with the physics, make the data explicit, and let the algorithms emerge naturally from the requirements rather than imposing complex abstractions prematurely.
