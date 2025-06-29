# Asthenosphere, Lithosphere, Atmosphere and Space

This codebase simulates the flow of energy from a planet's core to the layers of the planet to the surface and
outwards into space. This is designed to produce the creation of the Lithosphere, tectonic plates, and ultimately
continental bodies and mountains. 

## A note on what we are _not_ simulating

Surface features are what we primarily care about. That means inner core mechanics do not matter - only
the amount of radiant energy they produce. So the top few hundred kilometers of layers are the only thing we are
simulating and the rest is just represented as an output of energy (Joules) from an un-simulated source as a numeric 
value. 

## The layers and the Release Priorities

## The Planet

### Radius 

Planetary radius drives everything; at this stage it is the only planetary property we compute around. To allow simulations
of other worlds we vary every unit of volume etc. against planetary radius. 

### Mapping the surface

We are using [H3](https://h3geo.org/)/[H3o](https://docs.rs/h3o/latest/h3o/) mapping to define the planet as an indexed series of Columns each corresponding to a identity of the location on the planet. H3 was Uber's original mapping system dividing the 
planet into a series of hexagons with hexidesimal ids (happy coincidence) for each location. The mapping has a series of nesting
starting with a 122 base grid; each level of resolution below that has 7 hexagons for every upper hex, meaning the next resoliton
level (resolution one) has 842 hexagons then 5882 for level two and so on 

### Asthenosphere

**Phase One** this is the most important early layer and we are simulating its energy and volume mass as a series of
_energyMass_ instances in a Column. Columns are located on h3 cells; that determines their real world 3d location and lat and longitude. 

#### Columns of data
Each cell has an area driven by their radius and while they are technically hexagons we can treat them as cylinders 
for easy volume computations. While technically they are _cones_ representing the slight increase and decrease in radius when
you go up and down a few  hundred km is not important enough to account for. 

### Layers, temperature and energy

Each column has a series of layers stacked on top of each other. If you fix the height at an arbitrary number (10..50) 
you can then account for the volume and therefore mass of each cell. This matters because temperature (kelvin) is a product of 
Mass (volume x density), energy (joules) and Specific heat. Density is mass / unit of volume (in our case km^3). 

This all flattens out to:or rather volme x density x joules 

```
Temp(kelvin) = volume (Km ^ 3) x density (Mass (kg) / volume (km ^ 3) * Specifig heat (joules / (Kg * Kelvin) )
```

converting energy to temperature is fundamental because materials have melting points and cooling points at which 
material changes state and therefore thermal criteria like energy conductivity and density so you have to track
when matter converts into different states; also matter doesn't automatically change state at various temperatures;
they require consumption or expulsion of energy to transition - it takes time and takes (or produces) energy.

#### Conductivity and Energy Capacity 

Every material has a conductivity rating which determines how fast it can radiate energy to its hotter or colder neighbors,
and a capacity which determines how much energy (Joules) it can contain. Energy past this saturation point radiates outward 
to neighbors and ultimately to space or atmosphere. 

### Density 

for the most part we assume a fixed density for solid masses (astheosphere, lithosphere) but not for Atmosphere 
which as an exponential drop off in maximum density. This in turn also defines the atmosphere's temperature and energy
capacity allowing the atmosphere to further insulate the planet as the energy it radiates into space is balanced by the 
energy that the solid layers below radiate into it. 

### Atmospheric vs. Surface Energy Radiation

Surface radiation of atmosphere into space depends on the Stefan-Boltzmann law which is driven off the surface area 
of the planet or atmosphere and the respective temperatures and material properties of the atmosphere and the vacuum of 
space. This means that the amount of radiation into space is not driven by the volume of atmosphere beneath the atmosphere
only its density, temperature and material properties. 

Surfaces on the other hand DO drive energy based on those properties AND their volumes so there is a different dynamic
for solid bodies and atmospheric bodies.

See [SCIENCE.formula.md](SCIENCE.formula.md) for the specific formulaic functions for these interactions

## Layer Evolution: Asthenosphere to Lithosphere
- **Early stage**: no lithosphere, so asthenosphere provides the radiating slab.
- **Later stage**: lithosphere thickens; if its thickness exceeds skin depth, radiation applies only to the lithosphere portion above the skin depth; deeper asthenosphere remains via conduction.

## Atmosphere Buffering
- **Atmosphere mass per area** influences its heat capacity: more mass → more thermal inertia.
- Atmospheric layers exchange with the surface via radiation (and optionally convection), and radiate upward to space, reducing direct surface-to-space loss.

## Layer to Layer convection

Energy travels from the Earth's core to the surface; these mechanics are affected by surface thickness as well unlike the final
radiation which is only driven by the surface area. 



