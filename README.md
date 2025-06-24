
# Coding Conventions and Units

## Units 

* distance units default to km;
* steps in the simulation are in MIO Years
* "Years" in this context are earth years; the planetary orbit time and rotation time aren't part of this model yet
* The resolution of the simulation is configurable as a property but the default assumption is Resolution::Two
* with a variable resolution and planet size constants are either resolution independant like temperature or have the presumption of an earth-sized planet and should be scaled based on the relative radius of earth 
* cells' heat quotient are in Joules; this is the default energy measurement in the system.
* Some things like lithosphere generation rates are measured in celsius or Kelvin which are different numerically but the same scale of units. 

## Code

* impl properties have suffixes that indicate their unit in the suffix name
* Most constructors/new methods take a parameter of objects; in the parameter signature the units suffix are omitted and sometimes shorter names are used (res for resolution)
* shared properties or reused values are cached or denormalized downwards to promote self sufficient units; eg., the resolution property in Simulation is propagated to the cells as is layer height and count. 
* Given the amount of inter-addition and multiplication, all physical units are measured in f64

### AsthCell

AsthCell is short for AsthenosphereCell represents a vertical stack of magma material underneath the lithosphere and the lithosphere material abovve it. 

### `.layers`
layers start at 0 for the surface to layer `cell.layer_count - 1`; 
they are all of the same radius and areas; technically they are hexagonal but for all practical purposes they are cylinders.each layer is `.layer_height_km` kilometers high; the heights are maintained with the presumption that teh top and bottom may push out above or below until lithosphere mass levels it down at which point it compresses. So for practical purposes the layers remain parallel to each other. 

## Spatial location h30 and cell indexes

This library leans heavily on h30, a Rust variant of Uber's H3 library for planetary locations based on subdivided isosceles polygonal spheres. 
Locations on the planet are modelled in depth in "columns"; at this scale we are considering each of the layers to have the same volume and top surface area although they are slices of a cone going from the center of the planet

H3o relates cells on the planet surface to each other as neighbors. Neighbor ids are stored in `neighbor_cell_ids` 