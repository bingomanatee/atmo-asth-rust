
# Coding Conventions and Units

## Units 

* distance units default to km;
* steps in the simulation are configurable via `years_per_step`. We began with the assumption of
  1 MIO years/step but are adjusting to 100,000 years as the standard; Years per step varies with each simulation 
* "Years" in this context are earth years; the planetary orbit time and rotation time aren't part of this model yet
* The H3o of the simulation is configurable as a property but the default assumption is Resolution::Two; this system is built around resolution 1..3 based systems, performnce is not guaranteed for a finer grained resolution simulation
* with a variable resolution and planet size constants are either resolution independant like temperature or have the presumption of an earth-sized planet and should be scaled based on the relative radius of earth 
* cells' heat quotient are in Joules; this is the default energy measurement in the system.
* Some things like lithosphere generation rates are measured in celsius or Kelvin which are different numerically but the same scale of units. 

## The H30 library 

Resolution and cell ids are based on the h30 library, a variant of Uber's h3 library for planetary location. in H3/o each region of the planet is divided into a subdivided icoseles sphere starting at level 0 with 122 cells and dividing each cell into seven hexagonal sub-cells. 

Cells are identified by a binary string; each cell has one parent, seven children and six neighbors using a hexagonal layout. Even though cells are hexagons, the columns are treated as cylinders for easier math. 

## Code

* impl properties have suffixes that indicate their unit in the suffix name
* Most constructors/new methods take a parameter of objects; in the parameter signature the units suffix are omitted and sometimes shorter names are used (res for resolution)
* shared properties or reused values are cached or denormalized downwards to promote self sufficient units; eg., the resolution property in Simulation is propagated to the cells as is layer height and count. 
* Given the amount of inter-addition and multiplication, all physical units are measured in f64

### AsthCell

AsthCell is short for AsthenosphereCell represents a vertical stack of magma material underneath the lithosphere and the lithosphere material abovve it. it has layers which is a "top to bottom" stack and lithospheres which is a "bottom to top" stack. the reason is that most often you are concerned about the _top layer of the asthenosphere_ and the _bottom layer of the lithosphere_. 

### `.layers` (of the Asthenosphere)
layers start at 0 for the surface to layer `cell.layer_count - 1`; 
they are all have the same radius and areas; technically they are hexagonal but for all practical purposes they are cylinders. Each layer is `.layer_height_km` kilometers high; the heights are maintained with the presumption that teh top and bottom may push out above or below until lithosphere mass levels it down at which point it compresses. So for practical purposes the layers remain parallel to each other. 

The layer(n) method returns a tuple(current, next) layer data for a given layer. 

### `.lithospheres`

Lithosphere is a "bottom-up" system of lithosphere in a column. _almost all columns won't have more than one entry in the lithospheres /
lithospheres_next Vecs`. 

The lithosphere() method reutrns a tuple (current, next) of the lithosphere

## Spatial location h30 and cell indexes

This library leans heavily on h30, a Rust variant of Uber's H3 library for planetary locations based on subdivided isosceles polygonal spheres. 
Locations on the planet are modelled in depth in "columns"; at this scale we are considering each of the layers to have the same volume and top surface area although they are slices of a cone going from the center of the planet

H3o relates cells on the planet surface to each other as neighbors. Neighbor ids are stored in `neighbor_cell_ids` 