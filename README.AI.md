# some guidelines for AI

## Testing

After updating a feature or process

1. create test to validate it and reduce regression
2. when all the tests pass remove any comments / printlns you used to do step 1
3. remove any comments that just repeat what is obvious from method / parameter names
4. reduce any code errors that generate warning (ex. unused imports)
5. run the global tests and check to eliminate broken code and tests
6. Prefer tests without println noise/debug output in test functions
7. Prefer extracting inner algorithm logic into separate testable methods
8. Prefer tests to use real material data from material_composite file rather than hardcoded values
9. Prefer tests to be calibrated to work with the JSON-based material system
10. Prefer adding unit tests around critical energy banking scenarios when fixing bugs

## Comments 

The comments we DO need is headers that describe the utility of methods when the method name cannot easily do so.
Also add documentation about the physical laws that the code relies on

## Performance 

Run code in parallel when possible to increase performance 

## Code and units

1. Use Suffixes to define the physical units
2. When possible upscale/downscale all units to use the fundamental
   physical  units of the global scale: kilometers (km, km2, km3), kilograms (kg), years, kelvin (k) and joules (j)
3. when a method takes more than one parameter create "struct params" to document which units the method requires and to prevent order errors
4. Numbers should be f64 whenever possible
   to reduce the requirement of `as` conversion
5. the exception are array indices that should be usize
6. when you have deep low level computations to modify or mutate a
   instance, turn it into methods or utility library functions
7. prefer match branches to if a = b {} branches.
8. there are some physical laws that are based on non-standard units
   in those cases use or create conversion constants in the `constants.rs` file
9. Anything having to do with materials should be embedded in the `materials.json` file
10. Prefer destructuring to get first items from iterators instead of .next().unwrap()
11. Prefer match statements over unwrap_or_else when handling returns that should panic
12. Prefer explicit 'kelvin' terminology instead of ambiguous 'temperature' in function names
13. Prefer panicking over returning Options when data should be guaranteed to exist
14. Prefer extracting redundant arrays into constants rather than repeating them inline
15. Prefer avoiding unnecessary cloning and use borrowing when possible
16. Prefer 'self' instead of 'Self::' for method calls within the same struct

## Config files

When possible realize scientific values like chemical compound or material properties in JSON config files.

## Thermal System Architecture

1. Temperature (in Kelvin) is derived from volume and energy, not an independent parameter
2. Prefer thermal simulation design with fixed temperature boundaries: core heat source and space heat sink layers
3. Volume calculations should use height * area formula, with 1 km² area as the base unit
4. Prefer thermal simulation architecture where each cell creates PendingUpdates for a PendingChanges queue
5. Prefer binary exchange thermal diffusion algorithm with pairwise energy exchanges based on distance falloff
6. Earth core radiance should be modeled as 2.52e12 J per square km per year with initial heat flux of 300 mW/m²
7. Energy transfer coefficient = conductivity × 0.00004, with conductivity values of 2.5-3.5 (average 3.0)
8. Temperature and material state must never be inconsistent - constructors should take temperature and call resolve_state_to_temp

## Communication 

Don't waste time telling me how great my ideas are or telling you what you are going to do or repeating what I ask you do do - just do your work. 

Make sure your work is validated (see Testing) 
before you claim you are "done."

In general - say as little as possible - don't echo back code to the conversation - the thinner your chat log is the more we can do without restarting. 

## Task Completion Checklist

* minimize or remove comments or println!
* add coverage for new code 
* remove tests for deprecated code 
* remove deprecated code
* validate entire codebase (check)
* validate entire test suite