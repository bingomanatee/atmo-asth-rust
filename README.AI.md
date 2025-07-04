# some guidelines for AI

## Testing 

After updating a feature or proces 

1. create test to validate it and reduce regression
2. when all the tests pass remove any comments / printlns you used to do step 1
3. remove any comments that just repeat what is obvious from method / parameter names
4. reduce any code errors that generate warning (ex. unused imports)
5. run the global tests and check to eliminate broken code and tests

## Comments 

The comments we DO need is headers that describe the utility of methods when the method name cannot easily do so.
Also add documentation about the physical laws that the code relies on

## Performance 

Run code in parallel when possible to increase performance 

## Code and units 

1. Use Suffixes to define the physical units 
2. When possible upscale/downscale all units to use the fundamental 
   physical  units of the global scale: kilometers (km, km2, km3), kilograms (kg), years, kelvin (k) and joules (j)
3. when a method takes more than one parameter create "struct params" to document wich units the method requires and to prevent order errors
4. Numbers should be f64 whenever possible 
   to reduce the requirement of `as` conversion 
5. the exception are array indices that should be usize
6. when you have deep low level computations to modify or mutate a 
   instance, turn it into methods or utility library functions 
7. prefer match branches to if a = b {} branches. 
8. there are some physical laws that are based on non-standard units 
   in those cases use or create conversion constants in the `constants.rs` file 
9. Anything having to do with materials should be embedded in the `materials.json` file

## Config files

When possible realize scientific values like chemical compound or material properties in JSON config files. 

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