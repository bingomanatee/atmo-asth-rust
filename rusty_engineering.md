Week Two of Rust: "I'm sorry Dave"

I've spent several months trying to get a Tectonic Plate demo out of javascript; in frustration I changed to rust and I have to say
the progress was much faster than with javascript due to the built in structuring of Rust and to be honest some great assistance from
Claude. but there were still some hitches. 

my initial goal was to make sure that the sytem can operate over a large data set, and do it fast. So I anted to put the data in a
store, pull it out as needed, process it and put it back. And I wanted to dos o with a multi-processor arrangement for rapid execution. 
One of JavaScript's bit limitations is that it is a single-threaded environment and that meant performance had a finite ceiling. 

With Rust I tried using Rayon and Rocks DB; Rocks is a fast key-value store by Facebook. but there is a little bit of a limit; like
SQLite, RocksDB is designed for a single tenant system, so having multiple clients write to it is not part of its design. Also, I 
found out that reading and writing to RocksDB was the driving force slowing doen operations. So instead I opted to put operations 
completely in memory, writing back to RocksDB in non-blocking threads. This vastly increased the performance of updates. 

## Visualizing the operations

In order to validate the design I had Rust write to PNGs with the help of AI. I soon found out that while AI is good at rapid 
design of visualizations it also made design decisions that did not facility the fastest executions; for instance instead of developing
a color table, it constantly executed the color with every pixel. Enforcing color table caching and retaining the h30 map of coordinates 
to ppxels signicantly increased the renering time of PNGs. 

# The structure: Solid as a Rocks

I started running the simulation out of Rocks. At first I tried to overengineer the system by constantly reading into and out of rocks. 
in faireness I was also trying to minimize the amount of data I had in memory but as I was using Rayon to thread things I thought I could 
interate over the store while operating on small units. This proved unfeasible but once I moved the sim into a pure memory operation it was
virtually instance; the only slowdown was baking the data into a voronoi graph of heat over space and that is a huge computational task. 

We chose RocksDB to offload state from RAM—freeing memory for core simulation work—and to produce a durable on-disk record 
of each timestep’s data, enabling post-run analysis and visualization.

## Notes things I like about Rust

1. its type system is native and very easy to read and use
2. it forces you to think about all the possible results of a fail-able activity 
3. its got a native test sytem you can write straight into the source  files
4.its fast
5. it uses snake case
6. it uses as little syntax as possible - no unnecessary parentheses around if statements of for's

## Things I wish wouldn't happen 

1 the sharing rules often makes you write code that is more verbose than necessary
2. it has very verbose ways of combining types; eg why not multiply a f32 by an f64 or divide a f64 by a u32?
3. no _compact_ inline if then else ( A ? B : C ); its if  then else _does_ return a value but it is a multi line expression which bloats the code. 
4. no inheritance of structs; traits provide some structural inheritance but without a lot of the flexibility (no props in traits for instance)
5. do fault arguments / parameters
6. overly rigid borrowing rules; significant time and effort is taken to allow the memory conservative handling of properties 

## Developing with multiple iterations 

I spent several months with the JS implementation. One thin i learned is that tolerating bad architecture was ultimately costly so instead 
I developed several implementations improving on clarity, accuracy and speed with each one. AI helps a lot in spinning out code but it often
opts for easy hacks ofer solid architecture so its important to get down in the weeds wth the code it produces.

*The first iteration read and wrote to RocksDB as mentioned earlier. It worked but I wanted faster throughput.
* the next implementation kept all the data in memory at once; i was dubious that 5000 records in memory being updated was a good design 
  but I went with it. I used a system of linked cells, keeping the current one in one reference and the next one in a doubly linked reference.
  The overhead of Rust's' reference system proved bulky so I tried again.
* the third and current one uses a structs with two references, current and next. This is simple, easy to traverse and lower memory.

### The quantum change: Layers of magma

There is a big advantage to using AI: quantum changes can be off-ported to the assistant. The biggest example was that the model used to be single-level
but when multi-level simulation was called for the energy and volume registry had to be made into an array; that meant all the accessers and 
updaters had to change their access pattern. These sort of changes are as tedious as they are error prone. AI reduces the overhead of these 
kind of shifts; Rust and its strongly typed nature and a strong set of unit tests help too. 

## On Simulation 

Some notes on earlier iterations I focused on implementation - how I was going sto structure the data and changes, store information, etc. That meant 
by the time I discovered my sim was out of balanced it was so large and sophisticated that tuning it was very difficult given the volume of code.
In this latest iteration I am focused on defining the formula before the structure so that when I do make structure I can test it against the output from my
simulations to ensure system balance

Some useful notes on developing a simulation

### Know your units 

Science has a bunch of units and especially with metric system based units (which you should use) many orders of magnitude variations or synonyms- for instance 
Kelvin against Celsius (1 Kelvin = 1 Celsius + 273.15 ), meters vs km., Joules vs Watts. By choosing a unit basis for the whole thing (Kelvin, Watts and Km) 
I can reduce the number of conversion steps when my systems interact and focus on the formula. 

### Find the Balance

Mr. Myagi was right - find the forces that balance each other out and find equilibrium. Specifically find the largest input and outputs and ensure that in the end
they balance to the point you want them to. In my case the heat balance in Earth is dominated by the radiant core and the radiation of surfaces out to the core. 
There are many smaller modifiers - atmosphere, solar energy, 

At some point the inputs and outputs will find an equilibrium and the temperatures of the various layers will equalize; this is not the same as _becoming equal_.
Think of it as a flow of water from the mountain top (planetary core) to the ocean (space); the water in the river will have different heights, but it is the 
energy of these heights which push the water downward. As the volumes find their balance, the rivers will maintain a constant depth at any given distance 
from the mountain and therefore a constant volume, even as the water that makes up this volume is constantly in motion down stream. 
But in the early stages this is not true;
the volumes may change and be chaotic or overflow or be dry at given locations until the flow of the water stabilizes. If you want to simulate both the creation and the final equilibrium you have to understand the fixed parts of the equation (volume, distance, time, scientific laws) and the variables (rates of conductivity, density thresholds, input rate) to reproduce the outcome. 

In our case we want the temperatures to produce a layer of lithosphere on top of a layer of magma (asthenosphere) such that the lithosphere is not too thick or too thin. That means the energy in the system has to keep the lower levels above the melting points of the material while not overflowing and melting _all_ of the 
material so _some_ lithosphere is created.

Also, when lithosphere is created, the energy dynamics will change because the surface properties of the lithosphere are different 
which will change the energy flow, increasing the asthenosphere's temperature which may then melt the lithosphere below it. So there will be a cycle of lithosphere forming, heading up the asthenosphere, then melting and allowing the energy to flow more freely, cooling it down. After a certain period, the lithosphere should remain at least partly in existence so that the atmosphere and oceans can form -- again changing the equation of the flow of energy. 

Another point of equilibrium was to ensure that the overall temperatures ended up in the balance point between being too cool and allowing the entire 
layer system to freeze over into magma, and too hot, preventing any lithosphere from forming at all. This involved tuning some constants which were not 
absolutely locked down by scientific laws which adjusted heat transfer through system nodes. 

### Adding complexity 

Once you find the balance you can allow variations to compound the system by _normalizing_ their effect. For instance in the initial system I didn't allow the differences in temperature to drive the flow of energy; when I added this feature by adjusting the speed around a given point of normality as such: 

```
energy FLow = /* ... existing formula */  * ((a.temp - T.temp).abs()/10).pow(1.5).min(0.66)
```

This meant that extreme temperature differences around the core would flow energy faster, pushing out the asthenosphere, while slower energy differences would
crete less flow.  

- the minimum rate of 0.66 ensures that even minor differences in energy would allow some flow
- the power exponent meant extreme temperature differences would move exceptionally fast while similar temperatures don't transfer energy as fast

the overall effect of this is to achieve the same balance but to do it with realistic accelleration and deceleration based on "temperature pressure"

Doing this can maintain the overall equilibrium topology while stretching out certain parts - for instance adjusting where the asthenosphere stops and the lithosphere starts.  

### Visualize analyze and revise

Earlier I was storing data in RocksDB exclusively. As I started to back out to the experimental realm I put out summations into CSV files so I could graph and
observe progress over time. This paid off in revealing overal treands. Attempting to balance the system to achieve 