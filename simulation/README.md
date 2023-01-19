# echino-sim
Environment for simulating pleurocystitid robot using the new Dojo simulator package from RExLab at CMU. It allows fast and efficient contact rich simulation and provides full gradients for trajectory optimization, reinforcement learning, etc. 

## Dependencies

[Julia](https://julialang.org/downloads/) - tested with 1.6.3 but latest version should work.

[Dojo](https://github.com/dojo-sim/Dojo.jl) - `Dojo` can be added via the Julia package manager (type `]`):
```julia
pkg> add Dojo
```

## Usage

The main script for performing simulations and optimizations is trajopt/pleuro_trajopt.jl. 
