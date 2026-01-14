from juliacall import Main as jl

# Run this setup script once before running the agent code
# Initialize the package manager within the PythonCall environment
jl.seval("import Pkg")

# Install required packages
# This ensures the solver is available when you run the agent
print("Installing Julia dependencies for PythonCall environment...")
jl.Pkg.add("JuMP")
jl.Pkg.add("PATHSolver")
jl.Pkg.add("LinearAlgebra")
print("Dependencies installed.")