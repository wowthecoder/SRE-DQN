
using JuMP
import PATHSolver
using LinearAlgebra

"""
    solve_strategically_robust_bimatrix_game(U1, U2, epsilon_values, num_repeats; verbose=false)

Solves a strategically robust bimatrix game formulated as a mixed complementarity problem (MCP)
using the PATH solver. The function repeatedly attempts to solve the MCP with different random
starting points to (hopefully) find all distinct solutions.

# Arguments
- `U1::AbstractMatrix`: Utility matrix for player 1.
  Dimensions: (nA, nB) where nA = number of actions for player 1.
- `U2::AbstractMatrix`: Utility matrix for player 2.
  Expected dimensions: (nB, nA). If U2 is supplied in the same orientation as U1,
  it is assumed to be transposed and will be fixed.
- `epsilon_values::Vector{<:Real}`: A 2-element vector of robustness parameters,
  where `epsilon_values[1]` is used for player 1 and `epsilon_values[2]` for player 2.
- `num_repeats::Int`: The number of random starting points (initializations) to try.

# Keyword Arguments
- `verbose::Bool`: If true, detailed output is printed for each run.

# Returns
A tuple `(solutions_p, utilities_sr, utilities_nominal)` where:
- `solutions_p` is a vector of dictionaries containing the equilibrium strategies (the probability vectors)
  for each player (keys `:p1` and `:p2`).
- `utilities_sr` is a vector of two-element vectors containing the strategically robust utilities for players 1 and 2.
- `utilities_nominal` is a vector of two-element vectors containing the nominal utilities computed at the solution.
"""
function solve_strategically_robust_bimatrix_game(U1, U2, epsilon_values, num_repeats; verbose=false)
    # --- Fix U2 if necessary: if U2 has the same dimensions as U1, assume it is transposed.
    if size(U2) == size(U1)
        U2 = transpose(U2)
    end

    # Determine the number of actions for each player.
    nA = size(U1, 1)  # number of actions for player 1
    nB = size(U2, 1)  # number of actions for player 2

    actionsA = 1:nA
    actionsB = 1:nB

    # Define the distance functions.
    # Here we assume a simple distance metric: 1 for off-diagonals and 0 on the diagonal.
    dist_A = ones(nB, nB) .- I(nB)
    dist_B = ones(nA, nA) .- I(nA)

    # Containers to store solutions and utility values.
    solutions_p = Vector{Dict{Symbol, Vector{Float64}}}()
    utilities_sr = Vector{Vector{Float64}}()
    utilities_nominal = Vector{Vector{Float64}}()

    # --- Generate Random Starting Points ---
    starting_points_p = rand(num_repeats, nA + nB)
    total_other = 2 + nB + nA + 2 + nB^2 + nA^2
    starting_points_other = 100 * rand(num_repeats, total_other) .- 50

    # --- Set Up the Optimization Model ---
    m = Model(PATHSolver.Optimizer)
    set_optimizer_attribute(m, "minor_iteration_limit", 100000)
    set_optimizer_attribute(m, "convergence_tolerance", 1e-10)
    set_optimizer_attribute(m, "cumulative_iteration_limit", 100000)
    set_optimizer_attribute(m, "nms_searchtype", "path")
    if !verbose
        set_silent(m)
    end

    # Decision variables.
    @variable(m, p1[i in actionsA] >= 0)
    @variable(m, p2[j in actionsB] >= 0)

    @variable(m, lambda1 >= 0)
    @variable(m, lambda2 >= 0)

    @variable(m, xi1[j in actionsB])
    @variable(m, xi2[i in actionsA])

    @variable(m, eta1[i in actionsB, j in actionsB] >= 0)
    @variable(m, eta2[i in actionsA, j in actionsA] >= 0)

    @variable(m, kappa1)
    @variable(m, kappa2)

    # --- Complementarity Conditions ---
    @constraint(m, [i in actionsA],
        -kappa1 - sum(eta1[k, l] * U1[i, l] for l in actionsB, k in actionsB) ⟂ p1[i])
    @constraint(m, [j in actionsB],
        -kappa2 - sum(eta2[k, l] * U2[j, l] for l in actionsA, k in actionsA) ⟂ p2[j])
    @constraint(m, epsilon_values[1] - sum(eta1[k, l] * dist_A[k, l] for l in actionsB, k in actionsB) ⟂ lambda1)
    @constraint(m, epsilon_values[2] - sum(eta2[k, l] * dist_B[k, l] for l in actionsA, k in actionsA) ⟂ lambda2)
    @constraint(m, [j in actionsB],
        -p2[j] + sum(eta1[j, k] for k in actionsB) ⟂ xi1[j])
    @constraint(m, [i in actionsA],
        -p1[i] + sum(eta2[i, k] for k in actionsA) ⟂ xi2[i])
    @constraint(m, [i in actionsB, j in actionsB],
        -xi1[i] + sum(U1[k, j] * p1[k] for k in actionsA) + lambda1 * dist_A[i, j] ⟂ eta1[i, j])
    @constraint(m, [i in actionsA, j in actionsA],
        -xi2[i] + sum(U2[k, j] * p2[k] for k in actionsB) + lambda2 * dist_B[i, j] ⟂ eta2[i, j])
    @constraint(m, 1 - sum(p1[i] for i in actionsA) ⟂ kappa1)
    @constraint(m, 1 - sum(p2[j] for j in actionsB) ⟂ kappa2)

    # --- Main Loop: Try Different Starting Points ---
    for k in 1:num_repeats
        # Set starting values for probability variables.
        for i in actionsA
            set_start_value(p1[i], starting_points_p[k, i])
        end
        for j in actionsB
            set_start_value(p2[j], starting_points_p[k, nA + j])
        end

        # Unpack starting_points_other.
        idx = 1
        set_start_value(lambda1, starting_points_other[k, idx]); idx += 1
        set_start_value(lambda2, starting_points_other[k, idx]); idx += 1
        for j in actionsB
            set_start_value(xi1[j], starting_points_other[k, idx]); idx += 1
        end
        for i in actionsA
            set_start_value(xi2[i], starting_points_other[k, idx]); idx += 1
        end
        set_start_value(kappa1, starting_points_other[k, idx]); idx += 1
        set_start_value(kappa2, starting_points_other[k, idx]); idx += 1
        for i in actionsB, j in actionsB
            set_start_value(eta1[i, j], starting_points_other[k, idx]); idx += 1
        end
        for i in actionsA, j in actionsA
            set_start_value(eta2[i, j], starting_points_other[k, idx]); idx += 1
        end

        optimize!(m)

        if verbose
            println("Solution for starting point $k:")
            println("p1: ", value.(p1))
            println("p2: ", value.(p2))
            println("lambda1: ", value(lambda1), ", lambda2: ", value(lambda2))
            println("xi1: ", value.(xi1))
            println("xi2: ", value.(xi2))
            println("kappa1: ", value(kappa1), ", kappa2: ", value(kappa2))
            println("eta1: ", value.(eta1))
            println("eta2: ", value.(eta2))
            println()
        end

        if is_solved_and_feasible(m)
            sol = Dict{Symbol, Vector{Float64}}()
            sol[:p1] = [round(value(p1[i]), digits=4) for i in actionsA]
            sol[:p2] = [round(value(p2[j]), digits=4) for j in actionsB]

            utility1_sr = sum(value(p2[j]) * value(xi1[j]) for j in actionsB) - value(lambda1) * epsilon_values[1]
            utility2_sr = sum(value(p1[i]) * value(xi2[i]) for i in actionsA) - value(lambda2) * epsilon_values[2]

            utility1_nominal = dot([value(p1[i]) for i in actionsA], U1 * [value(p2[j]) for j in actionsB])
            utility2_nominal = dot([value(p2[j]) for j in actionsB], U2 * [value(p1[i]) for i in actionsA])
            
            if !(sol in solutions_p)
                push!(solutions_p, sol)
                push!(utilities_sr, [utility1_sr, utility2_sr])
                push!(utilities_nominal, [utility1_nominal, utility2_nominal])
            end
        end
    end

    return solutions_p, utilities_sr, utilities_nominal
end
