# AGENTS.md — Key Paper Summaries for the Deep SRE Project

This file provides context on the three foundational papers for this project. Any AI agent working on this codebase should read this file to understand the mathematical foundations, algorithms, and experimental setups that the project builds on.

---

## 1. Deep Q-Learning for Nash Equilibria: Nash-DQN
**Authors:** Philippe Casgrain, Brian Ning, Sebastian Jaimungal (2022)
**Core contribution:** A computationally tractable deep RL algorithm for learning Nash equilibria in general-sum stochastic games with continuous state and action spaces.

### 1.1 Problem Setting
- N agents in a stochastic game with continuous states x ∈ ℝ^{d_x} and continuous actions u_i ∈ ℝ^{d_i}.
- Each agent i chooses a deterministic Markov policy π_i: X → U_i to maximise its discounted cumulative reward R_i(x; π_i, π_{-i}) = E[Σ_{t=0}^∞ γ_i^t r_i(x_t, π_{i,t}, π_{-i,t})].
- Agents seek a Nash equilibrium: a joint policy π* such that no agent can unilaterally improve their objective.

### 1.2 Nash-Bellman Equation
The value function V(x) = (V_i(x))_{i∈N} and Q-function Q(x;u) = (Q_i(x;u_i,u_{-i}))_{i∈N} satisfy:

    V(x) = Nash_{u∈U} Q(x;u) = Nash_{u∈U} { r(x;u) + γ E_{x'~p(·|x,u)}[V(x')] }

The Nash operator N_{u∈U} f(u) maps a collection of N concave functions to their Nash equilibrium value.

### 1.3 Locally Linear-Quadratic Q-Function Decomposition
**This is the key architectural innovation.** The Q-function is decomposed as:

    Q̂_i^θ(x; u) = V̂_i^{θ_V}(x) + A_i^{θ_A}(x; u)

where the advantage function has a locally linear-quadratic (LQ) form:

    A_i^θ(x; u) = −(u_i − μ_i(x), u_{-i} − μ_{-i}(x))^T P_i(x) (u_i − μ_i(x), u_{-i} − μ_{-i}(x)) + (u_{-i} − μ_{-i}(x))^T ψ_i(x)

with block structure:

    P_i(x) = [ P_{11,i}(x)  P_{12,i}(x) ]
             [ P_{21,i}(x)  P_{22,i}(x) ]

**Critical constraints and properties:**
- P_{11,i}(x) must be positive-definite for all x (ensures Q_i is concave in u_i). Enforced via Cholesky: P_{11,i} = L_{11,i} L_{11,i}^T where the network outputs the lower-triangular L_{11,i}.
- P_{12,i} = P_{21,i}^T (WLOG, since only the symmetric combination matters).
- μ^θ(x) represents the Nash equilibrium action at state x. At this point, the advantage is zero: A(x; μ(x)) = 0.
- Therefore V̂(x) = Nash_{u∈U} Q̂(x;u) and μ(x) = arg Nash_{u∈U} Q̂(x;u) — both readable directly from network outputs.

**What the neural networks model (separately):**
- V̂^{θ_V}(x): ℝ^{d_x} → ℝ^N (value function for each agent)
- μ^θ(x): ℝ^{d_x} → ℝ^{d_u} (equilibrium actions)
- P_i^θ(x): state-dependent quadratic coefficient matrices
- ψ_i^θ(x): ℝ^{d_x} → ℝ^{d_{-i}} (linear cross-agent term)

### 1.4 Loss Function
For observed transition triples (x_m, u_m, x'_m):

    L(θ) = (1/M) Σ_m ||V̂(x_m) + A(x_m; u_m) − r(x_m, u_m) − γ V̂(x'_m)||²

Since A(x'; μ(x')) = 0 at the Nash equilibrium, the target simplifies to γV̂(x'_m).

**Optimisation improvement (important):** Adding L¹ regularisation on ψ^θ(x):

    L̃(y_m, θ_V, θ_A) := L̂(y_m, θ_V, θ_A) + β ||ψ^θ(x)||

with optimal β = 100 (found by grid search over 1000 random simulations).

### 1.5 Actor-Critic Training
Parameters θ = (θ_V, θ_A) are optimised by alternating between minimisation over θ_V and θ_A. Uses a replay buffer of transition tuples and minibatch SGD.

### 1.6 Simplifying Game Structures
- **Label invariance:** If the game is symmetric under permutation of agent labels, all agents share the same V̂, μ, P, ψ functions. Uses permutation-invariant neural networks (Deep Sets architecture, motivated by Arnold-Kolmogorov representation theorem).
- **Identical preferences:** If agents share the same reward function, modelling reduces by a factor of N.
- **Sub-population invariance:** Groups of agents (e.g., cooperators vs non-cooperators) can share parameters within groups.

### 1.7 Algorithm Pseudocode (Algorithm 5.1)
```
Input: B episodes, minibatch size M̂, N game steps, exploration noise {σ_b}
Initialise: Replay buffer D, parameters (θ_A, θ_V)
For episode b = 1 to B:
  Reset simulation, get x_0
  For game steps t = 1 to N:
    Select actions u ← μ^{θ_A}(x) + ε, ε ~ N(0, σ_b I)
    Observe transition y_t = (x_{t-1}, u, x_t)
    Store D ← y_t
    Sample Y = {y_i}_{i=1}^{M̂} randomly from D ∪ {y_t}
    Optimisation step of (1/M̂+1) Σ L̂(y, θ_V, θ_A) over θ_V
    Optimisation step of (1/M̂+1) Σ L̂(y, θ_V, θ_A) over θ_A
Return (θ_A, θ_V)
```

### 1.8 Experimental Setup: Statistical Arbitrage Trading Game
**Environment:** N agents trading a single asset with a stochastic price process affected by their actions.
- Agent i chooses a trading rate ν_{i,t} ∈ ℝ (continuous, scalar action) at discrete decision points t_1, ..., t_M.
- Inventory: q_{i,t} = q_{i,0} + ∫_0^t ν_{i,s} ds. Inventories are private; total order flow is public.
- At terminal time T, remaining inventory incurs a penalty. Agents must liquidate.
- **Asset price process:**
  - Price follows a mean-reverting process S_t with drift affected by aggregate trading.
  - Includes permanent price impact and transient price impact Y_t.
- **Agent objective (reward):** Each agent maximises expected P&L minus risk penalty, accounting for execution costs, inventory costs, and market impact.

**State features (inputs to neural networks):**
1. Price S_t (scalar)
2. Time t (scalar)
3. Total order flow Σ_{i∈N}(q_{i,0} − q_{i,t}) (scalar)
4. Cumulative transient price impact Y_t (scalar)

All features are assumed non-label-invariant in this experiment.

**Model parameters used:**
| κ | θ | σ | γ | ρ | η | b₁ | b₂ | b₃ | ΔT |
|---|---|---|---|---|---|----|----|----|----|
| 0.1 | 10 | 0.01 | 0.02 | 0.5 | 0.05 | 0.1 | 0.1 | 0 | 0.5 |

**Network architecture:**
- Advantage network: permutation-invariant layer (3 hidden layers × 20 nodes, SiLU) feeding into main network (4 hidden layers × 32 nodes). Outputs μ, P, ψ.
- Value network: 4 hidden layers × 32 nodes. Outputs V̂ for all agents.
- Optimiser: Adam with weight decay 0.001, learning rate 0.003.
- Training: max 20,000 iterations, early stopping if no improvement in last 3,000 iterations.
- Minibatch: 10 full episodes.

**Baseline: Fictitious Play (FP).** Each iteration computes the best response to the empirical average of opponents' strategies. Each FP iteration uses approximately the same resources as a full Nash-DQN training cycle.

**Results:** Nash-DQN produces policies essentially identical to FP. The null hypothesis that means differ cannot be rejected at the 5% level, confirming Nash-DQN learns the correct Nash equilibrium while being significantly more data-efficient.

### 1.9 Key Limitations
- The LQ advantage assumption restricts expressiveness: cannot represent multi-modal, asymmetric, or non-concave Q-functions in actions.
- The concavity in u_i is imposed by the architecture (P_{11,i} ≻ 0), not verified against the true game. If the true Q-function is not approximately quadratic in actions, the approximation will be poor.
- Best suited for games with smooth, approximately quadratic payoff interactions (e.g., trading, Cournot competition, resource allocation). Poorly suited for environments with discontinuous rewards (e.g., collision-based games like Simple Tag).

---

## 2. Strategically Robust Game Theory via Optimal Transport
**Authors:** Nicolas Lanzetti, Nicolas Fricker, Saverio Bolognani, Florian Dörfler, Dario Paccagnan (2025)
**Core contribution:** Defines Strategically Robust Equilibria (SRE) using optimal transport ambiguity sets, proves existence and computational complexity results, and provides algorithms for finite and continuous action games.

### 2.1 Motivation
Nash equilibria are fragile: they offer no protection against out-of-equilibrium play. Security strategies (maximin) are overly conservative. SRE interpolates between the two via a single parameter ε controlling the robustness level.

### 2.2 Key Definitions

**N-player game G(A, u):** Agents i ∈ {1,...,N}, action spaces A_i, payoff functions u_i: A_i × A_{-i} → ℝ. Mixed strategies: probability distributions p_i ∈ Δ_i over A_i.

**Wasserstein distance (1-Wasserstein):** For distributions σ₁, σ₂ over action space A with ground metric d:

    W_s(σ₁, σ₂) = inf_{γ ∈ Γ(σ₁,σ₂)} ∫ d(x,y)^s dγ(x,y)

**Wasserstein ball:** B^i_ε(p_{-i}) = {σ_{-i} ∈ P(A_{-i}) : W_s(σ_{p_{-i}}, σ_{-i}) ≤ ε}, where σ_{p_{-i}} is the product distribution of other agents' strategies.

**Strategically Robust Best Response:** Agent i's SR best response to p_{-i}:

    r^i_SR(p_{-i}) = argmax_{p_i ∈ Δ_i} min_{σ_{-i} ∈ B^i_ε(p_{-i})} U_i(p_i, σ_{-i})

**Strategically Robust Equilibrium (SRE):** A tuple (p̄_1,...,p̄_N) such that each p̄_i is a SR best response to p̄_{-i}. Formally:

    min_{σ_{-i} ∈ B_ε(p̄_{-i})} U_i(p̄_i, σ_{-i}) ≥ min_{σ_{-i} ∈ B_ε(p̄_{-i})} U_i(p_i, σ_{-i}),  ∀p_i ∈ Δ_i, ∀i

### 2.3 Interpolation Property
- ε = 0 → SRE collapses to Nash equilibrium (Wasserstein ball is a singleton).
- ε → ∞ → SRE becomes security strategies (Wasserstein ball covers all distributions).
- Intermediate ε → tunable robustness.

### 2.4 Existence Results
**Theorem 1 (Existence):** Under standard assumptions (compact action spaces, continuous payoffs), a strategically robust equilibrium exists for any ε ≥ 0 when ambiguity sets are based on optimal transport. This requires the same assumptions as existence of mixed Nash equilibria.

**Theorem 3 (Pure SRE in Concave Games):** For concave games (compact convex action spaces, payoffs concave in own action), a pure SRE exists. Pure strategies suffice for robustness against mixed deviations.

### 2.5 Computational Complexity
**Theorem 2:** The computational complexity of SRE in N-player finite games lies in PPAD (same class as Nash equilibria — no harder).

**Proposition 2 (2-player finite games):** SRE computation reduces to a Linear Complementarity Problem (LCP). For N-player games, it's a multilinear complementarity problem.

**Key insight for computation:** SRE can be computed by reformulating as a NE of a surrogate concave game.

### 2.6 Reformulation of SR Best Response (Proposition 1)
Using DRO duality, the max-min best response becomes a single optimisation:

    r^i_SR(p_{-i}) = argmax_{p_i ∈ Δ_i} max_{λ_i ≥ 0} { −λ_i ε^s + E_{a_{-i}~σ_{p_{-i}}} [ min_{â_{-i} ∈ A_{-i}} { E_{a_i~p_i}[u_i(a_i, â_{-i})] + λ_i d_{-i}(a_{-i}, â_{-i})^s } ] }

This introduces a dual variable λ_i (Lagrange multiplier for the Wasserstein constraint) and a fictitious adversary choosing worst-case deviations â_{-i} penalised by transport cost.

### 2.7 Computation for Finite Action Games (Section 3)

**Surrogate game formulation:** SRE of game G equals NE of surrogate game G̃ where each agent's augmented strategy is (p_i, λ_i) ∈ Δ_i × [0, M_i] with payoffs:

    ũ_i((p_i, λ_i), (p_{-i}, λ_{-i})) = −λ_i ε^s + E_{a_{-i}~σ_{p_{-i}}} [ min_{â_{-i}} { E_{a_i~p_i}[u_i(a_i, â_{-i})] + λ_i d_{-i}(a_{-i}, â_{-i})^s } ]

where M_i = 2 max|u_i| / ε^s is a proven upper bound on the dual variable.

**For 2-player games:** Reduces to LCP, solvable by PATH solver or Lemke's algorithm.

### 2.8 Computation for Continuous/Concave Games (Section 4)

**Proposition 3 (Equivalent Concave Game):** For concave game G, the pure SRE equals the NE of surrogate game G̃_ε with action spaces Ã_i = A_i × [0, M_i] and payoffs:

    ũ_i_ε((a_i, λ_i), (a_{-i}, λ_{-i})) = min_{â_{-i} ∈ A_{-i}} { u_i(a_i, â_{-i}) + λ_i d_{-i}(a_{-i}, â_{-i})^s } − λ_i ε^s

**Quadratic games (Section 4.3):** With payoffs u_i(a_i, a_{-i}) = a_i^T Q_i a_i + a_i^T B_i a_{-i} + q_i^T a_i and type-2 Wasserstein:

In the unconstrained case, the optimal dual variable is λ_{i,*} = ||(B_i)^T a_i|| / (2ε), yielding the regularised payoff:

    ũ_i = u_i(a_i, a_{-i}) − ε ||(B_i)^T a_i||

**Interpretation:** SRE = NE of a game with original payoffs plus an ε-scaled regularisation term penalising sensitivity to other agents' actions. This is the key bridge to Nash-DQN.

### 2.9 Experimental Results

**Pedestrian game (Section 1.2):** Vehicle (Maintain/Decelerate/Stop) vs Pedestrians (Wait/Cross). Shows NE (Maintain, Wait) is fragile, security strategy (Stop) is conservative, SRE (Decelerate, Wait) provides a middle ground. SRE maintains positive payoff under moderate probability of pedestrian deviation.

**Congestion game (Section 3.4):** Demonstrates "coordination via robustification" — SRE leads to less congestion and higher payoffs for all agents compared to NE.

**Cournot competition (Section 4.4):** N = 4 firms, T = 3 markets. Parameters:
| Market | α_t | β_t |
|--------|-----|-----|
| 1 | 100 | 0.8 |
| 2 | 120 | 0.6 |
| 3 | 110 | 0.7 |

| Firm | c_i | K_i |
|------|-----|-----|
| 1 | 40 | 100 |
| 2 | 45 | 120 |
| 3 | 50 | 90 |
| 4 | 55 | 80 |

Results show that strategic robustness induces firms to reduce production, protecting against over-production by competitors. All firms achieve higher nominal payoffs — a "coordination via robustification" effect where robustness against deviations paradoxically improves everyone's outcome.

For 2 firms, 1 market, no production bounds, the regularised payoff is:
    −β(a_i)² − βa_i a_{-i} + (α − (c_i + εβ))a_i
This is equivalent to the original Cournot game but with increased effective cost c_i + εβ, causing reduced production.

### 2.10 Conjecture
The paper conjectures that larger ε may facilitate computation (since ε → ∞ gives security strategies, computable in polynomial time via LP). This is left as an open question.

---

## 3. Strategically Robust Q-Learning (SRQ)
**Author:** Jack Brand (2025, Imperial College London, supervised by Prof. Dario Paccagnan)
**Core contribution:** A tabular MARL algorithm that extends Nash Q-learning by replacing the Nash operator with the SRE operator, providing tunable robustness in multi-agent environments.

### 3.1 Relationship to NashQ
SRQ modifies the NashQ algorithm (Hu & Wellman, 2003) by changing the equilibrium computation:

**NashQ operator:**
    NashQ_i(s') = π¹(s') ⋯ πⁿ(s') ⋅ Q_i(s')

**SRQ operator:**
    SRQ_i(s', ε) = π¹_SR(s', ε) ⋯ πⁿ_SR(s', ε) ⋅ Q_i(s')

where π^k_SR are the SRE mixed strategy probabilities.

### 3.2 Distance Metric: Total Variation Cost (TVC)
For discrete actions a_k, a_l ∈ {a_1, ..., a_n}:

    TVC(a_k, a_l) = 1 if a_k ≠ a_l, 0 otherwise

With TVC, the Wasserstein-1 ball simplifies to:

    B^{W₁}_ε(p) = { q : ½||p − q||₁ ≤ ε }

Consequence: ε ∈ [0, 1] covers the full range. ε = 0 → Nash, ε = 1 → security strategy (no need for ε → ∞).

### 3.3 Algorithm (Algorithm 3)
```
Initialise:
  ε ∈ [0, 1] (robustness parameter)
  ε_explore ∈ (0, 1] (exploration parameter)
  t = 0, get initial state s_0
  Agent i. For all s, a^j: Q^j_i(s, a_1,...,a_n) = 0 for j = 1,...,n.

Loop:
  Choose action a_i using ε_explore-greedy sampling from SRE on Q_i(s)
  Observe rewards r¹,...,rⁿ, actions a¹,...,aⁿ, next state s' = s_{t+1}
  Update Q^j_i for j = 1,...,n:
    Q^j_i(s, a) = (1 − α_t) Q^j_i(s, a) + α_t [r^j + γ SRQ^j_i(s', ε)]
  Linearly decay ε, ε_explore, α
  t := t + 1
```

### 3.4 Key Design Decisions
- **ε decay:** Linear decay over training. High robustness early (policies are uncertain, need hedging), low robustness late (policies stabilise, avoid over-conservatism).
- **Multiple SRE selection:** When multiple SRE exist for a bimatrix game, select the one with highest expected joint reward for all agents. Agents must share the same selection rule.
- **Each agent models all agents' Q-functions:** Agent i maintains Q^j_i(s, a₁,...,aₙ) for j = 1,...,n. Space: n|S|·|A|^n.
- **SRE solver:** Modified from Fricker's MATLAB code, using the PATH solver for the LCP formulation.

### 3.5 Complexity
- Same time complexity per iteration as NashQ (computing SRE is in PPAD, same class as NE).
- Space: n|S|·|A|^n (exponential in number of agents — the tabular curse).

### 3.6 Experimental Setup

**Environment 1:** 3×3 grid-world (9 states), 2 agents, 4 actions each (Up, Down, Left, Right).
- Agent 1 starts bottom-left, goal top-right.
- Agent 2 starts bottom-right, goal top-left.
- Simultaneous actions.
- Rewards: −1 for non-goal transition, +100 for reaching goal, −100 for collision (both agents move to same state; agents reset to previous positions).
- Episode terminates when both agents reach goals or 500 steps exceeded.
- 3,000 episodes per trial.

**Environment 2:** Variation of Environment 1 (different grid layout, described in the appendix).

**Metrics:**
- Average reward across all training episodes.
- Average reward and standard deviation over last 1,000 episodes (convergence quality).

**Agent matchups tested:** All combinations of SimpleQ, NashQ, and SRQ:
- (SimpleQ, SimpleQ)
- (SimpleQ, NashQ)
- (NashQ, NashQ)
- (SimpleQ, SRQ)
- (NashQ, SRQ)
- (SRQ, SRQ) with various ε₀ ∈ {0.25, 0.5, 0.75, 1.0}

**SimpleQ:** Standard single-agent Q-learning (Watkins). Each agent's Q-function only takes own state-action pairs — does not model other agents.

**NashQ:** Hu & Wellman algorithm. Each agent models all agents. At each step, constructs a 4×4 bimatrix game from Q-values, computes NE via Lemke-Howson, selects equilibrium with highest joint reward if multiple exist. ε-greedy action selection over equilibrium probabilities.

**SRQ:** Same as NashQ but computes SRE instead of NE over the 4×4 bimatrix game.

### 3.7 Key Results
- In environments with frequent negative interactions (collisions), SRQ achieves higher rewards and more stable convergence than NashQ and SimpleQ.
- SRQ agents tend to converge in fewer episodes.
- In environments with rare interactions, SRQ with high ε can be overly conservative, resulting in suboptimal rewards.
- The optimal ε depends on the environment: high-conflict → high ε, low-conflict → low ε.
- SRQ with ε = 0 reduces to NashQ (confirmed experimentally).

### 3.8 Future Directions Identified
- Extending to larger/more complex environments beyond simple grid-worlds.
- Deep learning integration to overcome the curse of dimensionality (→ this project).
- More adaptive exploration strategies.
- Computational efficiency improvements for larger action spaces.

---

## Summary: How the Three Papers Connect

```
NashQ (Hu & Wellman 2003)          Nash-DQN (Casgrain et al. 2022)
  Tabular, discrete actions    →     Deep, continuous actions
  NE via enumeration                 NE via LQ decomposition (analytic)
  Space: n|S||A|^n                   Neural net generalisation
         │                                    │
    Replace NE with SRE                  Replace NE with SRE
    (using SRE theory)                   (using regularisation result)
         │                                    │
         ▼                                    ▼
SRQ (Brand 2025)                     SRE-DQN (this project)
  Tabular, discrete actions    →     Deep, continuous actions
  SRE via LCP/PATH solver             SRE via regularised NE (analytic)
  Space: n|S||A|^n                    Neural net generalisation
                                              │
                               Also proposed: Deep SRQ
                                 Deep, discrete actions
                                 SRE via LCP/PATH on network-output bimatrix
```

**The key theoretical bridge** from the SRE theory paper: for concave/quadratic games, SRE = NE of a game with regularised payoffs (u_i − ε||(B_i)^T a_i||). This is what makes SRE-DQN tractable for continuous actions — Nash-DQN's LQ advantage defines a local quadratic game, and the SRE of that game can be computed analytically via the regularisation result.
