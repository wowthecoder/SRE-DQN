# AGENTS.md 

## Research papers

Read `relevant_papers/core` folder contains summaries of the 3 research papers that lays the foundation to this project. 
1. The `Strategically Robust Game Theory via Optimal Transport.md` paper is the main theoretical paper with all the important equations. The `strategically-robust-game-theory/` directory is the source code implementation of the theory paper.
2. The `Nash DQN.md` paper is relevant for the `continuous_action_space` folder only. The `Nash-DQN/` directory is the source code implementation of the paper. 
3. The `Strategically Robust Q-Learning.md` paper is relevant for the `discrete_action_space` folder only. The `sre-sandbox/` directory is the source code implementation of the paper. 

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

---

## Folder structure

- `continuous_action_space` folder focuses on 2 cases:
    - `locally_linear_quadratic` subfolder is to extend the idea from the Nash DQN paper to build SRE DQN algorithm based on the same locally linear quadratic assumption. Evaluates on the same environment as the Nash DQN paper. 
    - `just_concave` tries to relax the assumption by using fictitious play and the surrogate game payoff techniques from the theory paper. Evaluates on Multi Particle environment. 

- `discrete_action_space` folder is to extend the Strategically Robust Q-Learning paper which is a tabular Q-learning algorithm to a deep Q-learning version. It wraps around a stage game equilibrium solver. 2 main cases: bimatrix game (2 player) and N player general sum games. Several approaches currently:
    - DuelingDoubleDQN with `PathLCPBimatrixSolver` as the SRE stage game solver. A pooled version where multiple PATH workers run as parallel processes is also implemented. `PathMcpNPlayerSreSolver` is the N player version. 
    - `sre_solvers` folder contains all the different solvers for 2 players and N players. 

- `relevant_papers` folder contains the research papers with interesting ideas that can be tried to improve the existing algorithms. It contains a README detailing which papers are implemented and which are not. 

Do 
```bash
source venv/bin/activate
``` 
before trying to run any files, notebooks or tests. 

Always read the `Strategically Robust Game Theory via Optimal Transport.md` paper, along with whichever other relevant papers in the `relevant_papers` folder first before answering any prompts regarding implementation of research ideas or justification of algorithm details. Only skip if it's simple refactoring or small changes/features that does not affect the algorithm. 

