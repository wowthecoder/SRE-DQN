# AGENTS.md 

Read `relevant_papers/core/core_papers.md` for the 3 research papers that lays the foundation to this project. The `sre-sandbox/` and `strategically-robust-game-theory/` directories are source code implementation of the papers.

## Folder structure

- `continuous_action_space` folder is to extend the idea from the Nash DQN paper to build SRE DQN algorithm for continuous action space, concave games. One algorithm focuses on the locally linear quadratic assumption and others focus on fixed point iterative approximation methods. Evaluates on the same environment as the Nash DQN paper. 

- `discrete_action_space` folder is to extend the Strategically Robust Q-Learning paper which is a tabular Q-learning algorithm to a deep Q-learning version. It wraps around a stage game equilibrium solver. There is different folders for different environments to be evaluated on. 

- `relevant_papers` folder contains the research papers with interesting ideas that can be tried to improve the existing algorithms. 

Do 
```bash
source venv/bin/activate
``` 
before trying to run any files, notebooks or tests. 

Always read the `core_papers.md` first before answering any prompts regarding implementation of research ideas or justification of algorithm details. Only skip if it's simple refactoring or small changes/features that does not affect the algorithm. 

