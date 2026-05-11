from dataclasses import dataclass


@dataclass
class DinesSreConfig:
    # Game shape
    num_actions: int = 5        # T: actions per player (bimatrix: same for both)

    # Model hyper-parameters (from DINES §5.1)
    embed_dim: int = 32         # D: player/action embedding size
    num_rounds: int = 30        # K: unrolled iteration steps
    num_heads: int = 4          # attention heads in self-attention layers
    ffn_hidden: int = 64        # hidden size in FFN layers
    eps_embed_dim: int = 16     # size of ε embedding before projection to D

    # Training
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_steps: int = 50_000
    eval_every: int = 500
    eval_games: int = 256
    checkpoint_every: int = 5_000

    # Frank-Wolfe best-response (for loss computation)
    fw_iters: int = 15          # inner FW iterations for BR oracle

    # Paths
    checkpoint_dir: str = "discrete_action_space/dines_sre/checkpoints"
    run_name: str = "default"

    # Device
    device: str = "cpu"         # override with "cuda" if available
