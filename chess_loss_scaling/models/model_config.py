"""Model configurations with training losses."""
from chess_loss_scaling.utils.logging_config import get_logger

logger = get_logger()

MODELS = [
    {
        "name": "gpt2",
        "hf_id": "gpt2",
        "params": "124M",
        "training_loss": None,  # Not publicly documented in detail
        "eval_loss": 3.31,  # Estimated from WebText validation
        "source": "https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf",
        "notes": "Original GPT-2 small model, baseline"
    },
    {
        "name": "gpt-neo-1.3B",
        "hf_id": "EleutherAI/gpt-neo-1.3B",
        "params": "1.3B",
        "training_loss": None,  # Training details less documented
        "eval_loss": 2.85,  # Estimated from Pile validation
        "source": "https://github.com/EleutherAI/gpt-neo",
        "notes": "EleutherAI's GPT-3 replication effort"
    },
    {
        "name": "pythia-1b",
        "hf_id": "EleutherAI/pythia-1b",
        "params": "1.0B",
        "training_loss": 2.68,  # From Pythia paper, final training loss
        "eval_loss": 2.74,  # From Pythia paper, validation loss
        "source": "https://arxiv.org/abs/2304.01373",
        "notes": "Pythia suite, fully documented training"
    },
    {
        "name": "pythia-1.4b",
        "hf_id": "EleutherAI/pythia-1.4b",
        "params": "1.4B",
        "training_loss": 2.58,  # From Pythia paper
        "eval_loss": 2.64,  # From Pythia paper
        "source": "https://arxiv.org/abs/2304.01373",
        "notes": "Pythia suite, larger variant"
    },
    {
        "name": "pythia-2.8b",
        "hf_id": "EleutherAI/pythia-2.8b",
        "params": "2.8B",
        "training_loss": 2.39,  # From Pythia paper
        "eval_loss": 2.47,  # From Pythia paper
        "source": "https://arxiv.org/abs/2304.01373",
        "notes": "Pythia suite, largest practical for testing"
    },
    # OLMo 2 models - final checkpoints
    {
        "name": "olmo-2-1b",
        "hf_id": "allenai/OLMo-2-0425-1B",
        "params": "1.0B",
        "training_loss": None,  # Available in WandB: wandb.ai/ai2-llm/OLMo2-1B
        "eval_loss": None,  # See WandB or paper Table 9
        "source": "https://arxiv.org/abs/2501.00656",
        "notes": "OLMo 2 1B, trained on 4T tokens (stage1) + 50B tokens (stage2 mid-training)"
    },
    {
        "name": "olmo-2-1b-mid",
        "hf_id": "allenai/OLMo-2-0425-1B",
        "hf_revision": "stage1-step950000-tokens1993B",
        "params": "1.0B",
        "training_loss": None,  # Available in WandB: wandb.ai/ai2-llm/OLMo2-1B
        "eval_loss": None,  # See WandB
        "source": "https://arxiv.org/abs/2501.00656",
        "notes": "OLMo 2 1B at ~50% training (2.0T/4.0T tokens), stage1 only"
    },
    {
        "name": "olmo-2-7b",
        "hf_id": "allenai/OLMo-2-1124-7B",
        "params": "7.0B",
        "training_loss": None,  # Available in WandB: wandb.ai/ai2-llm/OLMo2-7B
        "eval_loss": None,  # See WandB or paper Table 9
        "source": "https://arxiv.org/abs/2501.00656",
        "notes": "OLMo 2 7B, trained on 4T tokens (stage1) + 50B tokens (stage2, 3 runs merged)"
    },
    {
        "name": "olmo-2-7b-mid",
        "hf_id": "allenai/OLMo-2-1124-7B",
        "hf_revision": "stage1-step477000-tokens2001B",
        "params": "7.0B",
        "training_loss": None,  # Available in WandB: wandb.ai/ai2-llm/OLMo2-7B
        "eval_loss": None,  # See WandB
        "source": "https://arxiv.org/abs/2501.00656",
        "notes": "OLMo 2 7B at ~50% training (2.0T/4.0T tokens), stage1 only"
    },
    {
        "name": "olmo-2-13b",
        "hf_id": "allenai/OLMo-2-1124-13B",
        "params": "13.0B",
        "training_loss": None,  # Available in WandB: wandb.ai/ai2-llm/OLMo2-13B
        "eval_loss": None,  # See WandB or paper Table 9
        "source": "https://arxiv.org/abs/2501.00656",
        "notes": "OLMo 2 13B, trained on 5T tokens (stage1) + 100B+300B tokens (stage2, 4 runs merged)"
    },
    {
        "name": "olmo-2-13b-mid",
        "hf_id": "allenai/OLMo-2-1124-13B",
        "hf_revision": "stage1-step298000-tokens2500B",
        "params": "13.0B",
        "training_loss": None,  # Available in WandB: wandb.ai/ai2-llm/OLMo2-13B
        "eval_loss": None,  # See WandB
        "source": "https://arxiv.org/abs/2501.00656",
        "notes": "OLMo 2 13B at ~50% training (2.5T/5.0T tokens), stage1 only"
    },
    {
        "name": "olmo-2-32b",
        "hf_id": "allenai/OLMo-2-0325-32B",
        "params": "32.0B",
        "training_loss": None,  # Available in WandB (see paper or model card)
        "eval_loss": None,  # See WandB or paper Table 9
        "source": "https://arxiv.org/abs/2501.00656",
        "notes": "OLMo 2 32B, trained on 6T tokens (stage1) + 100B+300B tokens (stage2, 4 runs merged)"
    },
    {
        "name": "olmo-2-32b-mid",
        "hf_id": "allenai/OLMo-2-0325-32B",
        "hf_revision": "stage1-step358000-tokens3004B",
        "params": "32.0B",
        "training_loss": None,  # Available in WandB
        "eval_loss": None,  # See WandB
        "source": "https://arxiv.org/abs/2501.00656",
        "notes": "OLMo 2 32B at ~50% training (3.0T/6.0T tokens), stage1 only"
    },
]


def get_model_config(name: str) -> dict:
    """Get configuration for a specific model by name."""
    for model in MODELS:
        if model["name"] == name:
            return model
    raise ValueError(f"Model {name} not found in configuration")


def list_model_names() -> list:
    """Get list of all configured model names."""
    return [m["name"] for m in MODELS]


def get_reference_loss(model_config: dict) -> float | None:
    """Get the reference training/eval loss for correlation analysis."""
    # Prefer eval_loss if available (more stable), fall back to training_loss
    if model_config["eval_loss"] is not None:
        return model_config["eval_loss"]
    elif model_config["training_loss"] is not None:
        return model_config["training_loss"]
    else:
        logger.warning(f"No reference loss available for {model_config['name']}")
        return None
