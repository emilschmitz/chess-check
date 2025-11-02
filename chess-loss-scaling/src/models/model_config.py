"""Model configurations with training losses."""

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


def get_reference_loss(model_config: dict) -> float:
    """Get the reference training/eval loss for correlation analysis."""
    # Prefer eval_loss if available (more stable), fall back to training_loss
    if model_config["eval_loss"] is not None:
        return model_config["eval_loss"]
    elif model_config["training_loss"] is not None:
        return model_config["training_loss"]
    else:
        raise ValueError(f"No reference loss available for {model_config['name']}")
