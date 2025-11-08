"""LLM integration for chess move prediction."""
import gc

import chess
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from chess_loss_scaling.utils.logging_config import get_logger

logger = get_logger()


class ChessLLM:
    """Load HuggingFace LLM and get chess move predictions."""

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        load_in_8bit: bool = False,
    ):
        """
        Load model and tokenizer.

        Args:
            model_id: HuggingFace model identifier
            device: Device to use ("cuda", "cpu", or "auto")
            load_in_8bit: Use 8-bit quantization to save memory
        """
        self.model_id = model_id
        self.device = self._setup_device(device)
        self.load_in_8bit = load_in_8bit
        self.logger = get_logger()

        self.logger.info(f"Loading model {model_id} on {self.device}...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
        )

        # Some models don't have pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
        }

        if load_in_8bit and self.device == "cuda":
            load_kwargs["load_in_8bit"] = True
            load_kwargs["device_map"] = "auto"
        elif self.device == "cuda":
            load_kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            **load_kwargs
        )

        # Move to device if not using device_map
        if "device_map" not in load_kwargs:
            self.model = self.model.to(self.device)

        self.model.eval()
        self.logger.info("Model loaded successfully")

    def _setup_device(self, device: str) -> str:
        """Setup device for model."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def get_move_probabilities(
        self,
        game_history: str,
        legal_moves: list[str],
        board: chess.Board,
    ) -> dict[str, float]:
        """
        Get probability distribution over legal moves.

        Args:
            game_history: Game history prompt
            legal_moves: List of legal moves in SAN
            board: Current board position

        Returns:
            Dict mapping SAN moves to probabilities
        """
        if not legal_moves:
            return {}

        # Tokenize the prompt
        inputs = self.tokenizer(
            game_history,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]  # Last token logits

        # Get probabilities for each legal move
        move_probs = {}

        for move in legal_moves:
            # Tokenize the move (may be multiple tokens)
            move_tokens = self.tokenizer.encode(
                " " + move,  # Space before move is important
                add_special_tokens=False
            )

            if not move_tokens:
                continue

            # For simplicity, use probability of first token
            # More sophisticated: multiply probabilities of all tokens
            token_id = move_tokens[0]
            move_probs[move] = float(torch.softmax(logits, dim=0)[token_id].cpu())

        # Normalize to sum to 1.0
        total_prob = sum(move_probs.values())
        if total_prob > 0:
            move_probs = {move: prob / total_prob for move, prob in move_probs.items()}
        else:
            # Fallback to uniform distribution
            uniform_prob = 1.0 / len(legal_moves)
            move_probs = {move: uniform_prob for move in legal_moves}

        # Ensure all legal moves are present (even with tiny probability)
        for move in legal_moves:
            if move not in move_probs:
                move_probs[move] = 1e-10

        # Renormalize
        total_prob = sum(move_probs.values())
        move_probs = {move: prob / total_prob for move, prob in move_probs.items()}

        return move_probs

    def unload(self):
        """Unload model from memory."""
        self.logger.info(f"Unloading model {self.model_id}...")
        del self.model
        del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_chess_llm(
    model_id: str,
    device: str = "auto",
    load_in_8bit: bool = False,
) -> ChessLLM:
    """
    Factory function to load a chess LLM.

    Args:
        model_id: HuggingFace model identifier
        device: Device to use
        load_in_8bit: Use 8-bit quantization

    Returns:
        ChessLLM instance
    """
    return ChessLLM(
        model_id=model_id,
        device=device,
        load_in_8bit=load_in_8bit,
    )
