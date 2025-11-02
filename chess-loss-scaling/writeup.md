# AI Forecasting Hackathon Report Template

**Author**: Emil
**Affiliation**: Independent Research
**With**: Apart Research

## Abstract

[To be completed after implementation - This study investigates whether general training loss in large language models correlates with domain-specific performance on chess move prediction. We evaluate 5 LLMs spanning 124M to 7B parameters against Leela Chess Zero's superhuman move probability distributions across 100 grandmaster-level games. Our findings will test the robustness of using training loss as a proxy for AI capabilities on tasks likely underrepresented in training data.]

**Keywords**: Language model evaluation, chess prediction, loss scaling, domain-specific performance, Leela Chess Zero

## 1. Introduction

**Research Question**: Does general training loss in LLMs correlate with domain-specific performance on chess move prediction?

**Hypothesis**: LLMs with lower general training losses will show lower cross-entropy loss when their chess move predictions are compared against Leela Chess Zero's probability distributions.

**Contribution**: This work tests the robustness of using training loss as a proxy for general AI capabilities by examining performance on a specific, measurable task (chess) that is likely underrepresented in training data.

**Background**: Based on Epoch AI's "Direct Approach" paper which posits that training loss correlates with AI's ability to imitate human performance across all intellectual tasks. However, training datasets may be unbalanced, and low overall loss might come from perfect replication of common content while performing poorly on rare, difficult content like grandmaster chess notation.

## 2. Methods

### 2.1 Models Evaluated
- GPT-2 (124M parameters) - baseline small model
- GPT-Neo 1.3B - mid-size open model
- Pythia 1.4B - specifically designed with documented training
- Pythia 2.8B - larger variant with known training trajectory
- Llama 2 7B - recent state-of-the-art open model

Model training losses sourced from original papers and documentation.

### 2.2 Chess Ground Truth
- Leela Chess Zero (lc0) as superhuman reference
- Model: Latest available network weights (T60+ series)
- Rationale: Open source, superhuman strength (3500+ Elo), provides move probability distributions via policy head

### 2.3 Dataset
- Source: Lichess Elite Database / FICS Games Database
- Number of games: 100
- Game quality: Grandmaster level (Elo > 2500)
- Time controls: Classical (to ensure high-quality moves)

### 2.4 Procedure
1. Load chess games in PGN format
2. For each position in each game:
   - Convert position to text format for LLM (full game history in algebraic notation)
   - Get LLM's probability distribution over next move (via logits over legal moves)
   - Convert position to Leela Zero input format
   - Get Leela's probability distribution over next move (via policy network output)
   - Calculate cross-entropy loss H(Leela || LLM) between distributions
3. Average loss per game
4. Average loss across all games for each model
5. Correlate with published training/evaluation losses

### 2.5 Technical Implementation
- Python with PyTorch, Transformers (HuggingFace), python-chess
- CUDA when available, CPU fallback for accessibility
- Leela Chess Zero via python-lczero or UCI interface
- Modern Python tooling: uv for package management, ruff for linting

## 3. Results

[After implementation - include graphs showing:
- Chess loss vs general training loss scatter plot with regression line
- Per-model performance breakdown (bar chart)
- Example predictions from different models showing probability distributions
- Statistical correlation coefficient (Pearson's r)]

### 3.1 Key Findings
[To be filled]

### 3.2 Model Performance Summary
[Table with columns: Model, Parameters, Training Loss, Chess Loss, Correlation]

## 4. Discussion and Conclusion

[After implementation - discuss:
- Whether significant correlation was found
- Implications for using training loss as general capability proxy
- Why chess may or may not generalize to other domains
- Comparison with human grandmaster performance if available
- Limitations and caveats
- Future work suggestions]

### 4.1 Limitations
- Dataset size limited by computational resources
- Chess notation may be tokenized differently across models
- Leela Zero's probability distribution may differ from human grandmaster intuition
- Sample size of models tested (5 models, limited by compute/time)
- Chess may not generalize to other specialized domains

### 4.2 Future Work
- Test on multiple game engines and average their distributions
- Weaken engines slightly to match human grandmaster level exactly
- Expand to other domains (mathematics, code generation, scientific reasoning)
- Use constrained generation to ensure valid move outputs
- Test correlation with model size, training compute, and data quantity separately
- Investigate whether chess-specific fine-tuning correlates with general capability

## 5. References

1. Epoch AI. (2024). Direct Approach to AI Forecasting. https://epoch.ai/files/direct-approach.pdf
2. Ruoss, A., et al. (2024). Grandmaster-Level Chess Without Search. arXiv:2402.04494
3. Leela Chess Zero Project. https://lczero.org/
4. Hoffmann, J., et al. (2022). Training Compute-Optimal Large Language Models. arXiv:2203.15556
5. Biderman, S., et al. (2023). Pythia: A Suite for Analyzing Large Language Models. arXiv:2304.01373
6. Black, S., et al. (2021). GPT-Neo: Large Scale Autoregressive Language Modeling with Mesh-Tensorflow.
7. Touvron, H., et al. (2023). Llama 2: Open Foundation and Fine-Tuned Chat Models. arXiv:2307.09288

## 6. Appendix: Security Considerations

**Potential Limitations**:
- Dataset may not represent full spectrum of chess positions (opening/middlegame/endgame balance)
- Leela Zero's probability distribution represents superhuman play, not human grandmaster intuition
- Sample size of models tested limited by computational resources
- Chess may not generalize to other domains due to its structured, rule-based nature
- Different tokenization schemes across models may affect move prediction

**Suggestions for Future Work**:
- Test on multiple game engines (Stockfish, Komodo) and average their distributions
- Calibrate engine strength to match 2700 Elo human grandmasters
- Expand to other domains with verifiable ground truth (mathematical proofs, code correctness)
- Use constrained generation or tool-use paradigm to ensure valid move outputs
- Separate correlation analysis for model size, training compute, and data quantity
- Investigate whether including chess games in training data improves correlation
