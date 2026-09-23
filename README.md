# Market Regime Detection | Portfolio Risk Optimisation
data for latent market-regime detection and a regime-adaptive multi-asset strategy.

## Pipeline
- Use 6-asset market with persistent latent regimes
- Engineer rolling return/volatility features
- Select a Gaussian Mixture Model using BIC
- Infer regimes out-of-sample with a walk-forward split
- Map regimes to risk characteristics using training data only
- Compare an adaptive allocation with an equal-weight baseline

## Run
```bash
pip install -r requirements.txt
python run.py
```
