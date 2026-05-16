# SMILES-2026 Signal Interference Cancellation (solution)

## Reproducibility instructions

Create an environment with Python 3 (tested on Python 3.12.10.) and install the required packages:

```bash
pip install numpy scipy gdown
```

Then run:

```bash
python applicant_solution.py
```

This command:

- fixate seeds of pseudorandom number generators
- loads `challenge.mat` (from project work folder or download from internet with gdown)
- computes the provided baseline
- runs implementation from `your_canceller(...)`
- writes `results.json`

### Files
- `applicant_solution.py` contains solution of the problem, described in `README.md`
- `applicant_solution - all iterations.py` contains solution of the problem with all iterations of code in comments at the end of file
- `SOLUTION.md` - contains explanation of the solution
- `results.json` - contains the baseline and final solution metrics

## Final solution description

I use `fit_tx_prediction()` from the baseline to remove TX components first. After that I remove E spatial source as 1-rank matrix in SVD (like in scorer function) from RX residual after TX prediction without score filter, because E source may be outside a narrow frequency band (1.9 +- 0.6 where TX interference is concentrated).

Correction E as 1-rank component without scorer filter had the greatest impact on the scorer metric.

### Stage 1: TX nonlinearity cancellation with ordinary least squares (OLS)

First, I use `fit_tx_prediction()` method from the baseline to remove TX components.

For each received channel c in {0, 1, 2, 3}, the interference is modeled as a linear combination of nonlinear physical features (cross-products of TX channels) and their time lags. X is the $N \times M$ feature matrix (where M is the number of terms $\times$ lags) and y is the received signal.

The optimal weights w are found by minimizing the mean squared error (MSE) using OLS with L2-regularization:

$$ w = (X^* X + \lambda I)^{-1} X^H y $$

where $X^*$ is the conj. transpose of X, and $\lambda = 10^{-6}$ is the regularization parameter.

The predicted TX interference is then subtracted from the original signal to form the residual:

$$ R = rx - X w $$

It is important, that all X terms and y signal pass through a narrowband filter that cuts off the frequency band [1.9 - 0.6, 1.9 + 0.6] MHz.

### Stage 2: External source E correction with Rank-1 SVD

The residual matrix R of shape (N, 4) now mostly contains the background noise and the external spatially coherent source E. Since E comes from a single physical source, it appears across all 4 RX channels with different complex amplitudes (phases and delays), making it a rank-1 spatial component.

To extract it, I compute the $4 \times 4$ spatial covariance matrix of the residual:

$$ C = \frac{1}{N} R^H R $$

Applying an eigenvalue decomposition gives:

$$ C v_i = \lambda_i v_i $$

The principal eigenvector $v_{max}$ corresponding to the largest eigenvalue represents the spatial signature of the external source. I project the residual onto this vector to extract the shared 1D temporal signal $s = R v_{max}$, and then reconstruct the rank-1 source matrix $\hat{E}$ across all channels:

$$ \hat{E} = \frac{\langle s, R \rangle}{\| s \|^2} s $$

Finally, the cleaned signal is obtained by subtracting this external component:

$$ \hat{rx} = R - \hat{E} $$

## Experiments and failed attempts

0. Baseline with only `F_c( TX )` correction by basic OLS from `helpers["fit_tx_prediction"]`.

```text
=== Baseline ===
  ch0: 3.98 dB
  ch1: 4.86 dB
  ch2: 3.49 dB
  ch3: 3.74 dB
  Metric [baseline]: 4.02 dB
```

1. In baseline E correction is not presented, so, use `rank1_from_band_matrix()` function (from helpers) to find it and subtract.

```text
=== Your Solution ===
  ch0: 7.56 dB
  ch1: 6.72 dB
  ch2: 8.20 dB
  ch3: 5.56 dB
  Metric [yours]: 7.01 dB
```

Good. In next iterations it will be good to not use score filter before E correction, because E can be presented far from 1.9 MHz.

2. In baseline is only 10 model terms. Use more terms: all linear and cubic, so it will be 72 terms. Using L2-regularization as in baseline. Make dataset 2 times larger.

```text
=== Your Solution ===
  ch0: 6.93 dB
  ch1: 6.76 dB
  ch2: 8.07 dB
  ch3: 5.55 dB
  Metric [yours]: 6.83 dB
```

Discard. Metric decreased slightly. May be it is too much terms: we have 72 terms with 13 lags, so it is 936 parameters - too much for np.linalg.solve(). Or for scorer is not matter to the evaluator that we add new non-linear terms.

3. Change np.linalg.solve() with gradient boosting (HistGradientBoostingRegressor), that will find nonlinearities autonomously.

```text
=== Your Solution ===
  INVALID: explainability 0.883 < 0.95
  ch0: 0.00 dB
  ch1: 0.00 dB
  ch2: 0.00 dB
  ch3: 0.00 dB
  Metric [yours]: 0.00 dB
```

Discard. Scorer uses only 10 original terms. This solution doesn't fit into this model.

4. Main problem is to find `F_c( TX )` and `E` at the same time, to ensure that the dirty signal doesn't affect the prediction of one of the components.

```text
=== Your Solution ===
  INVALID: unexplained/residual 0.98 > 0.80
  ch0: 0.00 dB
  ch1: 0.00 dB
  ch2: 0.00 dB
  ch3: 0.00 dB
  Metric [yours]: 0.00 dB
```

This decision is wrong. Is seems, that i attemt to hack the scorer, not to really find the answer. So, change my mind.

5. Use neural network to find all nonliniarities.

Problem statement:

rx = s(n, c) + F_c(TX) + E(n, c) + η

F_c(TX) = NN(TX)
Loss = MSE(TX, RX)

E(n, c) = SVD(RX)

```text
=== Your Solution ===
 INVALID: explainability 0.832 < 0.95
  ch0: 0.00 dB
  ch1: 0.00 dB
  ch2: 0.00 dB
  ch3: 0.00 dB
  Metric [yours]: 0.00 dB
```

6. Make weights complex (dtype=torch.complex128)

```text
=== Your Solution ===
  ch0: 4.00 dB
  ch1: 2.14 dB
  ch2: 4.18 dB
  ch3: 1.51 dB
  Metric [yours]: 2.96 dB
```

7. Make neural network with 3 branches to train F_c(TX) and E at the same time. Make bias=False for convolutions for F_c(TX).

```text
=== Your Solution ===
  INVALID: explainability 0.258 < 0.95
  ch0: 0.00 dB
  ch1: 0.00 dB
  ch2: 0.00 dB
  ch3: 0.00 dB
  Metric [yours]: 0.00 dB
```

It is bad idea to use neural network to find features. Need to use fixed featured like in baseline.

8. Use branched idea with fixed features (10 terms)

Applying learned physical model to the full signal...
```text
=== Your Solution ===
  ch0: 2.93 dB
  ch1: 1.57 dB
  ch2: 5.65 dB
  ch3: 2.08 dB
  Metric [yours]: 3.06 dB
```

9. Refactor main idea with DeepUnfoldedCanceller net

```text
=== Your Solution ===
  INVALID: explainability 0.950 < 0.95
  ch0: 0.00 dB
  ch1: 0.00 dB
  ch2: 0.00 dB
  ch3: 0.00 dB
  Metric [yours]: 0.00 dB
```

Funny. I think, that approach with neural networks to find nonlinearities don't fit for this or I do smthg wrong. Return to idea 2 and improve it.

10. Do not use narrow band for E correction, because E influence can be on frequencies outside narrow band 1.9 +- 0.6 MHz (band from baseline with maximum of interference errors).

```text
=== Your Solution ===
  ch0: 10.77 dB
  ch1: 8.85 dB
  ch2: 11.42 dB
  ch3: 7.70 dB
  Metric [yours]: 9.69 dB
```

Good.