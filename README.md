# rsomics-het-tests

Breusch-Pagan (Koenker robust variant) and White Lagrange-multiplier tests for
heteroscedasticity on OLS residuals — a value-exact, faster reimplementation of
`statsmodels.stats.diagnostic.het_breuschpagan` and `het_white`.

Both tests regress the squared residuals on an auxiliary design and report the
Lagrange-multiplier statistic `lm = nobs · R²`, its χ² p-value, and the auxiliary
F-statistic with its F p-value — the same `(lm, lm_pvalue, fvalue, f_pvalue)`
tuple statsmodels returns.

- **Breusch-Pagan** regresses `resid²` on the supplied design (Koenker's
  studentized form, statsmodels' default `robust=True`). `lm_pvalue` uses
  `df = ncols − 1`.
- **White** augments the design with every square and cross-product of its
  columns (the upper-triangular pairwise products `x_i · x_j`, `i ≤ j`, in
  `numpy.triu_indices` order) and regresses `resid²` on that. `lm_pvalue` uses
  `df = rank − 1`.

## Install

```sh
cargo install rsomics-het-tests
```

## Usage

The exog design matrix must **include the constant column** (a column of 1s), as
statsmodels expects. Provide the data two ways:

```sh
# separate residual vector and design matrix
rsomics-het-tests --test breuschpagan --resid resid.txt --exog design.tsv

# one combined file: column 0 is the residual, the rest are that row's design
rsomics-het-tests --test white --data combined.tsv

# stdin (`-`) and machine-readable output
cat combined.tsv | rsomics-het-tests --test white --data - --json
```

Output is one line `lm<TAB>lm_pvalue<TAB>fvalue<TAB>f_pvalue`.

An exactly collinear auxiliary design (singular `XᵀX`) fails loud. statsmodels'
pseudoinverse instead silently reduces such a design to lower rank; this crate
does not replicate that — a rank-deficient design is reported as an error rather
than fit at reduced rank.

## Origin

This crate is an independent Rust reimplementation of statsmodels'
`het_breuschpagan` and `het_white` based on:

- The statsmodels source (`statsmodels/stats/diagnostic.py`, BSD-3-Clause),
  read and cited for the exact auxiliary-design construction (White's
  `numpy.triu_indices` pairwise products), the Koenker `lm = nobs · R²` form,
  and the degrees-of-freedom bookkeeping (`df_model = rank − 1`,
  `df_resid = nobs − rank`).
- The Cephes special-function routines `igamc` (chi-squared survival, via
  `scipy.stats.chi2.sf`) and `incbet` (the F-distribution survival via
  `scipy.special.fdtrc`), read and ported so the p-value tails match SciPy's
  special-function path to machine precision.

The published references for the methods: Breusch, T. S. & Pagan, A. R. (1979),
"A Simple Test for Heteroskedasticity and Random Coefficient Variation",
*Econometrica* 47(5):1287-1294; Koenker, R. (1981), "A note on studentizing a
test for heteroskedasticity", *Journal of Econometrics* 17(1):107-112; White, H.
(1980), "A Heteroskedasticity-Consistent Covariance Matrix Estimator and a
Direct Test for Heteroskedasticity", *Econometrica* 48(4):817-838.

License: MIT OR Apache-2.0.
Upstream credit: statsmodels <https://github.com/statsmodels/statsmodels>
(BSD-3-Clause); Cephes <https://www.netlib.org/cephes/> (via SciPy).
