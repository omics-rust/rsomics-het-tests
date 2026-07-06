//! The auxiliary OLS regression both het tests run on the squared residuals.
//!
//! statsmodels fits `OLS(y, X).fit()` with the Moore-Penrose pseudoinverse and
//! reports `rsquared = 1 − ssr/centered_tss` (a constant column is present, so
//! the centered total sum of squares is used), `df_model = rank − 1`, and
//! `df_resid = nobs − rank`. R² is rank-invariant, so we take it from the normal
//! equations `(XᵀX) β = Xᵀy`; for the well-conditioned designs here the normal
//! solve agrees with the SVD pseudoinverse to a few ULP.
//!
//! The degrees of freedom, though, hinge on the *numerical* rank, which the
//! normal-equations LU cannot see: forming `XᵀX` squares the condition number
//! and buries a near-collinear direction's tiny singular value in roundoff, and
//! partial pivoting only trips on an *exactly* zero pivot. White's auxiliary
//! design is routinely rank-deficient (a constant column makes `const·xj = xj`
//! duplicate the originals, and a collinear input column like `0.3·x` never
//! produces an exact-zero pivot), so we compute the rank from the singular
//! values of `X` directly via one-sided Jacobi — which keeps tiny singular
//! values to high relative accuracy — and count them above statsmodels'
//! `numpy.linalg.matrix_rank` tolerance `σ_max · ncols · eps`.
//!
//! An *exactly* singular `XᵀX` (integer/dummy collinearity) still fails loud in
//! the LU solve rather than being fit at reduced rank via a pseudoinverse.

use rsomics_common::{Result, RsomicsError};

/// Quantities of the auxiliary regression that drive the LM and F statistics.
pub struct AuxFit {
    /// R² of the regression, `1 − ssr/centered_tss`.
    pub rsquared: f64,
    /// Explained sum of squares, `centered_tss − ssr`.
    pub ess: f64,
    /// Residual sum of squares.
    pub ssr: f64,
    /// Model degrees of freedom, `rank − 1`.
    pub df_model: f64,
    /// Residual degrees of freedom, `nobs − rank`.
    pub df_resid: f64,
}

impl AuxFit {
    /// F-statistic of the fully specified model, `mse_model / mse_resid`
    /// (statsmodels' nonrobust path).
    #[must_use]
    pub fn fvalue(&self) -> f64 {
        (self.ess / self.df_model) / (self.ssr / self.df_resid)
    }
}

/// Pairwise (cascade) summation, matching `numpy.sum`'s accumulation so the
/// means and sums of squares round identically to statsmodels.
fn pairwise_sum(xs: &[f64]) -> f64 {
    const BLOCK: usize = 128;
    let n = xs.len();
    if n <= BLOCK {
        let mut s = 0.0;
        for &x in xs {
            s += x;
        }
        return s;
    }
    let half = (n / 2).div_ceil(BLOCK) * BLOCK;
    pairwise_sum(&xs[..half]) + pairwise_sum(&xs[half..])
}

/// Fit `y ~ X` (X already includes a constant column) and return the quantities
/// the het statistics need. `x` is row-major `nobs × ncols`.
pub fn fit(y: &[f64], x: &[f64], nobs: usize, ncols: usize) -> Result<AuxFit> {
    let beta = solve_normal_equations(y, x, nobs, ncols)?;

    let resid: Vec<f64> = (0..nobs)
        .map(|i| {
            let row = &x[i * ncols..(i + 1) * ncols];
            let yhat: f64 = row.iter().zip(&beta).map(|(&xj, &bj)| xj * bj).sum();
            y[i] - yhat
        })
        .collect();
    let sq: Vec<f64> = resid.iter().map(|&r| r * r).collect();
    let ssr = pairwise_sum(&sq);

    let ybar = pairwise_sum(y) / nobs as f64;
    let dev: Vec<f64> = y.iter().map(|&yi| (yi - ybar) * (yi - ybar)).collect();
    let centered_tss = pairwise_sum(&dev);

    let rank = numerical_rank(x, nobs, ncols);
    let df_model = (rank - 1) as f64;
    let df_resid = (nobs - rank) as f64;

    Ok(AuxFit {
        rsquared: 1.0 - ssr / centered_tss,
        ess: centered_tss - ssr,
        ssr,
        df_model,
        df_resid,
    })
}

/// Solve `(XᵀX) β = Xᵀy` for β by forming the normal equations and running LU
/// with partial pivoting (LAPACK `dgesv`). Fails loud on a singular `XᵀX`.
fn solve_normal_equations(y: &[f64], x: &[f64], nobs: usize, ncols: usize) -> Result<Vec<f64>> {
    let mut xtx = vec![0.0; ncols * ncols];
    let mut xty = vec![0.0; ncols];
    for i in 0..nobs {
        let row = &x[i * ncols..(i + 1) * ncols];
        let yi = y[i];
        for (a, &xa) in row.iter().enumerate() {
            xty[a] += xa * yi;
            let dst = &mut xtx[a * ncols..a * ncols + ncols];
            for (b, &xb) in row.iter().enumerate() {
                dst[b] += xa * xb;
            }
        }
    }
    lu_solve(&mut xtx, &xty, ncols).ok_or_else(|| {
        RsomicsError::InvalidInput(
            "auxiliary design XᵀX is singular: the regressors are exactly collinear".into(),
        )
    })
}

/// Numerical rank of the row-major `nobs × ncols` design, matching
/// `numpy.linalg.matrix_rank` on statsmodels' cached singular values: count the
/// singular values of `X` above `σ_max · ncols · f64::EPSILON`.
fn numerical_rank(x: &[f64], nobs: usize, ncols: usize) -> usize {
    let sv = jacobi_singular_values(x, nobs, ncols);
    let smax = sv.iter().copied().fold(0.0_f64, f64::max);
    let tol = smax * ncols as f64 * f64::EPSILON;
    sv.iter().filter(|&&s| s > tol).count()
}

/// Singular values of a row-major `nrows × ncols` matrix (`nrows ≥ ncols`) by
/// one-sided Jacobi rotation. Orthogonalising the columns in place drives their
/// norms to the singular values while preserving tiny values to high relative
/// accuracy — the property the near-collinear rank determination depends on.
fn jacobi_singular_values(a: &[f64], nrows: usize, ncols: usize) -> Vec<f64> {
    let mut m = a.to_vec();
    for _ in 0..100 {
        let mut rotated = false;
        for p in 0..ncols {
            for q in (p + 1)..ncols {
                let mut alpha = 0.0;
                let mut beta = 0.0;
                let mut gamma = 0.0;
                for i in 0..nrows {
                    let xp = m[i * ncols + p];
                    let xq = m[i * ncols + q];
                    alpha += xp * xp;
                    beta += xq * xq;
                    gamma += xp * xq;
                }
                if gamma.abs() <= f64::EPSILON * (alpha * beta).sqrt() {
                    continue;
                }
                rotated = true;
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = if zeta == 0.0 {
                    1.0
                } else {
                    zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                for i in 0..nrows {
                    let xp = m[i * ncols + p];
                    let xq = m[i * ncols + q];
                    m[i * ncols + p] = c * xp - s * xq;
                    m[i * ncols + q] = s * xp + c * xq;
                }
            }
        }
        if !rotated {
            break;
        }
    }
    (0..ncols)
        .map(|j| {
            let s: f64 = (0..nrows).map(|i| m[i * ncols + j].powi(2)).sum();
            s.sqrt()
        })
        .collect()
}

/// Solve `A x = b` for an `m×m` system by LU decomposition with partial pivoting.
/// `a` is consumed as the working LU store. `None` on an exactly zero pivot.
fn lu_solve(a: &mut [f64], b: &[f64], m: usize) -> Option<Vec<f64>> {
    let mut x = b.to_vec();
    for col in 0..m {
        let mut p = col;
        let mut best = a[col * m + col].abs();
        for r in (col + 1)..m {
            let v = a[r * m + col].abs();
            if v > best {
                best = v;
                p = r;
            }
        }
        if a[p * m + col] == 0.0 {
            return None;
        }
        if p != col {
            for c in 0..m {
                a.swap(col * m + c, p * m + c);
            }
            x.swap(col, p);
        }
        let pivot = a[col * m + col];
        for r in (col + 1)..m {
            let factor = a[r * m + col] / pivot;
            a[r * m + col] = factor;
            for c in (col + 1)..m {
                a[r * m + c] -= factor * a[col * m + c];
            }
            x[r] -= factor * x[col];
        }
    }
    for i in (0..m).rev() {
        let mut s = x[i];
        for c in (i + 1)..m {
            s -= a[i * m + c] * x[c];
        }
        x[i] = s / a[i * m + i];
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rel(a: f64, b: f64) -> f64 {
        (a - b).abs() / b.abs().max(f64::MIN_POSITIVE)
    }

    #[test]
    fn fit_recovers_known_line() {
        // y = 2 + 3·x exactly; aux fit on (const, x) → ssr 0, rsq 1.
        let x = [1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0];
        let y = [2.0, 5.0, 8.0, 11.0];
        let f = fit(&y, &x, 4, 2).unwrap();
        assert!(f.ssr < 1e-20, "ssr {}", f.ssr);
        assert!(rel(f.rsquared, 1.0) < 1e-12);
        assert_eq!(f.df_model, 1.0);
        assert_eq!(f.df_resid, 2.0);
    }

    #[test]
    fn singular_design_fails_loud() {
        // two identical non-constant columns → XᵀX singular.
        let x = [1.0, 2.0, 2.0, 1.0, 4.0, 4.0];
        let y = [1.0, 2.0];
        assert!(fit(&y, &x, 2, 3).is_err());
    }

    #[test]
    fn rank_counts_near_collinear_columns() {
        // const + x + 0.3·x: the third column is collinear with the second but
        // 0.3·x is not float-exact, so an XᵀX pivot never hits exact zero.
        let mut x = Vec::new();
        for xi in [1.0, 2.0, 3.5, -0.7, 4.2, -2.1] {
            x.extend_from_slice(&[1.0, xi, 0.3 * xi]);
        }
        assert_eq!(numerical_rank(&x, 6, 3), 2);

        // const + x + x² is genuinely full rank.
        let mut xf = Vec::new();
        for xi in [1.0, 2.0, 3.5, -0.7, 4.2, -2.1] {
            xf.extend_from_slice(&[1.0, xi, xi * xi]);
        }
        assert_eq!(numerical_rank(&xf, 6, 3), 3);
    }
}
