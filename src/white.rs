//! White's test for heteroscedasticity.
//!
//! statsmodels' `het_white(resid, exog)` builds the auxiliary design from the
//! upper-triangular pairwise products of the original columns and regresses the
//! squared residuals on it:
//!
//! ```text
//! i0, i1 = triu_indices(nvars0)      aux = exog[:, i0] · exog[:, i1]
//! lm = nobs · R²_aux                 lm_pvalue = chi2.sf(lm, df_model)
//! fvalue = aux F-statistic           f_pvalue  = f.sf(F, df_model, df_resid)
//! ```
//!
//! with `df_model = rank(aux) − 1`. Because the original design carries a
//! constant column, the products enumerate every square and cross-product plus
//! the original columns (`const·const = const`, `const·xj = xj`), in NumPy's
//! row-major upper-triangle order: for columns `0..p`, the pairs are
//! `(0,0), (0,1), … (0,p−1), (1,1), (1,2), …`.

use crate::igamc::chi2_sf;
use crate::incbet::f_sf;
use crate::ols::fit;
use crate::result::HetResult;
use rsomics_common::Result;

/// Auxiliary design column count for a `ncols`-wide original design:
/// `ncols·(ncols+1)/2` upper-triangle products.
#[must_use]
pub fn aux_ncols(ncols: usize) -> usize {
    ncols * (ncols + 1) / 2
}

/// Build the White auxiliary design (row-major `nobs × aux_ncols`) from the
/// original `nobs × ncols` design, in `numpy.triu_indices` column order.
fn white_design(x: &[f64], nobs: usize, ncols: usize) -> Vec<f64> {
    let acols = aux_ncols(ncols);
    let mut out = vec![0.0; nobs * acols];
    for i in 0..nobs {
        let row = &x[i * ncols..(i + 1) * ncols];
        let dst = &mut out[i * acols..(i + 1) * acols];
        let mut c = 0;
        for a in 0..ncols {
            for b in a..ncols {
                dst[c] = row[a] * row[b];
                c += 1;
            }
        }
    }
    out
}

/// Run White's test. `resid` is the OLS residual vector; `x` is the row-major
/// `nobs × ncols` design including its constant column.
pub fn white(resid: &[f64], x: &[f64], nobs: usize, ncols: usize) -> Result<HetResult> {
    let y: Vec<f64> = resid.iter().map(|&r| r * r).collect();
    let acols = aux_ncols(ncols);
    let aux_x = white_design(x, nobs, ncols);
    let aux = fit(&y, &aux_x, nobs, acols)?;

    let lm = nobs as f64 * aux.rsquared;
    let lm_pvalue = chi2_sf(lm, aux.df_model);
    let fvalue = aux.fvalue();
    let f_pvalue = f_sf(aux.df_model, aux.df_resid, fvalue);

    Ok(HetResult {
        lm,
        lm_pvalue,
        fvalue,
        f_pvalue,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn design_order_is_triu_rowmajor() {
        // one row, ncols=3 columns [2, 3, 5]: triu pairs (0,0)(0,1)(0,2)(1,1)(1,2)(2,2).
        let x = [2.0, 3.0, 5.0];
        let d = white_design(&x, 1, 3);
        assert_eq!(d, vec![4.0, 6.0, 10.0, 9.0, 15.0, 25.0]);
        assert_eq!(aux_ncols(3), 6);
    }
}
