//! Breusch-Pagan test (Koenker robust variant) for heteroscedasticity.
//!
//! statsmodels' `het_breuschpagan(resid, exog)` regresses the squared residuals
//! on `exog` (which carries the constant) and reports
//!
//! ```text
//! lm = nobs · R²_aux        lm_pvalue = chi2.sf(lm, nvars − 1)
//! fvalue = aux F-statistic  f_pvalue  = f.sf(F, nvars − 1, nobs − nvars)
//! ```
//!
//! where `nvars = exog.shape[1]` and the χ² degrees of freedom count the
//! regressors excluding the constant.

use crate::igamc::chi2_sf;
use crate::incbet::f_sf;
use crate::ols::fit;
use crate::result::HetResult;
use rsomics_common::Result;

/// Run the Breusch-Pagan test. `resid` is the OLS residual vector; `x` is the
/// row-major `nobs × ncols` design including its constant column.
pub fn breuschpagan(resid: &[f64], x: &[f64], nobs: usize, ncols: usize) -> Result<HetResult> {
    let y: Vec<f64> = resid.iter().map(|&r| r * r).collect();
    let aux = fit(&y, x, nobs, ncols)?;

    let lm = nobs as f64 * aux.rsquared;
    let lm_pvalue = chi2_sf(lm, (ncols - 1) as f64);
    let fvalue = aux.fvalue();
    let f_pvalue = f_sf(aux.df_model, aux.df_resid, fvalue);

    Ok(HetResult {
        lm,
        lm_pvalue,
        fvalue,
        f_pvalue,
    })
}
