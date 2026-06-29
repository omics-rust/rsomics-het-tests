//! The four-field result both tests return, matching the statsmodels tuple
//! `(lm, lm_pvalue, fvalue, f_pvalue)`.

use serde::Serialize;

/// `(lm, lm_pvalue, fvalue, f_pvalue)` of a het test.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct HetResult {
    /// Lagrange-multiplier statistic, `nobs · R²_aux`.
    pub lm: f64,
    /// χ² p-value of the LM statistic.
    pub lm_pvalue: f64,
    /// F-statistic of the auxiliary regression.
    pub fvalue: f64,
    /// F-distribution p-value of `fvalue`.
    pub f_pvalue: f64,
}
