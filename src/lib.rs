//! Breusch-Pagan (Koenker robust variant) and White Lagrange-multiplier tests
//! for heteroscedasticity on OLS residuals — value-exact, faster ports of
//! `statsmodels.stats.diagnostic.het_breuschpagan` and `het_white`.
//!
//! Both regress the squared residuals on an auxiliary design and report the LM
//! statistic `nobs · R²`, its χ² p-value, and the auxiliary F-statistic with its
//! F p-value. Breusch-Pagan uses the supplied exog directly (Koenker's robust
//! form, statsmodels' default); White augments it with every square and
//! cross-product of the columns. The exog matrix is expected to include the
//! constant column, as statsmodels expects.

mod breuschpagan;
mod igamc;
mod incbet;
mod io;
mod ols;
mod result;
mod white;

pub use breuschpagan::breuschpagan;
pub use io::{Dataset, read_combined, read_separate};
pub use result::HetResult;
pub use white::white;
