use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, ValueEnum};
use rsomics_common::{CommonFlags, Result, RsomicsError, ToolMeta, run};

use rsomics_het_tests::{Dataset, HetResult, breuschpagan, read_combined, read_separate, white};

pub const META: ToolMeta = ToolMeta {
    name: env!("CARGO_PKG_NAME"),
    version: env!("CARGO_PKG_VERSION"),
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum TestKind {
    /// Breusch-Pagan test, Koenker robust variant (statsmodels default).
    Breuschpagan,
    /// White's test (exog augmented with squares and cross-products).
    White,
}

/// Breusch-Pagan and White heteroscedasticity tests on OLS residuals —
/// value-exact `statsmodels` `het_breuschpagan` / `het_white`.
///
/// The auxiliary regression runs on the squared residuals. Supply the design
/// matrix `exog` *including its constant column* (a column of 1s), exactly as
/// statsmodels expects. Provide the data either as separate `--resid` and
/// `--exog` files, or as one combined `--data` file whose first column is the
/// residual and remaining columns are that row's design.
///
/// Output is one line `lm<TAB>lm_pvalue<TAB>fvalue<TAB>f_pvalue`.
#[derive(Parser, Debug)]
#[command(name = "rsomics-het-tests", version, about, long_about = None)]
pub struct Cli {
    /// Which heteroscedasticity test to run.
    #[arg(long = "test", value_enum)]
    pub test: TestKind,

    /// Residual vector, one value per line (`-` reads stdin). Use with `--exog`.
    #[arg(long, value_name = "FILE", requires = "exog", conflicts_with = "data")]
    pub resid: Option<PathBuf>,

    /// Design matrix `nobs × ncols`, constant column included. Use with `--resid`.
    #[arg(long, value_name = "FILE", conflicts_with = "data")]
    pub exog: Option<PathBuf>,

    /// Combined `resid x0 x1 …` per row (`-` reads stdin); column 0 is resid.
    #[arg(long, value_name = "FILE")]
    pub data: Option<PathBuf>,

    #[command(flatten)]
    pub common: CommonFlags,
}

impl Cli {
    fn load(&self) -> Result<Dataset> {
        match (&self.resid, &self.exog, &self.data) {
            (Some(r), Some(x), None) => read_separate(r, x),
            (None, None, Some(d)) => read_combined(d),
            _ => Err(RsomicsError::InvalidInput(
                "provide either --resid with --exog, or --data".into(),
            )),
        }
    }

    pub fn run(self) -> ExitCode {
        let common = self.common.clone();
        run(&common, META, || {
            let d = self.load()?;
            let res: HetResult = match self.test {
                TestKind::Breuschpagan => breuschpagan(&d.resid, &d.exog, d.nobs, d.ncols)?,
                TestKind::White => white(&d.resid, &d.exog, d.nobs, d.ncols)?,
            };
            if !common.json {
                println!(
                    "{}\t{}\t{}\t{}",
                    fmt(res.lm),
                    fmt(res.lm_pvalue),
                    fmt(res.fvalue),
                    fmt(res.f_pvalue)
                );
            }
            Ok(res)
        })
    }
}

/// Shortest round-trip decimal, switching to scientific notation for the tiny
/// magnitudes deep-tail p-values reach so a 1e-200 value is not 200 characters.
fn fmt(x: f64) -> String {
    if x != 0.0 && (x.abs() < 1e-4 || x.abs() >= 1e16) {
        format!("{x:e}")
    } else {
        format!("{x}")
    }
}

#[cfg(test)]
mod tests {
    use clap::CommandFactory;

    #[test]
    fn cli_definition_is_valid() {
        super::Cli::command().debug_assert();
    }
}
