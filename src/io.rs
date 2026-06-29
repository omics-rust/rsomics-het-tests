//! Input parsing: the residual vector and the exog design matrix.
//!
//! Two layouts, both whitespace/tab-delimited, `#` comments and blank lines
//! skipped:
//!
//! - **separate** — `--resid` is one residual per line; `--exog` is the
//!   `nobs × ncols` design, one row per line, the constant column included
//!   (statsmodels expects exog to carry the constant).
//! - **combined** — `--data` is `resid x0 x1 …` per line: the first field is the
//!   residual, the rest are that row's design columns (again including the
//!   constant column).

use std::fs::File;
use std::io::Read;
use std::path::Path;

use rsomics_common::{Result, RsomicsError};

/// A parsed problem: the residual vector and the row-major `nobs × ncols` design.
pub struct Dataset {
    pub resid: Vec<f64>,
    pub exog: Vec<f64>,
    pub nobs: usize,
    pub ncols: usize,
}

fn read_to_string(path: &Path) -> Result<String> {
    let mut buf = String::new();
    if path.as_os_str() == "-" {
        std::io::stdin()
            .lock()
            .read_to_string(&mut buf)
            .map_err(RsomicsError::Io)?;
    } else {
        File::open(path)
            .map_err(RsomicsError::Io)?
            .read_to_string(&mut buf)
            .map_err(RsomicsError::Io)?;
    }
    Ok(buf)
}

fn parse_field(tok: &str, lineno: usize) -> Result<f64> {
    fast_float2::parse(tok.as_bytes())
        .map_err(|_| RsomicsError::InvalidInput(format!("line {lineno}: '{tok}' is not a number")))
}

/// Parse a residual vector (one value per non-comment line).
fn parse_resid(text: &str) -> Result<Vec<f64>> {
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        out.push(parse_field(t, i + 1)?);
    }
    if out.is_empty() {
        return Err(RsomicsError::InvalidInput("no residuals in input".into()));
    }
    Ok(out)
}

/// Parse a rectangular matrix; every non-blank row must carry the same width.
fn parse_matrix(text: &str) -> Result<(Vec<f64>, usize, usize)> {
    let mut values = Vec::new();
    let mut width: Option<usize> = None;
    let mut rows = 0usize;
    for (i, line) in text.lines().enumerate() {
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.is_empty() || fields[0].starts_with('#') {
            continue;
        }
        match width {
            None => width = Some(fields.len()),
            Some(w) if w != fields.len() => {
                return Err(RsomicsError::InvalidInput(format!(
                    "line {}: row has {} columns but the matrix is {w} wide",
                    i + 1,
                    fields.len()
                )));
            }
            Some(_) => {}
        }
        for f in fields {
            values.push(parse_field(f, i + 1)?);
        }
        rows += 1;
    }
    let width = width.ok_or_else(|| RsomicsError::InvalidInput("empty matrix".into()))?;
    Ok((values, rows, width))
}

/// Read separate `--resid` and `--exog` files into a [`Dataset`].
pub fn read_separate(resid_path: &Path, exog_path: &Path) -> Result<Dataset> {
    let resid = parse_resid(&read_to_string(resid_path)?)?;
    let (exog, nobs, ncols) = parse_matrix(&read_to_string(exog_path)?)?;
    check_shape(resid.len(), nobs, ncols)?;
    Ok(Dataset {
        resid,
        exog,
        nobs,
        ncols,
    })
}

/// Read a combined `resid x0 x1 …` file into a [`Dataset`]: column 0 is the
/// residual, the remaining columns are that row's design.
pub fn read_combined(path: &Path) -> Result<Dataset> {
    let (raw, nobs, total) = parse_matrix(&read_to_string(path)?)?;
    if total < 2 {
        return Err(RsomicsError::InvalidInput(
            "combined input needs resid plus at least one exog column per row".into(),
        ));
    }
    let ncols = total - 1;
    let mut resid = Vec::with_capacity(nobs);
    let mut exog = Vec::with_capacity(nobs * ncols);
    for r in 0..nobs {
        resid.push(raw[r * total]);
        exog.extend_from_slice(&raw[r * total + 1..r * total + total]);
    }
    check_shape(resid.len(), nobs, ncols)?;
    Ok(Dataset {
        resid,
        exog,
        nobs,
        ncols,
    })
}

fn check_shape(nresid: usize, nobs: usize, ncols: usize) -> Result<()> {
    if nresid != nobs {
        return Err(RsomicsError::InvalidInput(format!(
            "resid has {nresid} rows but exog has {nobs}"
        )));
    }
    if ncols < 2 {
        return Err(RsomicsError::InvalidInput(
            "exog needs the constant column plus at least one regressor".into(),
        ));
    }
    if nobs <= ncols {
        return Err(RsomicsError::InvalidInput(format!(
            "need more observations ({nobs}) than regressors ({ncols})"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_matrix_shape() {
        let (v, r, c) = parse_matrix("1 2 3\n4 5 6\n").unwrap();
        assert_eq!((r, c), (2, 3));
        assert_eq!(v, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn rejects_ragged_matrix() {
        assert!(parse_matrix("1 2 3\n4 5\n").is_err());
    }

    #[test]
    fn combined_splits_resid_off() {
        let (raw, nobs, total) = parse_matrix("0.5 1 2\n0.6 1 3\n0.7 1 4\n").unwrap();
        assert_eq!((nobs, total), (3, 3));
        assert_eq!(raw[0], 0.5);
    }
}
