//! Compat against frozen `statsmodels` goldens — no statsmodels at test time.
//!
//! `tests/golden/expected.tsv` carries the little-endian IEEE-754 bits of the
//! `(lm, lm_pvalue, fvalue, f_pvalue)` that `het_breuschpagan` / `het_white`
//! produced under statsmodels 0.14.6, one row per (test, dataset). The data
//! files are `resid x0 x1 …` per row (x0 is the constant column). The LM and F
//! statistics flow through the auxiliary OLS solve, so they are asserted to
//! 1e-10 (the normal-equations solve tracks statsmodels' SVD pseudoinverse to a
//! few ULP); the p-values, transcendental and deep-tailed (down to 1e-200), are
//! asserted to 1e-10 as well, the conditioning of the auxiliary design
//! propagating through the steep χ²/F tail being the binding limit.

use std::path::PathBuf;
use std::process::{Command, Stdio};

fn bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_rsomics-het-tests"))
}

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden")
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(f64::MIN_POSITIVE)
}

fn from_hex(s: &str) -> f64 {
    let bytes = (0..16)
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect::<Vec<_>>();
    f64::from_le_bytes(bytes.try_into().unwrap())
}

fn run(args: &[&str]) -> Vec<f64> {
    let out = Command::new(bin()).args(args).output().expect("run binary");
    assert!(
        out.status.success(),
        "binary failed for {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout)
        .unwrap()
        .trim()
        .split('\t')
        .map(|s| s.parse().unwrap())
        .collect()
}

#[test]
fn matches_statsmodels_goldens() {
    let expected = std::fs::read_to_string(golden_dir().join("expected.tsv")).unwrap();
    let labels = ["lm", "lm_pvalue", "fvalue", "f_pvalue"];
    let mut checked = 0;
    for line in expected.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let c: Vec<&str> = line.split('\t').collect();
        let (test, file) = (c[0], c[1]);
        let want = [
            from_hex(c[2]),
            from_hex(c[3]),
            from_hex(c[4]),
            from_hex(c[5]),
        ];
        let path = golden_dir().join(file);

        let got = run(&["--test", test, "--data", path.to_str().unwrap()]);
        assert_eq!(got.len(), 4, "want 4 fields for {file} {test}");

        for i in 0..4 {
            assert!(
                rel(got[i], want[i]) <= 1e-10,
                "{test} {file} {}: got {}, want {}, rel {:e}",
                labels[i],
                got[i],
                want[i],
                rel(got[i], want[i])
            );
        }
        checked += 1;
    }
    assert!(checked >= 16, "expected >= 16 golden rows, ran {checked}");
}

#[test]
fn separate_matches_combined() {
    // Split a combined file into resid + exog and confirm the two input paths agree.
    let dir = tempfile::tempdir().unwrap();
    let combined = golden_dir().join("n200_k5_hetero.tsv");
    let text = std::fs::read_to_string(&combined).unwrap();
    let mut resid = String::new();
    let mut exog = String::new();
    for line in text.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        resid.push_str(f[0]);
        resid.push('\n');
        exog.push_str(&f[1..].join("\t"));
        exog.push('\n');
    }
    let rpath = dir.path().join("resid.txt");
    let xpath = dir.path().join("exog.tsv");
    std::fs::write(&rpath, resid).unwrap();
    std::fs::write(&xpath, exog).unwrap();

    for test in ["breuschpagan", "white"] {
        let combined_out = run(&["--test", test, "--data", combined.to_str().unwrap()]);
        let separate_out = run(&[
            "--test",
            test,
            "--resid",
            rpath.to_str().unwrap(),
            "--exog",
            xpath.to_str().unwrap(),
        ]);
        assert_eq!(combined_out, separate_out, "{test}: separate != combined");
    }
}

#[test]
fn stdin_matches_file() {
    let path = golden_dir().join("n50_k2_hetero.tsv");
    let text = std::fs::read_to_string(&path).unwrap();
    let from_file = run(&["--test", "white", "--data", path.to_str().unwrap()]);

    let out = Command::new(bin())
        .args(["--test", "white", "--data", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            child
                .stdin
                .take()
                .unwrap()
                .write_all(text.as_bytes())
                .unwrap();
            child.wait_with_output()
        })
        .expect("run stdin");
    let from_stdin: Vec<f64> = std::str::from_utf8(&out.stdout)
        .unwrap()
        .trim()
        .split('\t')
        .map(|s| s.parse().unwrap())
        .collect();
    assert_eq!(from_file, from_stdin, "stdin != file");
}

#[test]
fn json_envelope() {
    let path = golden_dir().join("n50_k3_hetero.tsv");
    let out = Command::new(bin())
        .args([
            "--test",
            "breuschpagan",
            "--data",
            path.to_str().unwrap(),
            "--json",
        ])
        .output()
        .expect("run --json");
    assert!(out.status.success());
    let s = String::from_utf8(out.stdout).unwrap();
    let v: serde_json::Value = serde_json::from_str(s.trim()).expect("one json envelope");
    assert_eq!(v["status"], "ok");
    assert!(v["result"]["lm"].is_number(), "missing lm: {s}");
    assert!(v["result"]["lm_pvalue"].is_number());
    assert!(v["result"]["fvalue"].is_number());
    assert!(v["result"]["f_pvalue"].is_number());
}

#[test]
fn singular_design_fails_loud() {
    // exog with two identical non-constant columns → XᵀX singular.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("singular.tsv");
    std::fs::write(
        &path,
        "1.0 1 2 2\n0.5 1 3 3\n0.2 1 4 4\n0.9 1 5 5\n0.1 1 6 6\n",
    )
    .unwrap();
    let out = Command::new(bin())
        .args(["--test", "breuschpagan", "--data", path.to_str().unwrap()])
        .output()
        .expect("run");
    assert!(
        !out.status.success(),
        "exactly collinear exog must fail loud"
    );
}

#[test]
fn help_exits_zero() {
    let out = Command::new(bin())
        .arg("--help")
        .output()
        .expect("run --help");
    assert!(out.status.success(), "--help did not exit 0");
}
