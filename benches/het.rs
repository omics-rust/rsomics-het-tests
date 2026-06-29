use criterion::{Criterion, criterion_group, criterion_main};
use rsomics_het_tests::{breuschpagan, white};
use std::hint::black_box;

fn xorshift() -> impl FnMut() -> u64 {
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    }
}

/// A `nobs × ncols` design (column 0 = const) with heteroscedastic residuals.
fn dataset(nobs: usize, ncols: usize) -> (Vec<f64>, Vec<f64>) {
    let mut next = xorshift();
    let unit = || (next() >> 11) as f64 / (1u64 << 53) as f64;
    let mut next2 = unit;
    let mut exog = vec![0.0; nobs * ncols];
    let mut resid = vec![0.0; nobs];
    for i in 0..nobs {
        exog[i * ncols] = 1.0;
        for j in 1..ncols {
            exog[i * ncols + j] = next2() * 2.0 - 1.0;
        }
        let scale = 1.0 + 2.0 * exog[i * ncols + 1].abs();
        resid[i] = (next2() * 2.0 - 1.0) * scale;
    }
    (resid, exog)
}

fn bench_breuschpagan(c: &mut Criterion) {
    let nobs = 100_000;
    let ncols = 4;
    let (resid, exog) = dataset(nobs, ncols);
    c.bench_function("breuschpagan_100k_k4", |b| {
        b.iter(|| black_box(breuschpagan(&resid, &exog, nobs, ncols).unwrap().lm));
    });
}

fn bench_white(c: &mut Criterion) {
    let nobs = 100_000;
    let ncols = 4;
    let (resid, exog) = dataset(nobs, ncols);
    c.bench_function("white_100k_k4", |b| {
        b.iter(|| black_box(white(&resid, &exog, nobs, ncols).unwrap().lm));
    });
}

criterion_group!(benches, bench_breuschpagan, bench_white);
criterion_main!(benches);
