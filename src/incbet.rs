//! Regularized incomplete beta integral and the F survival function it drives.
//!
//! `incbet(a, b, x)` is Cephes `incbet`, the regularized incomplete beta
//! I_x(a, b). The F-statistic p-value `scipy.stats.f.sf(F, dfn, dfd)` reduces to
//! it through `scipy.special.fdtrc(dfn, dfd, x) = incbet(dfd/2, dfn/2,
//! dfd/(dfd + dfn·x))`. Routing the complement through the `incbet` symmetry
//! relation rather than `1 − I_x` avoids the cancellation that wrecks the far
//! tail, where the het F p-values reach 1e-200 and below.

const MACHEP: f64 = 1.110_223_024_625_156_540_423_631_668_090_820_312_5e-16;
const MAXLOG: f64 = 7.097_827_128_933_839_730_962_063_185_871e2;
const MINLOG: f64 = -7.451_332_191_019_412e2;
const MAXGAM: f64 = 171.624_376_956_302_7;
const BIG: f64 = 4.503_599_627_370_496e15;
const BIGINV: f64 = 2.220_446_049_250_313e-16;

fn lgam(x: f64) -> f64 {
    libm::lgamma(x)
}

fn lbeta(a: f64, b: f64) -> f64 {
    lgam(a) + lgam(b) - lgam(a + b)
}

fn beta(a: f64, b: f64) -> f64 {
    lbeta(a, b).exp()
}

/// Upper-tail F survival function `P(F > x)` with `dfn`/`dfd` degrees of
/// freedom — Cephes `fdtrc`, the driver behind `scipy.stats.f.sf`.
#[must_use]
pub fn f_sf(dfn: f64, dfd: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    let w = dfd / (dfd + dfn * x);
    incbet(0.5 * dfd, 0.5 * dfn, w)
}

/// Regularized incomplete beta integral I_x(a, b), Cephes `incbet`.
#[must_use]
pub fn incbet(aa: f64, bb: f64, xx: f64) -> f64 {
    if aa <= 0.0 || bb <= 0.0 {
        return f64::NAN;
    }
    if xx <= 0.0 {
        return 0.0;
    }
    if xx >= 1.0 {
        return 1.0;
    }

    let mut flag = 0;
    if bb * xx <= 1.0 && xx <= 0.95 {
        return pseries(aa, bb, xx);
    }

    let w0 = 1.0 - xx;
    let (a, b, xc, x);
    if xx > aa / (aa + bb) {
        flag = 1;
        a = bb;
        b = aa;
        xc = xx;
        x = w0;
    } else {
        a = aa;
        b = bb;
        xc = w0;
        x = xx;
    }

    let mut t;
    if flag == 1 && b * x <= 1.0 && x <= 0.95 {
        t = pseries(a, b, x);
    } else {
        let y = x * (a + b - 2.0) - (a - 1.0);
        let w = if y < 0.0 {
            incbcf(a, b, x)
        } else {
            incbd(a, b, x) / xc
        };

        let y = a * x.ln();
        let tt = b * xc.ln();
        if (a + b) < MAXGAM && y.abs() < MAXLOG && tt.abs() < MAXLOG {
            t = xc.powf(b);
            t *= x.powf(a);
            t /= a;
            t *= w;
            t *= 1.0 / beta(a, b);
        } else {
            let mut yy = y + tt - lbeta(a, b);
            yy += (w / a).ln();
            t = if yy < MINLOG { 0.0 } else { yy.exp() };
        }
    }

    if flag == 1 {
        t = if t <= MACHEP { 1.0 - MACHEP } else { 1.0 - t };
    }
    t
}

/// Power series for I_x(a, b); used when `b·x` is small and `x` not near 1.
fn pseries(a: f64, b: f64, x: f64) -> f64 {
    let ai = 1.0 / a;
    let mut u = (1.0 - b) * x;
    let mut v = u / (a + 1.0);
    let t1 = v;
    let mut t = u;
    let mut n = 2.0;
    let mut s = 0.0;
    let z = MACHEP * ai;
    while v.abs() > z {
        u = (n - b) * x / n;
        t *= u;
        v = t / (a + n);
        s += v;
        n += 1.0;
    }
    s += t1;
    s += ai;

    u = a * x.ln();
    if (a + b) < MAXGAM && u.abs() < MAXLOG {
        let t = 1.0 / beta(a, b);
        (s * t) * x.powf(a)
    } else {
        let t = -lbeta(a, b) + u + s.ln();
        if t < MINLOG { 0.0 } else { t.exp() }
    }
}

/// Continued-fraction expansion #1 for I_x(a, b).
fn incbcf(a: f64, b: f64, x: f64) -> f64 {
    let mut k1 = a;
    let mut k2 = a + b;
    let mut k3 = a;
    let mut k4 = a + 1.0;
    let mut k5 = 1.0;
    let mut k6 = b - 1.0;
    let mut k7 = k4;
    let mut k8 = a + 2.0;

    let mut pkm2 = 0.0;
    let mut qkm2 = 1.0;
    let mut pkm1 = 1.0;
    let mut qkm1 = 1.0;
    let mut ans = 1.0;
    let mut r = 1.0;
    let thresh = 3.0 * MACHEP;

    for _ in 0..300 {
        let mut xk = -(x * k1 * k2) / (k3 * k4);
        let mut pk = pkm1 + pkm2 * xk;
        let mut qk = qkm1 + qkm2 * xk;
        pkm2 = pkm1;
        pkm1 = pk;
        qkm2 = qkm1;
        qkm1 = qk;

        xk = (x * k5 * k6) / (k7 * k8);
        pk = pkm1 + pkm2 * xk;
        qk = qkm1 + qkm2 * xk;
        pkm2 = pkm1;
        pkm1 = pk;
        qkm2 = qkm1;
        qkm1 = qk;

        if qk != 0.0 {
            r = pk / qk;
        }
        let t = if r != 0.0 {
            let t = ((ans - r) / r).abs();
            ans = r;
            t
        } else {
            1.0
        };
        if t < thresh {
            break;
        }

        k1 += 1.0;
        k2 += 1.0;
        k3 += 2.0;
        k4 += 2.0;
        k5 += 1.0;
        k6 -= 1.0;
        k7 += 2.0;
        k8 += 2.0;

        if qk.abs() + pk.abs() > BIG {
            pkm2 *= BIGINV;
            pkm1 *= BIGINV;
            qkm2 *= BIGINV;
            qkm1 *= BIGINV;
        }
        if qk.abs() < BIGINV || pk.abs() < BIGINV {
            pkm2 *= BIG;
            pkm1 *= BIG;
            qkm2 *= BIG;
            qkm1 *= BIG;
        }
    }
    ans
}

/// Continued-fraction expansion #2 for I_x(a, b).
fn incbd(a: f64, b: f64, x: f64) -> f64 {
    let mut k1 = a;
    let mut k2 = b - 1.0;
    let mut k3 = a;
    let mut k4 = a + 1.0;
    let mut k5 = 1.0;
    let mut k6 = a + b;
    let mut k7 = a + 1.0;
    let mut k8 = a + 2.0;

    let mut pkm2 = 0.0;
    let mut qkm2 = 1.0;
    let mut pkm1 = 1.0;
    let mut qkm1 = 1.0;
    let z = x / (1.0 - x);
    let mut ans = 1.0;
    let mut r = 1.0;
    let thresh = 3.0 * MACHEP;

    for _ in 0..300 {
        let mut xk = -(z * k1 * k2) / (k3 * k4);
        let mut pk = pkm1 + pkm2 * xk;
        let mut qk = qkm1 + qkm2 * xk;
        pkm2 = pkm1;
        pkm1 = pk;
        qkm2 = qkm1;
        qkm1 = qk;

        xk = (z * k5 * k6) / (k7 * k8);
        pk = pkm1 + pkm2 * xk;
        qk = qkm1 + qkm2 * xk;
        pkm2 = pkm1;
        pkm1 = pk;
        qkm2 = qkm1;
        qkm1 = qk;

        if qk != 0.0 {
            r = pk / qk;
        }
        let t = if r != 0.0 {
            let t = ((ans - r) / r).abs();
            ans = r;
            t
        } else {
            1.0
        };
        if t < thresh {
            break;
        }

        k1 += 1.0;
        k2 -= 1.0;
        k3 += 2.0;
        k4 += 2.0;
        k5 += 1.0;
        k6 += 1.0;
        k7 += 2.0;
        k8 += 2.0;

        if qk.abs() + pk.abs() > BIG {
            pkm2 *= BIGINV;
            pkm1 *= BIGINV;
            qkm2 *= BIGINV;
            qkm1 *= BIGINV;
        }
        if qk.abs() < BIGINV || pk.abs() < BIGINV {
            pkm2 *= BIG;
            pkm1 *= BIG;
            qkm2 *= BIG;
            qkm1 *= BIG;
        }
    }
    ans
}

#[cfg(test)]
mod tests {
    use super::{f_sf, incbet};

    fn rel(got: f64, want: f64) -> f64 {
        (got - want).abs() / want.abs().max(f64::MIN_POSITIVE)
    }

    #[test]
    fn incbet_matches_scipy_betainc() {
        let cases = [
            (0.5, 0.5, 0.3, 0.369_010_119_565_545_36),
            (2.0, 3.0, 0.4, 0.524_799_999_999_999_9),
            (1.0, 1.0, 0.25, 0.25),
            (5.0, 2.0, 0.7, 0.420_174_999_999_999_9),
            (50.0, 50.0, 0.5, 0.500_000_000_000_000_3),
            (2.5, 7.5, 0.2, 0.401_238_698_247_191_7),
        ];
        for (a, b, x, want) in cases {
            let r = rel(incbet(a, b, x), want);
            assert!(r <= 1e-12, "betainc({a},{b},{x}) rel {r:e}");
        }
    }

    #[test]
    fn f_sf_matches_scipy_fdtrc() {
        let cases = [
            (2.0, 18.0, 5.0, 0.018_751_251_717_798_734),
            (3.0, 27.0, 8.123, 0.000_515_432_545_281_164),
            (4.0, 40.0, 0.5, 0.735_831_847_513_954),
            (1.0, 48.0, 19.601_339_715_270_75, 5.489_626_087_115_685_4e-5),
        ];
        for (dfn, dfd, x, want) in cases {
            let r = rel(f_sf(dfn, dfd, x), want);
            assert!(r <= 1e-12, "f_sf({dfn},{dfd},{x}) rel {r:e}");
        }
    }
}
