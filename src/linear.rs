use crate::Float;
use faer::Mat;
use std::sync::OnceLock;

// generic two-level linear numerical method matrix builder with periodic boundary conditions
// where ql, qc, qr are
// ql == [q_{-1}, q_{-2}, ..., q_{-l}]
// qc == q_0
// qr == [q_1, q_2, ..., q_r]
// in
// U^{n+1}_j = \sum_{m=-l}^{r} q_m U^n_{j+m}
// U^n_j = U^n_{j+N} for all j
pub(crate) fn linear_periodic_scheme(
    size: usize,
    ql: &[Float],
    qc: Float,
    qr: &[Float],
) -> Mat<Float> {
    let l = ql.len();
    let r = qr.len();
    assert!(l + r < size);

    let hb = Mat::from_fn(size, size, |i, j| {
        let m = j as isize - i as isize;
        let mm = m.unsigned_abs();
        if m == 0 {
            qc
        } else if m > 0 {
            if mm <= r {
                qr[mm - 1]
            } else if mm >= size - l {
                ql[(mm as isize - size as isize).unsigned_abs() - 1]
            } else {
                0.0
            }
        } else {
            if mm <= l {
                ql[mm - 1]
            } else if mm >= size - r {
                qr[(mm as isize - size as isize).unsigned_abs() - 1]
            } else {
                0.0
            }
        }
    });

    static P: OnceLock<Mat<Float>> = OnceLock::new();
    let p = P.get_or_init(|| {
        Mat::<Float>::from_fn(size + 1, size, |i, j| {
            if i == j || (i, j) == (size, 0) {
                1.0
            } else {
                0.0
            }
        })
    });

    static Q: OnceLock<Mat<Float>> = OnceLock::new();
    let q = Q.get_or_init(|| {
        Mat::<Float>::from_fn(size, size + 1, |i, j| if i == j { 1.0 } else { 0.0 })
    });

    // TODO: this will panick if this function is called multiple times
    //       with different sizes, because of the statics
    p * hb * q
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Schema {
    BackwardEuler,
    UpwindLeft,
    UpwindRight,
    LaxFriedrichs,
    LaxWendroff,
    LaxWarming,
}

pub use Schema::*;

#[derive(Clone, Debug)]
pub struct Driver {
    schema: Schema,
    b: Mat<Float>,
}

impl Driver {
    pub fn new(schema: Schema, t_step: Float, x_step: Float, x_size: usize, a: Float) -> Self {
        let p = a * t_step / x_step;

        let b = match schema {
            BackwardEuler => linear_periodic_scheme(x_size, &[0.5 * p], 1.0, &[-0.5 * p]),
            UpwindLeft => linear_periodic_scheme(x_size, &[p], 1.0 - p, &[]),
            UpwindRight => linear_periodic_scheme(x_size, &[], 1.0 + p, &[-p]),
            LaxFriedrichs => {
                linear_periodic_scheme(x_size, &[0.5 + 0.5 * p], 0.0, &[0.5 - 0.5 * p])
            }
            LaxWendroff => linear_periodic_scheme(
                x_size,
                &[0.5 * p * (p + 1.0)],
                1.0 - p * p,
                &[0.5 * p * (p - 1.0)],
            ),
            LaxWarming => linear_periodic_scheme(
                x_size,
                &[p * (2.0 - p), 0.5 * p * (p - 1.0)],
                1.0 - 1.5 * p + 0.5 * p * p,
                &[],
            ),
        };

        Self { schema, b }
    }

    pub fn next(&self, unj: &Mat<Float>) -> Mat<Float> {
        &self.b * unj
    }

    pub fn name(&self) -> &'static str {
        match self.schema {
            BackwardEuler => "Backward-Euler",
            UpwindLeft => "Upwind (Left)",
            UpwindRight => "Upwind (Right)",
            LaxFriedrichs => "Lax-Friedrichs",
            LaxWendroff => "Lax-Wendroff",
            LaxWarming => "Lax-Warming",
        }
    }
}
