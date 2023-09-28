use crate::{Domain, Float, Method};
use faer_core::{Mat, MatMut, MatRef};

// generic two-level linear numerical method matrix builder with periodic boundary conditions
// where ql, qc, qr are
// ql == [q_{-1}, q_{-2}, ..., q_{-l}]
// qc == q_0
// qr == [q_1, q_2, ..., q_r]
// in
// U^{n+1}_j = \sum_{m=-l}^{r} q_m U^n_{j+m}
// U^n_j = U^n_{j+N} for all j
pub(crate) fn linear_periodic_matrix(
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
        #[allow(clippy::collapsible_else_if, clippy::comparison_chain)]
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

    let p = Mat::<Float>::from_fn(size + 1, size, |i, j| {
        if i == j || (i, j) == (size, 0) {
            1.0
        } else {
            0.0
        }
    });

    let q = Mat::<Float>::from_fn(size, size + 1, |i, j| if i == j { 1.0 } else { 0.0 });

    p * hb * q
}

macro_rules! linear_method {
    ($name: ident, $p:ident, $ql: expr, $qc: expr, $qr: expr) => {
        pub struct $name(Mat<Float>);
        impl Method for $name {
            type Meta = Float;
            fn init(domain: &Domain, a: Float) -> Self {
                let (t_step_size, x_step_size, x_steps) = (
                    domain.time().step_size(),
                    domain.space().step_size(),
                    domain.space().steps(),
                );

                let $p = a * t_step_size / x_step_size;
                Self(linear_periodic_matrix(x_steps, $ql, $qc, $qr))
            }
            fn next_to(&mut self, current: MatRef<'_, Float>, out: MatMut<'_, Float>) {
                faer_core::mul::matmul(
                    out,
                    self.0.as_ref(),
                    current,
                    None,
                    1.0,
                    faer_core::Parallelism::None,
                )
            }

            fn name(&self) -> &'static str {
                stringify!($name)
            }
        }
    };
}

linear_method!(BackwardEuler, p, &[0.5 * p], 1.0, &[-0.5 * p]);
linear_method!(UpwindLeft, p, &[p], 1.0 - p, &[]);
linear_method!(UpwindRight, p, &[], 1.0 + p, &[-p]);
linear_method!(LaxFriedrichs, p, &[0.5 + 0.5 * p], 0.0, &[0.5 - 0.5 * p]);
linear_method!(
    LaxWendroff,
    p,
    &[0.5 * p * (p + 1.0)],
    1.0 - p * p,
    &[0.5 * p * (p - 1.0)]
);
linear_method!(
    LaxWarming,
    p,
    &[p * (2.0 - p), 0.5 * p * (p - 1.0)],
    1.0 - 1.5 * p + 0.5 * p * p,
    &[]
);
