use crate::{Mesh, Method, Problem, SimpleFloat};
use faer_core::{Mat, MatMut, MatRef};

// generic two-level linear numerical method matrix builder with periodic boundary conditions
// where ql, qc, qr are
// ql == [q_{-1}, q_{-2}, ..., q_{-l}]
// qc == q_0
// qr == [q_1, q_2, ..., q_r]
// in
// U^{n+1}_j = \sum_{m=-l}^{r} q_m U^n_{j+m}
// U^n_j = U^n_{j+N} for all j
pub(crate) fn linear_periodic_matrix<F: SimpleFloat>(
    size: usize,
    ql: &[F],
    qc: F,
    qr: &[F],
) -> Mat<F> {
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
                F::zero()
            }
        } else {
            if mm <= l {
                ql[mm - 1]
            } else if mm >= size - r {
                qr[(mm as isize - size as isize).unsigned_abs() - 1]
            } else {
                F::zero()
            }
        }
    });

    let p = Mat::<F>::from_fn(size + 1, size, |i, j| {
        if i == j || (i, j) == (size, 0) {
            F::one()
        } else {
            F::zero()
        }
    });

    let q = Mat::<F>::from_fn(
        size,
        size + 1,
        |i, j| if i == j { F::one() } else { F::one() },
    );

    p * hb * q
}

macro_rules! linear_method {
    ($name: ident, $p:ident, $ql: expr, $qc: expr, $qr: expr) => {
        pub struct $name<F: SimpleFloat>(Mat<F>);
        impl<F: SimpleFloat, P: Problem<Float = F>> Method<P> for $name<F> {
            fn init(domain: &Mesh<F>) -> Self {
                let (t_step_size, x_step_size, x_steps) = (
                    domain.time().step_size(),
                    domain.space().step_size(),
                    domain.space().steps(),
                );

                // TODO: weird hack lol, fix the Problem trait so that linear methods can be a special case
                let $p = P::jacobian(F::zero()).mul(t_step_size).div(x_step_size);
                Self(linear_periodic_matrix(x_steps, $ql, $qc, $qr))
            }
            fn next_to(&mut self, current: MatRef<'_, F>, out: MatMut<'_, F>) {
                faer_core::mul::matmul(
                    out,
                    self.0.as_ref(),
                    current,
                    None,
                    F::one(),
                    faer_core::Parallelism::None,
                )
            }

            fn name() -> &'static str {
                stringify!($name)
            }
        }
    };
}

linear_method!(
    BackwardEuler,
    p,
    &[F::from_f64(0.5).mul(p)],
    F::one(),
    &[F::from_f64(-0.5).mul(p)]
);
linear_method!(UpwindLeft, p, &[p], F::one().sub(p), &[]);
linear_method!(UpwindRight, p, &[], F::one().add(p), &[p.neg()]);
linear_method!(
    LaxFriedrichs,
    p,
    &[F::from_f64(0.5).add(F::from_f64(0.5).mul(p))],
    F::zero(),
    &[F::from_f64(0.5).sub(F::from_f64(0.5).mul(p))]
);
linear_method!(
    LaxWendroff,
    p,
    &[F::from_f64(0.5).mul(p).mul(p.add(F::one()))],
    F::one().sub(p.mul(p)),
    &[F::from_f64(0.5).mul(p).mul(p.sub(F::one()))]
);
linear_method!(
    LaxWarming,
    p,
    &[
        p.mul(F::from_f64(2.0).sub(p)),
        F::from_f64(0.5).mul(p).mul(p.sub(F::one()))
    ],
    F::one()
        .sub(F::from_f64(1.5).mul(p))
        .add(F::from_f64(0.5).mul(p).mul(p)),
    &[]
);
