use reborrow::*;

use crate::{Mesh, Method, Problem, SimpleFloat};
use faer_core::{zipped, Mat, MatMut, MatRef};

pub struct LaxFriedrichs<F> {
    ratio: F,
    x_steps: usize,
}

impl<F: SimpleFloat, P: Problem<Float = F>> Method<P> for LaxFriedrichs<F> {
    fn init(domain: &Mesh<P::Float>) -> Self {
        Self {
            ratio: domain.time().step_size().div(domain.space().step_size()),
            x_steps: domain.space().steps(),
        }
    }

    fn next_to(&mut self, current: MatRef<'_, P::Float>, mut out: MatMut<'_, P::Float>) {
        let f = P::flux;
        let vp = current.submatrix(2, 0, self.x_steps - 1, 1);
        let vm = current.submatrix(0, 0, self.x_steps - 1, 1);

        // lax-friedrichs schema
        // let schema = |vm, vp| 0.5 * (vm + vp) - 0.5 * self.ratio * (f(vp) - f(vm));
        let schema = |vm: P::Float, vp: P::Float| {
            (vm.add(vp).sub(self.ratio.mul(f(vp).sub(f(vm))))).mul(F::from_f64(0.5))
        };

        // compute solution at time t_{n+1}
        zipped!(out.rb_mut().submatrix(1, 0, self.x_steps - 1, 1), vm, vp)
            .for_each(|mut out, vm, vp| out.write(schema(vm.read(), vp.read())));

        // boundary conditions
        let a = current[(1, 0)];
        let b = current[(self.x_steps - 1, 0)];
        out[(0, 0)] = schema(b, a);
        out[(self.x_steps, 0)] = out[(0, 0)];
    }

    fn name() -> &'static str {
        "Conservative LaxFriedrichs"
    }
}

pub struct MacCormack<F: SimpleFloat> {
    ratio: F,
    x_steps: usize,
    w: Mat<F>,
}

impl<F: SimpleFloat, P: Problem<Float = F>> Method<P> for MacCormack<F> {
    fn init(domain: &Mesh<F>) -> Self {
        Self {
            ratio: domain.time().step_size().div(domain.space().step_size()),
            x_steps: domain.space().steps(),
            w: Mat::zeros(domain.space().steps() + 1, 1),
        }
    }

    fn next_to(&mut self, current: MatRef<'_, F>, mut out: MatMut<'_, F>) {
        let f = P::flux;
        let v = current.submatrix(1, 0, self.x_steps - 1, 1);
        let vp = current.submatrix(2, 0, self.x_steps - 1, 1);
        let w = self.w.as_mut().submatrix(1, 0, self.x_steps - 1, 1);

        // let schema_1 = |v, vp| v - self.ratio * (f(vp) - f(v));
        // let schema_2 = |v, wm, w| 0.5 * (v + w) - 0.5 * self.ratio * (f(w) - f(wm));
        let schema_1 = |v: F, vp: F| v.sub(self.ratio.mul(vp.sub(f(v))));
        let schema_2 = |v: F, wm: F, w: F| {
            v.add(w)
                .sub(self.ratio.mul(f(w).sub(f(wm))))
                .mul(F::from_f64(0.5))
        };

        zipped!(w, v, vp).for_each(|mut w, v, vp| w.write(schema_1(v.read(), vp.read())));

        let w = self.w.as_ref().submatrix(1, 0, self.x_steps - 1, 1);
        let wm = self.w.as_ref().submatrix(0, 0, self.x_steps - 1, 1);

        // compute solution at time t_{n+1}
        zipped!(out.rb_mut().submatrix(1, 0, self.x_steps - 1, 1), v, wm, w)
            .for_each(|mut out, v, wm, w| out.write(schema_2(v.read(), wm.read(), w.read())));

        // boundary conditions
        let a = current[(0, 0)];
        let b = current[(1, 0)];
        let c = schema_1(a, b);
        let d = self.w[(self.x_steps - 1, 0)];
        out[(0, 0)] = schema_2(a, d, c);
        out[(self.x_steps, 0)] = out[(0, 0)];
    }

    fn name() -> &'static str {
        "MacCormack"
    }
}
