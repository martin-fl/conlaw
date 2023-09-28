use std::rc::Rc;

use reborrow::*;

use crate::{Domain, Float, Method};
use faer_core::{zipped, Mat, MatMut, MatRef};

pub struct LaxFriedrichs {
    flux: Rc<dyn Fn(Float) -> Float>,
    ratio: Float,
    x_steps: usize,
}

impl Method for LaxFriedrichs {
    type Meta = Rc<dyn Fn(Float) -> Float>;

    fn init(domain: &Domain, flux: Self::Meta) -> Self {
        Self {
            flux,
            ratio: domain.time().step_size() / domain.space().step_size(),
            x_steps: domain.space().steps(),
        }
    }

    fn next_to(&mut self, current: MatRef<'_, Float>, mut out: MatMut<'_, Float>) {
        let f = self.flux.clone();
        let vp = current.submatrix(2, 0, self.x_steps - 1, 1);
        let vm = current.submatrix(0, 0, self.x_steps - 1, 1);

        // lax-friedrichs schema
        let schema = |vm, vp| 0.5 * (vm + vp) - 0.5 * self.ratio * (f(vp) - f(vm));

        // compute solution at time t_{n+1}
        zipped!(out.rb_mut().submatrix(1, 0, self.x_steps - 1, 1), vm, vp)
            .for_each(|mut out, vm, vp| out.write(schema(vm.read(), vp.read())));

        // boundary conditions
        let a = current[(1, 0)];
        let b = current[(self.x_steps - 1, 0)];
        out[(0, 0)] = schema(b, a);
        out[(self.x_steps, 0)] = out[(0, 0)];
    }

    fn name(&self) -> &'static str {
        "Conservative LaxFriedrichs"
    }
}

pub struct MacCormack {
    flux: Rc<dyn Fn(Float) -> Float>,
    ratio: Float,
    x_steps: usize,
    w: Mat<Float>,
}

impl Method for MacCormack {
    type Meta = Rc<dyn Fn(Float) -> Float>;

    fn init(domain: &Domain, flux: Self::Meta) -> Self {
        Self {
            flux,
            ratio: domain.time().step_size() / domain.space().step_size(),
            x_steps: domain.space().steps(),
            w: Mat::zeros(domain.space().steps() + 1, 1),
        }
    }

    fn next_to(&mut self, current: MatRef<'_, Float>, mut out: MatMut<'_, Float>) {
        let f = self.flux.clone();
        let v = current.submatrix(1, 0, self.x_steps - 1, 1);
        let vp = current.submatrix(2, 0, self.x_steps - 1, 1);
        let w = self.w.as_mut().submatrix(1, 0, self.x_steps - 1, 1);

        let schema_1 = |v, vp| v - self.ratio * (f(vp) - f(v));
        let schema_2 = |v, wm, w| 0.5 * (v + w) - 0.5 * self.ratio * (f(w) - f(wm));

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

    fn name(&self) -> &'static str {
        "MacCormack"
    }
}
