use std::rc::Rc;

use faer_core::{zipped, MatMut, MatRef};
use reborrow::*;

use crate::{
    method::{Buffers, Method},
    problem::FluxFunction,
    Ctx, SimpleFloat,
};

fn apply_flux_in_chunks<F: SimpleFloat>(
    u: MatRef<F>,
    fu: MatMut<F>,
    chunk_size: usize,
    flux: &Rc<dyn FluxFunction<F>>,
) {
    for (u, fu) in u
        .rb()
        .into_row_chunks(chunk_size)
        .zip(fu.into_row_chunks(chunk_size))
    {
        flux(u, fu)
    }
}

#[derive(Default)]
pub struct UpwindLeft<F: SimpleFloat> {
    buf: Buffers<F, 2>,
}

impl<F: SimpleFloat> Method<F> for UpwindLeft<F> {
    fn left_ghost_cells(&self) -> usize {
        1
    }

    fn right_ghost_cells(&self) -> usize {
        0
    }

    fn init(&mut self, ctx: Ctx<F>) {
        self.buf.resize(ctx.mesh.space.steps + 1);
    }

    fn apply(&mut self, ctx: Ctx<F>, flux: Rc<dyn FluxFunction<F>>, u: MatRef<F>, v: MatMut<F>) {
        // stores fluxes
        apply_flux_in_chunks(ctx.left(), self.buf.get_mut(0), ctx.system_size, &flux);
        apply_flux_in_chunks(u.rb(), self.buf.get_mut(1), ctx.system_size, &flux);

        // component-wise schema
        let r = ctx.mesh.time.delta.div(ctx.mesh.space.delta);
        let schema = |u: F, fum: F, fu: F| u.sub(r.mul(fu.sub(fum)));

        // apply
        zipped!(v, u, self.buf.get(0), self.buf.get(1))
            .for_each(|mut v, u, fum, fu| v.write(schema(u.read(), fum.read(), fu.read())))
    }

    fn name(&self) -> &'static str {
        "First-order upwind (left)"
    }
}

#[derive(Default)]
pub struct UpwindRight<F: SimpleFloat> {
    buf: Buffers<F, 2>,
}

impl<F: SimpleFloat> Method<F> for UpwindRight<F> {
    fn left_ghost_cells(&self) -> usize {
        1
    }

    fn right_ghost_cells(&self) -> usize {
        0
    }

    fn init(&mut self, ctx: Ctx<F>) {
        self.buf.resize(ctx.mesh.space.steps + 1);
    }

    fn apply(&mut self, ctx: Ctx<F>, flux: Rc<dyn FluxFunction<F>>, u: MatRef<F>, v: MatMut<F>) {
        // stores fluxes
        apply_flux_in_chunks(u.rb(), self.buf.get_mut(0), ctx.system_size, &flux);
        apply_flux_in_chunks(ctx.right(), self.buf.get_mut(1), ctx.system_size, &flux);

        // component-wise schema
        let r = ctx.mesh.time.delta.div(ctx.mesh.space.delta);
        let schema = |u: F, fu: F, fup: F| u.sub(r.mul(fup.sub(fu)));

        // apply
        zipped!(v, u, self.buf.get(0), self.buf.get(1))
            .for_each(|mut v, u, fu, fup| v.write(schema(u.read(), fu.read(), fup.read())))
    }

    fn name(&self) -> &'static str {
        "First-order upwind (right)"
    }
}

#[derive(Default)]
pub struct LaxFriedrichs<F: SimpleFloat> {
    buf: Buffers<F, 2>,
}

impl<F: SimpleFloat> Method<F> for LaxFriedrichs<F> {
    fn left_ghost_cells(&self) -> usize {
        1
    }

    fn right_ghost_cells(&self) -> usize {
        1
    }

    fn init(&mut self, ctx: Ctx<F>) {
        self.buf.resize(ctx.mesh.space.steps + 1);
    }

    fn apply(&mut self, ctx: Ctx<F>, flux: Rc<dyn FluxFunction<F>>, _u: MatRef<F>, v: MatMut<F>) {
        // stores fluxes
        apply_flux_in_chunks(ctx.left(), self.buf.get_mut(0), ctx.system_size, &flux);
        apply_flux_in_chunks(ctx.right(), self.buf.get_mut(1), ctx.system_size, &flux);

        // component-wise schema
        let r = ctx.mesh.time.delta.div(ctx.mesh.space.delta);
        let schema = |um: F, up: F, fum: F, fup: F| -> F {
            (um.add(up).sub(r.mul(fup.sub(fum)))).mul(F::from_f64(0.5))
        };

        // apply
        zipped!(v, ctx.left(), ctx.right(), self.buf.get(0), self.buf.get(1)).for_each(
            |mut v, um, up, fum, fup| v.write(schema(um.read(), up.read(), fum.read(), fup.read())),
        )
    }

    fn name(&self) -> &'static str {
        "Lax-Friedrichs"
    }
}

#[derive(Default)]
pub struct MacCormack<F: SimpleFloat> {
    buf_a: Buffers<F, 3>,
    buf_b: Buffers<F, 2>,
}

impl<F: SimpleFloat> Method<F> for MacCormack<F> {
    fn left_ghost_cells(&self) -> usize {
        1
    }

    fn right_ghost_cells(&self) -> usize {
        1
    }

    fn init(&mut self, ctx: Ctx<F>) {
        self.buf_a.resize(ctx.mesh.space.steps + 1);
        self.buf_b.resize(ctx.mesh.space.steps + 1);
    }

    fn apply(&mut self, ctx: Ctx<F>, flux: Rc<dyn FluxFunction<F>>, u: MatRef<F>, v: MatMut<F>) {
        // stores fluxes
        apply_flux_in_chunks(ctx.left(), self.buf_a.get_mut(0), ctx.system_size, &flux);
        apply_flux_in_chunks(u.rb(), self.buf_a.get_mut(1), ctx.system_size, &flux);
        apply_flux_in_chunks(ctx.right(), self.buf_a.get_mut(2), ctx.system_size, &flux);

        // component-wise schema
        let r = ctx.mesh.time.delta.div(ctx.mesh.space.delta);
        let forward = |u: F, fu: F, fup: F| -> F { u.sub(r.mul(fup.sub(fu))) };

        zipped!(
            self.buf_b.get_mut(0),
            u.rb(),
            self.buf_a.get(1),
            self.buf_a.get(2)
        )
        .for_each(|mut v, u, fu, fup| v.write(forward(u.read(), fu.read(), fup.read())));

        zipped!(
            self.buf_b.get_mut(1),
            ctx.left(),
            self.buf_a.get(0),
            self.buf_a.get(1)
        )
        .for_each(|mut v, um, fum, fu| v.write(forward(um.read(), fum.read(), fu.read())));

        apply_flux_in_chunks(
            self.buf_b.get(0),
            self.buf_a.get_mut(0),
            ctx.system_size,
            &flux,
        );

        apply_flux_in_chunks(
            self.buf_b.get(1),
            self.buf_a.get_mut(1),
            ctx.system_size,
            &flux,
        );

        let backward = |u: F, us: F, fusm: F, fus: F| -> F {
            (u.add(us).sub(r.mul(fus.sub(fusm)))).mul(F::from_f64(0.5))
        };

        // apply
        zipped!(
            v,
            u,
            self.buf_b.get(0),
            self.buf_a.get(1),
            self.buf_a.get(0)
        )
        .for_each(|mut v, u, us, fusm, fus| {
            v.write(backward(u.read(), us.read(), fusm.read(), fus.read()))
        })
    }

    fn name(&self) -> &'static str {
        "MacCormarck"
    }
}

// let f = P::flux;
//        let v = current.submatrix(1, 0, self.x_steps - 1, 1);
//        let vp = current.submatrix(2, 0, self.x_steps - 1, 1);
//        let w = self.w.as_mut().submatrix(1, 0, self.x_steps - 1, 1);

//        // let schema_1 = |v, vp| v - self.ratio * (f(vp) - f(v));
//        // let schema_2 = |v, wm, w| 0.5 * (v + w) - 0.5 * self.ratio * (f(w) - f(wm));
//        let schema_1 = |v: F, vp: F| v.sub(self.ratio.mul(vp.sub(f(v))));
//        let schema_2 = |v: F, wm: F, w: F| {
//            v.add(w)
//                .sub(self.ratio.mul(f(w).sub(f(wm))))
//                .mul(F::from_f64(0.5))
//        };

//        zipped!(w, v, vp).for_each(|mut w, v, vp| w.write(schema_1(v.read(), vp.read())));

//        let w = self.w.as_ref().submatrix(1, 0, self.x_steps - 1, 1);
//        let wm = self.w.as_ref().submatrix(0, 0, self.x_steps - 1, 1);

//        // compute solution at time t_{n+1}
//        zipped!(out.rb_mut().submatrix(1, 0, self.x_steps - 1, 1), v, wm, w)
//            .for_each(|mut out, v, wm, w| out.write(schema_2(v.read(), wm.read(), w.read())));
