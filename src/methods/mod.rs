use faer_core::{zipped, MatMut, MatRef};
use reborrow::*;

use crate::{
    method::{Buffers, Method},
    problem::FluxFunction,
    Ctx, SimpleFloat,
};

pub struct UpwindLeft<F: SimpleFloat> {
    buf: Buffers<F, 2>,
}
impl<F: SimpleFloat> Default for UpwindLeft<F> {
    fn default() -> Self {
        Self {
            buf: Buffers::default(),
        }
    }
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

    fn apply(&mut self, ctx: Ctx<F>, flux: impl FluxFunction<F>, u: MatRef<F>, v: MatMut<F>) {
        // stores fluxes
        flux(ctx.left().as_ref(), self.buf.get_mut(0));
        flux(u.rb(), self.buf.get_mut(1));

        // component-wise schema
        let r = ctx.mesh.time.delta.div(ctx.mesh.space.delta);
        let schema = |u: F, fum: F, fu: F| u.sub(r.mul(fu.sub(fum)));

        // apply
        zipped!(v, u, self.buf.get(0), self.buf.get(1))
            .for_each(|mut v, u, fum, fu| v.write(schema(u.read(), fum.read(), fu.read())))
    }

    fn name(&self) -> &'static str {
        "Upwind (left)"
    }
}
