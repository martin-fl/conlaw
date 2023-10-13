use std::{marker::PhantomData, rc::Rc};

use faer_core::{Mat, MatMut, MatRef};

use crate::{problem::FluxFunction, Ctx, SimpleFloat};

pub trait Method<F: SimpleFloat> {
    fn left_ghost_cells(&self) -> usize;
    fn right_ghost_cells(&self) -> usize;
    fn init(&mut self, ctx: Ctx<F>);
    fn apply(&mut self, ctx: Ctx<F>, flux: Rc<dyn FluxFunction<F>>, u: MatRef<F>, v: MatMut<F>);
    fn name(&self) -> &'static str;
}

#[derive(Default)]
pub struct Buffers<F: SimpleFloat, const N: usize> {
    inner: Mat<F>,
    _marker: PhantomData<[(); N]>,
}

impl<F: SimpleFloat, const N: usize> Buffers<F, N> {
    pub fn resize(&mut self, size: usize) {
        self.inner.resize_with(size, N, |_, _| F::zero())
    }

    pub fn get(&self, n: usize) -> MatRef<F> {
        self.inner.as_ref().col(n)
    }

    pub fn get_mut(&mut self, n: usize) -> MatMut<F> {
        self.inner.as_mut().col(n)
    }
}
