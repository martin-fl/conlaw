use std::marker::PhantomData;

use faer_core::{MatMut, MatRef};

use crate::{problem::BoundaryCondition, Ctx, SimpleFloat};

pub struct Periodic;

impl<F: SimpleFloat> BoundaryCondition<F> for Periodic {
    fn apply(&self, _ctx: Ctx<F>, mut left: MatMut<F>, center: MatRef<F>, mut right: MatMut<F>) {
        let corresponding = center.subrows(center.nrows() - left.nrows() - 1, left.nrows());
        left.clone_from(corresponding);

        let corresponding = center.subrows(0, right.nrows());
        right.clone_from(corresponding);
    }
}

pub struct Dirichlet<F, L, R> {
    pub(crate) _left: L,
    pub(crate) _right: R,
    _marker: PhantomData<F>,
}

impl<F, L, R> Dirichlet<F, L, R>
where
    F: SimpleFloat,
    L: Fn(F, MatMut<F>),
    R: Fn(F, MatMut<F>),
{
    pub fn new(left: L, right: R) -> Self {
        Self {
            _left: left,
            _right: right,
            _marker: PhantomData,
        }
    }
}

impl<F, L, R> BoundaryCondition<F> for Dirichlet<F, L, R>
where
    F: SimpleFloat,
    L: Fn(F, MatMut<F>),
    R: Fn(F, MatMut<F>),
{
    fn apply(&self, _ctx: Ctx<F>, _left: MatMut<F>, _center: MatRef<F>, _right: MatMut<F>) {
        todo!()
    }
}
