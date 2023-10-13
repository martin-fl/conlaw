use std::fmt;
use std::rc::Rc;

use faer_core::{MatMut, MatRef};

use crate::{Ctx, SimpleFloat};

pub trait FluxFunction<F: SimpleFloat>: Fn(MatRef<F>, MatMut<F>) {}
impl<F: SimpleFloat, T> FluxFunction<F> for T where T: Fn(MatRef<F>, MatMut<F>) {}

/// A hyperbolic PDE of the form `u_t + (f(u))_x = 0`
#[derive(Clone)]
pub struct ConservationLaw<F: SimpleFloat> {
    pub(crate) m: usize,
    pub(crate) flux: Rc<dyn FluxFunction<F>>,
}

impl<F: SimpleFloat> ConservationLaw<F> {
    pub fn new(m: usize, flux: impl FluxFunction<F> + 'static) -> Self {
        Self {
            m,
            flux: Rc::new(flux),
        }
    }
}

impl<F: SimpleFloat> fmt::Debug for ConservationLaw<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConservationLaw")
            .field("m", &self.m)
            .field("flux", &"<function>")
            .finish()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Domain<F> {
    pub time: (F, F),
    pub space: (F, F),
}

pub trait BoundaryCondition<F: SimpleFloat> {
    /// Arguments are a partition of the solution vector
    fn apply(&self, ctx: Ctx<F>, left: MatMut<F>, center: MatRef<F>, right: MatMut<F>);
}

pub trait InitialCondition<F: SimpleFloat>: Fn(F, MatMut<F>) {}
impl<F: SimpleFloat, T> InitialCondition<F> for T where T: Fn(F, MatMut<F>) {}

#[derive(Clone)]
pub struct Problem<F: SimpleFloat> {
    pub(crate) name: String,
    pub(crate) cl: ConservationLaw<F>,
    pub(crate) domain: Domain<F>,
    pub(crate) bc: Rc<dyn BoundaryCondition<F>>,
    pub(crate) u0: Rc<dyn InitialCondition<F>>,
}

impl<F: SimpleFloat> Problem<F> {
    pub fn new(
        name: impl AsRef<str>,
        cl: ConservationLaw<F>,
        domain: Domain<F>,
        bc: impl BoundaryCondition<F> + 'static,
        u0: impl InitialCondition<F> + 'static,
    ) -> Self {
        Self {
            name: name.as_ref().to_string(),
            cl,
            domain,
            bc: Rc::new(bc),
            u0: Rc::new(u0),
        }
    }
}

impl<F: SimpleFloat> fmt::Debug for Problem<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Problem")
            .field("name", &self.name)
            .field("cl", &self.cl)
            .field("domain", &self.domain)
            .field("bc", &"<function>")
            .field("u0", &"<function>")
            .finish()
    }
}
