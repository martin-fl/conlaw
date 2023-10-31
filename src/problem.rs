use core::fmt;
use std::rc::Rc;

use faer_core::{MatMut, MatRef};
use reborrow::*;

use crate::{Ctx, SimpleFloat};

/// A hyperbolic PDE of the form `u_t + (f(u))_x = 0`
pub trait ConservationLaw<F: SimpleFloat> {
    fn system_size(&self) -> usize;
    fn flux_function(&self, u: MatRef<F>, v: MatMut<F>);

    fn bulk_flux_function(&self, u: MatRef<F>, v: MatMut<F>) {
        for (u, v) in u
            .rb()
            .into_row_chunks(self.system_size())
            .zip(v.into_row_chunks(self.system_size()))
        {
            self.flux_function(u, v)
        }
    }
}

pub mod cl {
    use super::*;
    use faer_core::zipped;
    use std::marker::PhantomData;

    pub struct General<F, G> {
        system_size: usize,
        flux_function: G,
        _marker: PhantomData<F>,
    }

    impl<F: SimpleFloat, G: Fn(MatRef<F>, MatMut<F>)> General<F, G> {
        pub fn new(system_size: usize, flux_function: G) -> Self {
            Self {
                system_size,
                flux_function,
                _marker: PhantomData,
            }
        }
    }

    impl<F: SimpleFloat, G: Fn(MatRef<F>, MatMut<F>)> ConservationLaw<F> for General<F, G> {
        #[inline]
        fn system_size(&self) -> usize {
            self.system_size
        }

        #[inline]
        fn flux_function(&self, u: MatRef<F>, v: MatMut<F>) {
            (self.flux_function)(u, v)
        }
    }

    pub struct Scalar<F, G> {
        flux_function: G,
        _marker: PhantomData<F>,
    }

    impl<F: SimpleFloat, G: Fn(F) -> F> Scalar<F, G> {
        pub fn new(flux_function: G) -> Self {
            Self {
                flux_function,
                _marker: PhantomData,
            }
        }
    }

    impl<F: SimpleFloat, G: Fn(F) -> F> ConservationLaw<F> for Scalar<F, G> {
        #[inline]
        fn system_size(&self) -> usize {
            1
        }

        #[inline]
        fn flux_function(&self, u: MatRef<F>, v: MatMut<F>) {
            zipped!(v, u).for_each(|mut v, u| v.write((self.flux_function)(u.read())));
        }

        #[inline]
        fn bulk_flux_function(&self, u: MatRef<F>, v: MatMut<F>) {
            zipped!(v, u).for_each(|mut v, u| v.write((self.flux_function)(u.read())));
        }
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
pub struct Problem<'pb, F: SimpleFloat> {
    pub(crate) name: String,
    pub(crate) cl: Rc<dyn ConservationLaw<F> + 'pb>,
    pub(crate) domain: Domain<F>,
    pub(crate) bc: Rc<dyn BoundaryCondition<F> + 'pb>,
    pub(crate) u0: Rc<dyn InitialCondition<F> + 'pb>,
}

impl<'pb, F: SimpleFloat> Problem<'pb, F> {
    pub fn new(
        name: impl AsRef<str>,
        cl: impl ConservationLaw<F> + 'pb,
        domain: Domain<F>,
        bc: impl BoundaryCondition<F> + 'pb,
        u0: impl InitialCondition<F> + 'pb,
    ) -> Self {
        Self {
            name: name.as_ref().to_string(),
            cl: Rc::new(cl),
            domain,
            bc: Rc::new(bc),
            u0: Rc::new(u0),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

impl<F: SimpleFloat> fmt::Debug for Problem<'_, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Problem")
            .field("name", &self.name)
            .field("cl", &"<dyn ConservationLaw<_>>")
            .field("domain", &self.domain)
            .field("bc", &"<dyn BoundaryCondition<_>>")
            .field("u0", &"<dyn InitialCondition<_>>")
            .finish()
    }
}
