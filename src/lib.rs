pub mod faer_add;

pub mod linear;

pub type Float = f32;

pub mod grid;
use faer_core::{Mat, MatMut, MatRef};
pub use grid::{DimensionKind, Domain, Grid};

pub trait Driver {
    type Schema;

    /// Info required for the driver to run, e.g.
    /// - for linear equations u_t + Au_x = 0, it would be the matrix A,
    /// - for non-linear equations u_t + (f(u))_x = 0, it would either
    ///   be the flux function f or the flux function and its jacobian (f, f').
    type Meta;

    fn init(method: Self::Schema, domain: &Domain, meta: Self::Meta) -> Self;

    fn next_to(&self, unj: MatRef<'_, Float>, out: MatMut<'_, Float>);

    fn next(&self, unj: MatRef<'_, Float>) -> Mat<Float> {
        let mut out = unj.to_owned();
        self.next_to(unj, out.as_mut());
        out
    }

    fn summary(&self) -> &'static str {
        "Driver"
    }
}
