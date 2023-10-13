#![allow(clippy::pedantic)]

use faer_core::{MatRef, RealField, SimpleEntity};
use reborrow::*;

mod driver;
mod mesh;
mod method;
mod problem;
mod sim;

pub use driver::*;
pub use method::*;
pub use problem::*;
pub use sim::*;
pub mod bc;
pub mod methods;

pub trait SimpleFloat: RealField + SimpleEntity + Default {}
impl<T> SimpleFloat for T where T: RealField + SimpleEntity + Default {}

/// Information about the state of the simulation
#[derive(Debug, Clone, Copy)]
pub struct Ctx<'ctx, F: SimpleFloat> {
    /// Size of the system (number of conserved quantities), also the size of chunk in the solution vector.
    pub system_size: usize,
    /// Number of left ghost cells
    pub left_ghost_cells: usize,
    /// Number of right ghost cells
    pub right_ghost_cells: usize,

    /// Mesh
    pub mesh: &'ctx mesh::Mesh<F>,
    /// Current time step
    pub n: usize,
    /// Current time value
    pub t: F,
    /// Whole solution, including ghost cells
    u: MatRef<'ctx, F>,
}

impl<F: SimpleFloat> Ctx<'_, F> {
    pub(crate) fn slide(&self, p: isize) -> MatRef<F> {
        self.u.rb().subrows(
            self.left_ghost_cells
                .saturating_add_signed(p * self.system_size as isize),
            self.mesh.space.steps + 1,
        )
    }

    pub fn left(&self) -> MatRef<F> {
        self.slide(-1)
    }

    pub fn left2(&self) -> MatRef<F> {
        self.slide(-2)
    }

    pub fn right(&self) -> MatRef<F> {
        self.slide(1)
    }

    pub fn right2(&self) -> MatRef<F> {
        self.slide(2)
    }
}
