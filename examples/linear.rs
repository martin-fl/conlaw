use std::io;

use conlaw::{
    linear::{self, LinearProblem},
    DimensionKind, Driver, Grid1D, Mesh,
};

struct LinearAdvection;

impl linear::LinearProblem for LinearAdvection {
    type Float = f64;

    fn advection_coefficient() -> Self::Float {
        1.0
    }
}

fn main() -> io::Result<()> {
    let mesh = Mesh::new(
        Grid1D::from_step_size(0.0, 1.0, 1e-1),
        Grid1D::from_step_size(-1.0, 1.0, 1e-2),
    )
    .adjust_cfl(
        DimensionKind::Time,
        0.8,
        LinearAdvection::advection_coefficient(),
    );

    let mut solver = Driver::new(LinearAdvection, mesh, "output.mat")?;

    solver.solve::<linear::LaxWarming<_>>(|x| 0.5 * (-100.0 * (x + 0.5).powi(2)).exp() + 0.25)
}
