use std::io;

use conlaw::{non_linear, DimensionKind, Driver, Grid1D, Mesh};

struct LinearAdvection;

impl conlaw::Problem for LinearAdvection {
    type Float = f64;

    fn flux(u: Self::Float) -> Self::Float {
        u
    }
}

fn main() -> io::Result<()> {
    let mesh = Mesh::new(
        Grid1D::from_step_size(0.0, 1.0, 1e-1),
        Grid1D::from_step_size(-1.0, 1.0, 1e-2),
    )
    .adjust_cfl(DimensionKind::Time, 0.5, 1.0);

    let mut solver = Driver::new(LinearAdvection, mesh, "output.mat")?;

    solver.solve::<non_linear::MacCormack<_>>(|x| 0.5 * (-100.0 * (x + 0.5).powi(2)).exp() + 0.25)
}
