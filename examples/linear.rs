use std::io::{self, Write};
use tracing::info;

use conlaw::{faer_add::broadcast, linear, DimensionKind, Driver, Grid1D, Mesh};

fn main() -> io::Result<()> {
    tracing_subscriber::fmt::init();

    info!("setting up problem data");

    // advection coefficient
    let a = 1.0;
    // initial condition
    let u0 = |x: f32| 0.5 * (-100.0 * (x + 0.5).powi(2)).exp() + 0.25;
    // let u0 = |x| if -0.9 < x && x < -0.4 { 0.75 } else { 0.25 };

    let domain = Mesh::new(
        Grid1D::from_step_size(0.0, 1.0, 1e-1),
        Grid1D::from_step_size(-1.0, 1.0, 1e-2),
    )
    .adjust_cfl(DimensionKind::Time, 0.5, a);

    let xj = domain.space().get();

    // initial condition setup
    let u0j = broadcast(u0, xj.as_ref());

    info!("setting up serialization");

    let output = std::fs::File::create("output.csv")?;
    let mut output = std::io::BufWriter::new(output);

    info!("building problem solver");

    let mut solver =
        Driver::<linear::UpwindLeft<f32>, _>::init(domain.clone(), a).save_to(&mut output);

    info!("problem summary: {solver}");
    info!("solving problem");

    solver.solve(u0j.as_ref())?;

    info!("cleaning up");

    output.flush()?;

    info!("done");

    Ok(())
}
