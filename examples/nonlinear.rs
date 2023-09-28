use std::{
    io::{self, Write},
    rc::Rc,
};
use tracing::info;
use tracing_subscriber;

use conlaw::{faer_add::broadcast, non_linear, DimensionKind, Domain, Driver, Float, Grid};

fn main() -> io::Result<()> {
    tracing_subscriber::fmt::init();

    info!("setting up problem data");

    // flux
    let flux = |u: Float| u;
    // initial condition
    let u0 = |x: Float| 0.5 * (-100.0 * (x + 0.5).powi(2)).exp() + 0.25;
    // let u0 = |x: Float| if -0.9 < x && x < -0.4 { 0.75 } else { 0.25 };

    let domain = Domain::new(
        Grid::from_step_size(0.0, 1.0, 1e-1),
        Grid::from_step_size(-1.0, 1.0, 1e-2),
    )
    .adjust_cfl(DimensionKind::Time, 0.5, 1.0);

    let xj = domain.space().get();

    // initial condition setup
    let u0j = broadcast(u0, xj.as_ref());

    info!("setting up serialization");

    let output = std::fs::File::create("output.csv")?;
    let mut output = std::io::BufWriter::new(output);

    info!("building problem solver");

    let mut solver =
        Driver::<non_linear::MacCormack>::init(domain, Rc::new(flux)).save_to(&mut output);

    info!("problem summary: {solver}");
    info!("solving problem");

    solver.solve(u0j.as_ref())?;

    info!("cleaning up");

    output.flush()?;

    info!("done");

    Ok(())
}
