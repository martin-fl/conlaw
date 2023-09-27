use std::io::{self, Write};
use tracing::info;
use tracing_subscriber;

use conlaw::{
    faer_add::{broadcast, write_mat_to_buffer},
    linear::{self, LinearDriver},
    DimensionKind, Domain, Driver, Float, Grid,
};

fn main() -> io::Result<()> {
    tracing_subscriber::fmt::init();

    info!("setting up problem");

    // advection coefficient
    let a = 1 as Float;
    // initial condition
    let u0 = |x: Float| 0.5 * (-100.0 * (x + 0.5).powi(2)).exp() + 0.25;
    // let u0 = |x: Float| if -0.9 < x && x < -0.4 { 0.75 } else { 0.25 };

    let domain = Domain::new(
        Grid::from_step_size(0.0, 1.0, 1e-1),
        Grid::from_step_size(-1.0, 1.0, 1e-2),
    )
    .adjust_cfl(DimensionKind::Time, 0.5, a);
    let xj = domain.space().get();

    // initial condition setup
    let u0j = broadcast(u0, xj.as_ref());

    let solver = LinearDriver::init(linear::LaxWarming, &domain, a);

    info!("setting up serialization");

    let output = std::fs::File::create("output.csv")?;
    let mut output = std::io::BufWriter::new(output);

    info!(
        "solving problem (size = {} bytes ({} time steps, {} spatial steps), scheme = {})",
        (domain.time().steps() + 1) * (domain.space().steps() + 1) * std::mem::size_of::<Float>(),
        domain.time().steps(),
        domain.space().steps(),
        solver.summary()
    );

    // time marching resolution
    let mut unj = u0j.clone();

    let mut temp = unj.clone();
    for _ in 0..domain.time().steps() {
        solver.next_to(unj.as_ref(), temp.as_mut());
        write_mat_to_buffer(temp.as_ref(), &mut output)?;
        std::mem::swap(&mut unj, &mut temp);
    }

    info!("cleaning up");

    output.flush()?;

    info!("done");

    Ok(())
}
