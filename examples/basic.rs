use std::io::{self, Write};
use tracing::info;
use tracing_subscriber;

use conlaw::{
    faer_add::{apply_func, linspace, write_mat_to_buffer},
    linear, Float,
};

fn main() -> io::Result<()> {
    tracing_subscriber::fmt::init();

    info!("setting up problem");

    // advection coefficient
    let a = 1 as Float;
    // initial condition
    let u0 = |x: Float| 0.5 * (-100.0 * (x + 0.5).powi(2)).exp() + 0.25;
    // let u0 = |x: Float| if -0.9 < x && x < -0.4 { 0.75 } else { 0.25 };

    // CFL compatible temporal and spatial step sizes
    let ratio = 0.5 / a.abs();
    let x_step = 1e-2 as Float;
    let t_step = ratio * x_step;

    // temporal grid setup
    let t_max = 1.0;
    let t_size = (t_max / t_step) as usize;
    let _tn = linspace(0.0, t_size + 1, t_step);

    // spatial grid setup
    let (x_min, x_max) = (-1 as Float, 1 as Float);
    let x_size = ((x_max - x_min) / x_step) as usize;
    let xj = linspace(x_min, x_size + 1, x_step);

    // initial condition setup
    let u0j = apply_func(&xj, u0);

    let solver = linear::Driver::new(linear::LaxWarming, t_step, x_step, x_size, a);

    info!("setting up serialization");

    let output = std::fs::File::create("output.csv")?;
    let mut output = std::io::BufWriter::new(output);

    info!(
        "solving problem (size = {} bytes (t_size={}, x_size={}), scheme = {})",
        (t_size + 1) * (x_size + 1) * std::mem::size_of::<Float>(),
        t_size,
        x_size,
        solver.name()
    );

    // time marching resolution
    let mut unj = u0j.clone();
    for _ in 0..t_size {
        unj = solver.next(&unj);
        write_mat_to_buffer(&unj, &mut output)?;
    }

    info!("cleaning up");

    output.flush()?;

    info!("done");

    Ok(())
}
