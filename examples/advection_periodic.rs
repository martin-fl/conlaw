use conlaw::{self, bc, cl, methods, Domain, Driver, Problem, Resolution, Simulation};
use std::{fs, io};

fn main() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .init();

    let problem_name = "advection_periodic";

    let problem = Problem::<f32>::new(
        problem_name,
        cl::General::new(1, |u, mut v| v.clone_from(u)),
        Domain {
            time: (0., 2.),
            space: (-1., 1.),
        },
        bc::Periodic,
        |x, mut v| v[(0, 0)] = if x < 0. { 0. } else { 1. },
    );

    let sim = Simulation::new(problem)
        .with_method::<methods::UpwindLeft<_>>()
        .with_time_resolution(Resolution::Delta(0.0005))
        .with_space_resolution(Resolution::Delta(0.001));

    let mut output = io::BufWriter::new(
        fs::File::create(format!("bin/{}.csff1", problem_name))
            .expect("couldn't create output file"),
    );

    Driver::new(sim)
        .with_observer(conlaw::Logger)
        .with_observer(conlaw::Csff1Writer::new(&mut output))
        .with_time_sampling(Resolution::Steps(10))
        .with_space_sampling(Resolution::Steps(10))
        .run()
        .expect("failed to run simulation");
}
