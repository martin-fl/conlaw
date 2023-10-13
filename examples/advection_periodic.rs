use conlaw::{self, bc, methods, ConservationLaw, Domain, Driver, Problem, Resolution, Simulation};
use std::{fs, io};

fn main() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::TRACE)
        .init();

    let problem_name = "advection_periodic";

    let problem = Problem::<f32>::new(
        problem_name,
        ConservationLaw::new(1, |u, mut v| v.clone_from(u)),
        Domain {
            time: (0., 1.),
            space: (-1., 1.),
        },
        bc::Periodic,
        |x, mut v| v[(0, 0)] = if x < 0. { 0. } else { 1. },
    );

    let sim = Simulation::new(problem)
        .with_time_resolution(Resolution::Delta(0.0005))
        .with_space_resolution(Resolution::Delta(0.001))
        .with_method::<methods::UpwindLeft<_>>();

    let mut output = io::BufWriter::new(
        fs::File::create(format!("bin/{}.csff1", problem_name))
            .expect("couldn't create output file"),
    );

    Driver::new(sim)
        .with_sampling_period(Resolution::Delta(0.005))
        .with_observer(conlaw::Logger)
        .with_observer(conlaw::Csff1Writer::new(&mut output))
        .run()
        .expect("failed to run simulation");
}
