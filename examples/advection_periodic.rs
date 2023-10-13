use conlaw::{self, bc, methods, ConservationLaw, Domain, Driver, Problem, Resolution, Simulation};

fn main() {
    let problem = Problem::<f32>::new(
        "advection_periodic",
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

    println!("{sim}");

    Driver::new(sim).run();
}
