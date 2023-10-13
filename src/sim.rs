use std::fmt;

use crate::{
    mesh::{Grid, Mesh},
    method::Method,
    methods,
    problem::Problem,
    SimpleFloat,
};

#[derive(Debug, Clone)]
pub enum Resolution<F> {
    Delta(F),
    Steps(usize),
}

#[derive(Debug, Clone)]
pub struct Simulation<F: SimpleFloat, M> {
    pub(crate) problem: Problem<F>,
    pub(crate) mesh: Mesh<F>,
    pub(crate) method: M,
}

impl<F: SimpleFloat> Simulation<F, methods::MacCormack<F>> {
    pub fn new(problem: Problem<F>) -> Self {
        let mesh = Mesh::new(
            Grid::from_steps(problem.domain.time.0, problem.domain.time.1, 100),
            Grid::from_steps(problem.domain.space.0, problem.domain.space.1, 100),
        );

        Self {
            problem,
            mesh,
            method: methods::MacCormack::default(),
        }
    }
}

impl<F: SimpleFloat, M: Method<F>> Simulation<F, M> {
    pub fn with_time_resolution(mut self, r: Resolution<F>) -> Self
    where
        F: Into<f64>,
    {
        self.mesh.time = match r {
            Resolution::Delta(delta) => self.mesh.time.with_delta(delta),
            Resolution::Steps(steps) => self.mesh.time.with_steps(steps),
        };

        self
    }

    pub fn with_space_resolution(mut self, r: Resolution<F>) -> Self
    where
        F: Into<f64>,
    {
        self.mesh.space = match r {
            Resolution::Delta(delta) => self.mesh.space.with_delta(delta),
            Resolution::Steps(steps) => self.mesh.space.with_steps(steps),
        };

        self
    }

    pub fn with_method<N: Method<F> + Default>(self) -> Simulation<F, N> {
        Simulation {
            problem: self.problem,
            mesh: self.mesh,
            method: N::default(),
        }
    }
}

impl<F: SimpleFloat + fmt::LowerExp, M: Method<F>> fmt::Display for Simulation<F, M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "simulation of `{}` problem:\n\t- `{}` method\n\t- Δx = {:e} ({} steps)\n\t- Δt = {:e} ({} steps)",
            self.problem.name,
            self.method.name(),
            self.mesh.space.delta,
            self.mesh.space.steps,
            self.mesh.time.delta,
            self.mesh.time.steps
        )
    }
}
