pub mod faer_add;

pub mod linear;
pub mod non_linear;

pub use faer_add::Float;

pub mod grid;
use std::io::{self, Write};

use faer_core::{zipped, Mat, MatMut, MatRef};
pub use grid::{DimensionKind, Grid1D, Mesh};

pub trait Method<F: Float> {
    type Meta;

    fn init(domain: &Mesh<F>, meta: Self::Meta) -> Self;

    fn next_to(&mut self, current: MatRef<'_, F>, out: MatMut<'_, F>);

    fn next(&mut self, current: MatRef<'_, F>) -> Mat<F> {
        let mut out = current.to_owned();
        self.next_to(current, out.as_mut());
        out
    }

    fn name(&self) -> &'static str {
        "Unspecified"
    }
}

pub enum SolutionOutput<F: Float, W = io::Sink> {
    Memory(Mat<F>),
    IO(W),
}

impl<F: Float, W: Write> SolutionOutput<F, W> {
    fn save<'a>(&'a mut self, u: MatRef<'a, F>, i: usize) -> io::Result<()> {
        match self {
            SolutionOutput::Memory(m) => {
                assert!(u.ncols() == 1);
                zipped!(m.as_mut().col(i), u.as_ref()).for_each(|mut mi, ui| mi.write(ui.read()));
                Ok(())
            }
            SolutionOutput::IO(output) => faer_add::write_mat_to_buffer(u, output),
        }
    }
}

pub struct Driver<M, F: Float, W = io::Sink> {
    method: M,
    domain: Mesh<F>,
    output: Option<SolutionOutput<F, W>>,
}

impl<F: Float, M: Method<F>, W> Driver<M, F, W> {
    pub fn init(domain: Mesh<F>, meta: M::Meta) -> Self {
        Self {
            method: M::init(&domain, meta),
            domain,
            output: None,
        }
    }

    pub fn save_in_memory(mut self) -> Self {
        self.output = Some(SolutionOutput::Memory(Mat::zeros(
            self.domain.space().steps() + 1,
            self.domain.time().steps() + 1,
        )));

        self
    }

    pub fn save_to<V: Write>(self, out: V) -> Driver<M, F, V> {
        Driver {
            method: self.method,
            domain: self.domain,
            output: Some(SolutionOutput::IO(out)),
        }
    }

    pub fn get_solution(&self) -> Option<MatRef<'_, F>> {
        match &self.output {
            None | Some(SolutionOutput::IO(_)) => None,
            Some(SolutionOutput::Memory(m)) => Some(m.as_ref()),
        }
    }
}

impl<F: Float + std::fmt::Display, M: Method<F>, W> std::fmt::Display for Driver<M, F, W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} method, Δt={} ({} steps), Δx={} ({} steps) ({})",
            self.method.name(),
            self.domain.time().step_size(),
            self.domain.time().steps(),
            self.domain.space().step_size(),
            self.domain.space().steps(),
            match &self.output {
                Some(SolutionOutput::Memory(_)) => "saved in memory",
                Some(SolutionOutput::IO(_)) => "saved to IO",
                None => "not saved",
            }
        )
    }
}

impl<F: Float, M: Method<F>, W: Write> Driver<M, F, W> {
    pub fn solve(&mut self, u0j: MatRef<'_, F>) -> io::Result<()> {
        let mut unj = u0j.to_owned();
        let mut temp = unj.clone();

        if let Some(so) = self.output.as_mut() {
            so.save(unj.as_ref(), 0)?;
        }

        for n in 1..self.domain.time().steps() {
            self.method.next_to(unj.as_ref(), temp.as_mut());
            if let Some(so) = self.output.as_mut() {
                so.save(temp.as_ref(), n)?;
            }
            std::mem::swap(&mut unj, &mut temp);
        }

        Ok(())
    }
}
