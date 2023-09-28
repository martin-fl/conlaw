pub mod faer_add;

pub mod linear;
pub mod non_linear;

pub type Float = f32;

pub mod grid;
use std::io::{self, Write};

use faer_core::{zipped, Mat, MatMut, MatRef};
pub use grid::{DimensionKind, Domain, Grid};

pub trait Method {
    type Meta;

    fn init(domain: &Domain, meta: Self::Meta) -> Self;

    fn next_to(&mut self, current: MatRef<'_, Float>, out: MatMut<'_, Float>);

    fn next(&mut self, current: MatRef<'_, Float>) -> Mat<Float> {
        let mut out = current.to_owned();
        self.next_to(current, out.as_mut());
        out
    }

    fn name(&self) -> &'static str {
        "Unspecified"
    }
}

pub enum SolutionOutput<W = io::Sink> {
    Memory(Mat<Float>),
    IO(W),
}

impl<W: Write> SolutionOutput<W> {
    fn save<'a>(&'a mut self, u: MatRef<'a, Float>, i: usize) -> io::Result<()> {
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

pub struct Driver<M, W = io::Sink> {
    method: M,
    domain: Domain,
    output: Option<SolutionOutput<W>>,
}

impl<M: Method, W> Driver<M, W> {
    pub fn init(domain: Domain, meta: M::Meta) -> Self {
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

    pub fn save_to<V: Write>(self, out: V) -> Driver<M, V> {
        Driver {
            method: self.method,
            domain: self.domain,
            output: Some(SolutionOutput::IO(out)),
        }
    }

    pub fn get_solution(&self) -> Option<MatRef<'_, Float>> {
        match &self.output {
            None | Some(SolutionOutput::IO(_)) => None,
            Some(SolutionOutput::Memory(m)) => Some(m.as_ref()),
        }
    }
}

impl<M: Method, W> std::fmt::Display for Driver<M, W> {
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

impl<M: Method, W: Write> Driver<M, W> {
    pub fn solve(&mut self, u0j: MatRef<'_, Float>) -> io::Result<()> {
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
