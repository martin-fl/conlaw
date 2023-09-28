pub mod faer_add;

pub mod linear;

pub type Float = f32;

pub mod grid;
use std::io::{self, Write};

use faer_core::{Mat, MatMut, MatRef};
pub use grid::{DimensionKind, Domain, Grid};

pub trait Method {
    type Meta;

    fn init(domain: &Domain, meta: Self::Meta) -> Self;

    fn next_to(&self, current: MatRef<'_, Float>, out: MatMut<'_, Float>);

    fn next(&self, current: MatRef<'_, Float>) -> Mat<Float> {
        let mut out = current.to_owned();
        self.next_to(current, out.as_mut());
        out
    }

    fn name(&self) -> &'static str {
        "generic numerical method"
    }
}

pub enum SolutionOutput<W = io::Sink> {
    Memory(Mat<Float>),
    IO(W),
}

impl<W: Write> SolutionOutput<W> {
    fn save<'a>(&'a mut self, u: MatRef<'a, Float>) -> io::Result<()> {
        match self {
            SolutionOutput::Memory(m) => {
                assert!(m.nrows() == u.nrows() && u.ncols() == 1);
                m.resize_with(u.nrows(), m.ncols() + 1, |i, _| u.read(i, 1));
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

impl<M: Method> Driver<M> {
    pub fn init(domain: Domain, meta: M::Meta) -> Self {
        Self {
            method: M::init(&domain, meta),
            domain,
            output: None,
        }
    }

    pub fn save_in_memory(mut self) -> Self {
        self.output = Some(SolutionOutput::Memory(Mat::with_capacity(
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
}

impl<M: Method, W: Write> Driver<M, W> {
    pub fn solve(&mut self, u0j: MatRef<'_, Float>) -> io::Result<()> {
        let mut unj = u0j.to_owned();
        let mut temp = unj.clone();

        for _ in 0..self.domain.time().steps() {
            self.method.next_to(unj.as_ref(), temp.as_mut());
            if let Some(so) = self.output.as_mut() {
                so.save(temp.as_ref())?;
            }
            std::mem::swap(&mut unj, &mut temp);
        }

        Ok(())
    }
}
