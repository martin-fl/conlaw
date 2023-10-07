use std::{
    fs,
    io::{self, Write},
    path::Path,
};

use bytemuck::bytes_of;
use faer_core::{zipped, Mat, MatMut, MatRef, RealField, SimpleEntity};
use reborrow::*;

pub mod grid;
pub mod linear;
pub mod non_linear;

pub use grid::{DimensionKind, Grid1D, Mesh};

pub trait SimpleFloat: RealField + SimpleEntity + Copy {}
impl<T> SimpleFloat for T where T: RealField + SimpleEntity + Copy {}

pub trait Problem {
    type Float: SimpleFloat;

    fn flux(u: Self::Float) -> Self::Float;
}

pub trait Method<P: Problem> {
    fn init(mesh: &Mesh<P::Float>) -> Self;

    fn next_to(&mut self, current: MatRef<'_, P::Float>, out: MatMut<'_, P::Float>);

    fn next(&mut self, current: MatRef<'_, P::Float>) -> Mat<P::Float> {
        let mut out = Mat::zeros(current.nrows(), current.ncols());
        self.next_to(current, out.as_mut());
        out
    }

    fn name() -> &'static str {
        "Unspecified"
    }
}

const SOLUTION_FILE_FORMAT_HEADER: &'static [u8] = b"CSFF1";

pub struct Driver<P, F> {
    #[allow(dead_code)]
    problem: P,
    mesh: Mesh<F>,
    output: io::BufWriter<fs::File>,
}

impl<P, F> Driver<P, F>
where
    F: SimpleFloat,
    P: Problem<Float = F>,
{
    pub fn new(problem: P, mesh: Mesh<F>, output: impl AsRef<Path>) -> io::Result<Self> {
        Ok(Self {
            problem,
            mesh,
            output: io::BufWriter::new(fs::File::create(output)?),
        })
    }

    pub fn solve<M: Method<P>>(&mut self, u0: impl Fn(P::Float) -> P::Float) -> io::Result<()> {
        let mut buffer = Mat::zeros(self.mesh.space().steps() + 1, 2);
        let [mut u, mut v] = buffer.as_mut().split_at_col(1);

        // magic bytes
        self.output.write_all(SOLUTION_FILE_FORMAT_HEADER)?;
        // write float precision
        self.output
            .write_all(bytes_of(&(std::mem::size_of::<F>() as u8)))?;
        // write dimensions
        self.output.write_all(bytes_of(&(self.mesh.nx() as u32)))?;
        self.output.write_all(bytes_of(&1u32))?;
        self.output.write_all(bytes_of(&(self.mesh.nt() as u32)))?;
        // write bounds
        self.output
            .write_all(bytes_of(&self.mesh.space().lower()))?;
        self.output
            .write_all(bytes_of(&self.mesh.space().upper()))?;
        self.output.write_all(bytes_of(&self.mesh.time().lower()))?;
        self.output.write_all(bytes_of(&self.mesh.time().upper()))?;
        // write method name
        let name = M::name().as_bytes();
        self.output.write_all(bytes_of(&(name.len() as u32)))?;
        self.output.write_all(name)?;

        // set initial condition
        zipped!(u.rb_mut(), self.mesh.space().get().as_ref())
            .for_each(|mut u, x| u.write(u0(x.read())));

        // beginning marker
        self.output.write_all(&[0xFF, 0xFF, 0xFF, 0xFF])?;

        let mut col_buffer = Vec::<u8>::with_capacity(
            (self.mesh.space().steps() + 1) * 1 * std::mem::size_of::<F>(),
        );

        let mut save = |u: MatRef<'_, F>| -> io::Result<()> {
            let u_slice =
                unsafe { std::slice::from_raw_parts(F::from_group(u.rb().as_ptr()), u.nrows()) };
            col_buffer.extend(u_slice.iter().map(|x| bytes_of(x)).flatten());
            self.output.write_all(&col_buffer)?;
            col_buffer.truncate(0);
            Ok(())
        };

        save(u.rb())?;

        // setup method
        let mut method = M::init(&self.mesh);

        for _ in 0..self.mesh.time().steps() {
            // propagate u into v
            method.next_to(u.rb(), v.rb_mut());
            std::mem::swap(&mut u, &mut v);

            save(u.rb())?;
        }

        // end marker
        self.output.write_all(&[0xFF, 0xFF, 0xFF, 0xFF])?;

        self.output.flush()
    }
}
