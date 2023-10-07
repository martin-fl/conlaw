use faer::Mat;
use faer_core::{ComplexField, Entity, MatMut, MatRef, RealField, SimpleEntity};
use std::io::{self, Write};

pub trait SimpleFloat: RealField + SimpleEntity + Copy {}
impl<T> SimpleFloat for T where T: RealField + SimpleEntity + Copy {}

pub fn write_mat_to_buffer<F: SimpleEntity + std::fmt::Debug>(
    m: MatRef<'_, F>,
    output: &mut impl Write,
) -> io::Result<()> {
    // SAFETY: faer stores matrix in column-major order with the guarantee that columns
    //         are stored contiguously. We're here taking the first `m.nrows()` elements,
    //         which is the length of 1 column.
    let m_data = unsafe { std::slice::from_raw_parts(F::from_group(m.as_ptr()), m.nrows()) };

    writeln!(
        output,
        "{}",
        m_data
            .iter()
            .map(|x| format!("{:?}", x))
            .collect::<Vec<_>>()
            .join(",")
    )
}

pub fn broadcast_inplace<F: Entity>(f: impl Fn(F) -> F, m: MatMut<'_, F>) {
    faer_core::zipped!(m).for_each(|mut c| c.write(f(c.read())));
}

pub fn broadcast_to<F: Entity>(f: impl Fn(F) -> F, m: MatRef<'_, F>, out: MatMut<'_, F>) {
    faer_core::zipped!(out, m).for_each(|mut out, m| out.write(f(m.read())))
}

pub fn broadcast<F: ComplexField>(f: impl Fn(F) -> F, m: MatRef<'_, F>) -> Mat<F> {
    let mut out = Mat::zeros(m.nrows(), m.ncols());
    broadcast_to(f, m, out.as_mut());
    out
}
