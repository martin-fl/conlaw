use crate::Float;
use faer::Mat;
use faer_core::{MatMut, MatRef};
use std::io::{self, Write};

pub fn linspace(a: Float, size: usize, h: Float) -> Mat<Float> {
    Mat::<Float>::from_fn(size, 1, |i, _| a + h * i as Float)
}

pub fn apply_func(m: &Mat<Float>, f: impl Fn(Float) -> Float) -> Mat<Float> {
    Mat::from_fn(m.nrows(), m.ncols(), |i, j| f(m[(i, j)]))
}

pub fn write_mat_to_buffer(m: MatRef<'_, Float>, output: &mut impl Write) -> io::Result<()> {
    // SAFETY: faer stores matrix in column-major order with the guarantee that columns
    //         are stored contiguously. We're here taking the first `m.nrows()` elements,
    //         which is the length of 1 column.
    let m_data: &[Float] = unsafe { std::slice::from_raw_parts(m.as_ptr(), m.nrows()) };

    writeln!(
        output,
        "{}",
        m_data
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(",")
    )
}

pub fn broadcast_inplace(f: impl Fn(Float) -> Float, m: MatMut<'_, Float>) {
    faer_core::zipped!(m).for_each(|mut c| c.write(f(c.read())));
}

pub fn broadcast_to(f: impl Fn(Float) -> Float, m: MatRef<'_, Float>, out: MatMut<'_, Float>) {
    faer_core::zipped!(out, m).for_each(|mut out, m| out.write(f(m.read())))
}

pub fn broadcast(f: impl Fn(Float) -> Float, m: MatRef<'_, Float>) -> Mat<Float> {
    let mut out = m.to_owned();
    broadcast_to(f, m, out.as_mut());
    out
}
