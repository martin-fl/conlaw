use crate::Float;
use faer::Mat;
use std::io::{self, Write};

pub fn linspace(a: Float, size: usize, h: Float) -> Mat<Float> {
    Mat::<Float>::from_fn(size, 1, |i, _| a + h * i as Float)
}

pub fn apply_func(m: &Mat<Float>, f: impl Fn(Float) -> Float) -> Mat<Float> {
    Mat::from_fn(m.nrows(), m.ncols(), |i, j| f(m[(i, j)]))
}

pub fn write_mat_to_buffer(
    m: &Mat<Float>,
    output: &mut io::BufWriter<impl Write>,
) -> io::Result<()> {
    // SAFETY: faer stores matrix in column-major order with the guarantee that columns
    //         are stored contiguously. We're here taking the first `m.nrows()` elements,
    //         which is the length of 1 column.
    let m_data = unsafe { std::slice::from_raw_parts(m.as_ptr(), m.nrows()) };

    writeln!(
        output,
        "{}",
        m_data
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(",")
    )?;

    Ok(())
}
