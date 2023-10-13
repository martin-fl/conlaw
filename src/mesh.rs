use crate::SimpleFloat;

#[derive(Debug, Clone, Copy)]
pub struct Grid<F> {
    pub(crate) lower: F,
    pub(crate) upper: F,
    pub(crate) delta: F,
    pub(crate) steps: usize,
}

impl<F: SimpleFloat> Grid<F> {
    pub fn from_steps(lower: F, upper: F, steps: usize) -> Self {
        let delta = upper.sub(lower).div(F::from_f64(steps as f64));
        Self {
            lower,
            upper,
            delta,
            steps,
        }
    }

    pub fn from_delta(lower: F, upper: F, delta: F) -> Self
    where
        F: Into<f64>,
    {
        let steps = upper.sub(lower).div(delta).into().ceil() as usize;
        Self::from_steps(lower, upper, steps)
    }

    pub fn with_delta(self, delta: F) -> Self
    where
        F: Into<f64>,
    {
        Self::from_delta(self.lower, self.upper, delta)
    }

    pub fn with_steps(self, steps: usize) -> Self {
        Self::from_steps(self.lower, self.upper, steps)
    }

    pub fn iter(self) -> impl Iterator<Item = F> {
        (0..(self.steps + 1)).map(move |i| self.lower.add(self.delta.mul(F::from_f64(i as f64))))
    }
}

#[derive(Debug, Clone)]
pub struct Mesh<F> {
    pub(crate) time: Grid<F>,
    pub(crate) space: Grid<F>,
}

impl<F: SimpleFloat> Mesh<F> {
    pub fn new(time: Grid<F>, space: Grid<F>) -> Self {
        Self { time, space }
    }
}
