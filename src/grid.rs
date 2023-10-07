use crate::SimpleFloat;
use faer::Mat;

// grid[0] <-> lower
// grid[i] <-> lower + i * step_size forall i
// grid[steps] <-> upper
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Grid1D<F> {
    lower: F,
    upper: F,
    steps: usize,
    step_size: F,
}

impl<F: SimpleFloat> Grid1D<F> {
    pub fn from_steps(lower: F, upper: F, steps: usize) -> Self {
        let step_size = upper.sub(lower).div(F::from_f64(steps as f64));
        Grid1D {
            lower,
            upper,
            steps,
            step_size,
        }
    }

    pub fn from_step_size(lower: F, upper: F, step_size: F) -> Self
    where
        F: Into<f64>,
    {
        let steps = upper.sub(lower).div(step_size);
        Self::from_steps(lower, upper, steps.into().ceil() as usize)
    }

    pub fn set_steps(&mut self, steps: usize) {
        self.steps = steps;
        self.step_size = self
            .upper
            .sub(self.lower)
            .div(F::from_f64(self.steps as f64));
    }

    pub fn set_step_size(&mut self, step_size: F)
    where
        F: Into<f64>,
    {
        let steps = self.upper.sub(self.lower).div(step_size);
        self.set_steps(steps.into().ceil() as usize);
    }

    pub fn lower(&self) -> F {
        self.lower
    }

    pub fn upper(&self) -> F {
        self.upper
    }

    pub fn steps(&self) -> usize {
        self.steps
    }

    pub fn step_size(&self) -> F {
        self.step_size
    }

    pub fn get(&self) -> Mat<F> {
        Mat::from_fn(self.steps + 1, 1, |i, _| {
            self.lower.add(self.step_size.mul(F::from_f64(i as f64)))
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Mesh<F> {
    time: Grid1D<F>,
    space: Grid1D<F>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DimensionKind {
    Time,
    Space,
}

impl<F: SimpleFloat> Mesh<F> {
    pub fn new(time: Grid1D<F>, space: Grid1D<F>) -> Self {
        Self { time, space }
    }

    // Given a CFL number `cfl`, adjust the `kind` grid such that
    // `coeff * k / h == cfl` with `k` the time-grid step size and
    // `h` the space-grid step size.
    pub fn adjust_cfl(mut self, kind: DimensionKind, cfl: F, coeff: F) -> Self
    where
        F: Into<f64>,
    {
        match kind {
            DimensionKind::Time => self.time.set_step_size(cfl.mul(self.dx()).div(coeff)),
            DimensionKind::Space => self.space.set_step_size(coeff.mul(self.dt()).div(cfl)),
        }

        self
    }

    pub fn time(&self) -> Grid1D<F> {
        self.time
    }

    pub fn space(&self) -> Grid1D<F> {
        self.space
    }

    pub fn dt(&self) -> F {
        self.time.step_size
    }

    pub fn dx(&self) -> F {
        self.space.step_size
    }

    pub fn nt(&self) -> usize {
        self.time.steps
    }

    pub fn nx(&self) -> usize {
        self.space.steps
    }
}
