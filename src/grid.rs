use crate::Float;
use faer::Mat;

// grid[0] <-> lower
// grid[i] <-> lower + i * step_size forall i
// grid[steps] <-> upper
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Grid {
    lower: Float,
    upper: Float,
    steps: usize,
    step_size: Float,
}

impl Grid {
    pub fn from_steps(lower: Float, upper: Float, steps: usize) -> Self {
        let step_size = (upper - lower) / steps as Float;
        Grid {
            lower,
            upper,
            steps,
            step_size,
        }
    }

    pub fn from_step_size(lower: Float, upper: Float, step_size: Float) -> Self {
        let steps = (upper - lower) / step_size;
        Self::from_steps(lower, upper, steps.ceil() as usize)
    }

    pub fn set_steps(&mut self, steps: usize) {
        self.steps = steps;
        self.step_size = (self.upper - self.lower) / self.steps as Float;
    }

    pub fn set_step_size(&mut self, step_size: Float) {
        let steps = (self.upper - self.lower) / step_size;
        self.set_steps(steps.ceil() as usize);
    }

    pub fn lower(&self) -> Float {
        self.lower
    }

    pub fn upper(&self) -> Float {
        self.upper
    }

    pub fn steps(&self) -> usize {
        self.steps
    }

    pub fn step_size(&self) -> Float {
        self.step_size
    }

    pub fn get(&self) -> Mat<Float> {
        Mat::from_fn(self.steps + 1, 1, |i, _| {
            self.lower + self.step_size * i as Float
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Domain {
    time: Grid,
    space: Grid,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DimensionKind {
    Time,
    Space,
}

impl Domain {
    pub fn new(time: Grid, space: Grid) -> Self {
        Self { time, space }
    }

    // Given a CFL number `cfl`, adjust the `kind` grid such that
    // `coeff * k / h == cfl` with `k` the time-grid step size and
    // `h` the space-grid step size.
    pub fn adjust_cfl(mut self, kind: DimensionKind, cfl: Float, coeff: Float) -> Self {
        match kind {
            DimensionKind::Time => self.time.set_step_size(cfl * self.space.step_size / coeff),
            DimensionKind::Space => self.space.set_step_size(coeff * self.time.step_size / cfl),
        }

        self
    }

    pub fn time(&self) -> Grid {
        self.time
    }

    pub fn space(&self) -> Grid {
        self.space
    }
}
