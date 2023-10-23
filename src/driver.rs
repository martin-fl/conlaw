use std::{io::Write, rc::Rc};

use bytemuck::bytes_of;
use faer_core::{Mat, MatRef};
use reborrow::*;
use thiserror::Error;

use crate::{mesh::Mesh, method::Method, sim::Simulation, Ctx, Problem, Resolution, SimpleFloat};

#[derive(Error, Debug)]
pub enum SimError {
    #[error("output error")]
    Io(#[from] std::io::Error),
}

pub struct ObsCtx<'pb, 'ctx, F: SimpleFloat> {
    // Meta
    problem: &'ctx Problem<'pb, F>,
    mesh: &'ctx Mesh<F>,
    method: &'ctx dyn Method<F>,
    time_sampling: usize,
    space_sampling: usize,

    // Iteration info
    iter: usize,
    time: F,
    solution: MatRef<'ctx, F>, // current solution *without* ghost cells
}

impl<'pb, 'ctx, F: SimpleFloat> ObsCtx<'pb, 'ctx, F> {
    pub fn problem(&self) -> &Problem<'pb, F> {
        self.problem
    }

    pub fn mesh(&self) -> &Mesh<F> {
        self.mesh
    }

    pub fn method(&self) -> &dyn Method<F> {
        self.method
    }

    pub fn iter(&self) -> usize {
        self.iter
    }

    pub fn time(&self) -> F {
        self.time
    }

    pub fn solution(&self) -> MatRef<'_, F> {
        self.solution
    }

    pub fn sampling_period(&self) -> usize {
        self.time_sampling
    }
}

#[allow(unused_variables)]
pub trait Observer<F: SimpleFloat> {
    fn at_startup(&mut self, ctx: ObsCtx<F>) -> Result<(), SimError> {
        Ok(())
    }

    fn at_each_iteration(&mut self, ctx: ObsCtx<F>) -> Result<(), SimError> {
        Ok(())
    }

    fn at_cleanup(&mut self, ctx: ObsCtx<F>) -> Result<(), SimError> {
        Ok(())
    }
}

pub struct Driver<'pb, 'd, F: SimpleFloat, M> {
    pub(crate) sim: Simulation<'pb, F, M>,
    pub(crate) observers: Vec<Box<dyn Observer<F> + 'd>>,
    pub(crate) time_sampling: usize,
    pub(crate) space_sampling: usize,
}

impl<'pb, 'd, F: SimpleFloat, M: Method<F>> Driver<'pb, 'd, F, M> {
    pub fn new(sim: Simulation<'pb, F, M>) -> Self {
        let time_sampling = 1 + sim.mesh.time.steps / 10;
        Self {
            sim,
            observers: Vec::new(),
            time_sampling,
            space_sampling: 1,
        }
    }

    pub fn with_time_sampling(mut self, sampling_period: Resolution<F>) -> Self
    where
        F: Into<f64>,
    {
        self.time_sampling = match sampling_period {
            Resolution::Delta(sampling_period) => {
                sampling_period.div(self.sim.mesh.time.delta).into().ceil() as usize
            }
            Resolution::Steps(sampling_period) => sampling_period,
        };
        self
    }

    pub fn with_space_sampling(mut self, sampling_period: Resolution<F>) -> Self
    where
        F: Into<f64>,
    {
        self.space_sampling = match sampling_period {
            Resolution::Delta(sampling_period) => {
                sampling_period.div(self.sim.mesh.space.delta).into().ceil() as usize
            }
            Resolution::Steps(sampling_period) => sampling_period,
        };
        self
    }

    pub fn with_observer(mut self, observer: impl Observer<F> + 'd) -> Self {
        self.observers.push(Box::new(observer));
        self
    }

    pub fn run(&mut self) -> Result<(), SimError> {
        let Simulation {
            problem,
            mesh,
            method,
        } = &mut self.sim;

        let left_count = method.left_ghost_cells();
        let center_count = mesh.space.steps + 1;
        let right_count = method.right_ghost_cells();

        let mut buffer = Mat::<F>::zeros(left_count + center_count + right_count, 2);

        let [mut u, mut v] = buffer.as_mut().split_at_col(1);

        // set initial condition
        {
            let [left, right] = u.rb_mut().split_at_row(left_count);
            let [mut center, right] = right.split_at_row(center_count);

            for (x, u) in mesh
                .space
                .iter()
                .zip(center.rb_mut().into_row_chunks(problem.cl.system_size()))
            {
                (problem.u0)(x, u)
            }

            v.rb_mut()
                .subrows(left_count, center_count)
                .clone_from(center.rb());

            let ctx = Ctx {
                system_size: problem.cl.system_size(),
                left_ghost_cells: left_count,
                right_ghost_cells: right_count,
                mesh,
                n: 0,
                t: mesh.time.lower,
                // v here because we're mutating u directly
                u: v.rb(),
            };

            problem.bc.apply(ctx, left, center.rb(), right);

            // use this occasion to instantiate the method's buffer
            method.init(ctx);

            for o in self.observers.iter_mut() {
                o.at_startup(ObsCtx {
                    problem,
                    mesh,
                    method,
                    time_sampling: self.time_sampling,
                    space_sampling: self.space_sampling,
                    iter: 0,
                    time: mesh.time.lower,
                    solution: center.as_ref(),
                })?;
            }
        }

        // propagate solution
        for (n, t) in mesh.time.iter().enumerate().skip(1) {
            let ctx = Ctx {
                system_size: problem.cl.system_size(),
                left_ghost_cells: left_count,
                right_ghost_cells: right_count,
                mesh,
                n,
                t,
                // u here because we'll be mutating v
                u: u.rb(),
            };

            let u_center = u.rb().subrows(left_count, center_count);
            let [v_left, v_right] = v.rb_mut().split_at_row(left_count);
            let [mut v_center, v_right] = v_right.split_at_row(center_count);

            // apply numerical method to u into v
            method.apply(
                ctx,
                Rc::clone(&problem.cl),
                u_center.rb(),
                v_center.rb_mut(),
            );
            // apply boundary condition to v
            problem.bc.apply(ctx, v_left, v_center.rb(), v_right);

            if n % self.time_sampling == 0 {
                for o in self.observers.iter_mut() {
                    o.at_each_iteration(ObsCtx {
                        problem,
                        mesh,
                        method,
                        time_sampling: self.time_sampling,
                        space_sampling: self.space_sampling,
                        iter: n,
                        time: t,
                        solution: v_center.as_ref(),
                    })?;
                }
            }

            // exchange u and v
            std::mem::swap(&mut u, &mut v);
        }

        for o in self.observers.iter_mut() {
            o.at_cleanup(ObsCtx {
                problem,
                mesh,
                method,
                time_sampling: self.time_sampling,
                space_sampling: self.space_sampling,
                iter: mesh.space.steps + 1,
                time: mesh.space.upper.add(mesh.space.delta),
                solution: u.rb().subrows(left_count, center_count),
            })?;
        }

        Ok(())
    }
}

pub struct Logger;

impl<F: SimpleFloat + std::fmt::LowerExp> Observer<F> for Logger {
    fn at_startup(&mut self, ctx: ObsCtx<F>) -> Result<(), SimError> {
        tracing::event!(
            tracing::Level::INFO,
            "start of simulation of problem `{}` (`{}` method, Δx={:e} ({} steps), Δt={:e} ({} steps))",
            ctx.problem().name,
            ctx.method().name(),
            ctx.mesh().space.delta,
            ctx.mesh().space.steps,
            ctx.mesh().time.delta,
            ctx.mesh().time.steps,
        );
        Ok(())
    }

    fn at_each_iteration(&mut self, ctx: ObsCtx<F>) -> Result<(), SimError> {
        tracing::event!(
            tracing::Level::TRACE,
            "problem `{}`: step {}",
            ctx.problem().name,
            ctx.iter()
        );
        Ok(())
    }

    fn at_cleanup(&mut self, ctx: ObsCtx<F>) -> Result<(), SimError> {
        tracing::event!(
            tracing::Level::INFO,
            "finished simulation of problem `{}`",
            ctx.problem().name
        );
        Ok(())
    }
}

const CSFF1_HEADER: &[u8] = b"CSFF1";

pub struct Csff1Writer<W> {
    output: W,
}

impl<W: Write> Csff1Writer<W> {
    pub fn new(output: W) -> Self {
        Self { output }
    }
}

impl<F: SimpleFloat, W: Write> Observer<F> for Csff1Writer<W> {
    fn at_startup(&mut self, ctx: ObsCtx<F>) -> Result<(), SimError> {
        let output = &mut self.output;
        // magic bytes
        output.write_all(CSFF1_HEADER)?;
        // write float precision
        output.write_all(bytes_of(&(std::mem::size_of::<F>() as u8)))?;
        // write dimensions
        output.write_all(bytes_of(&(ctx.mesh.space.steps as u32)))?;
        output.write_all(bytes_of(&(ctx.time_sampling as u32)))?;
        output.write_all(bytes_of(&(ctx.space_sampling as u32)))?;
        output.write_all(bytes_of(&(ctx.problem.cl.system_size() as u32)))?;
        output.write_all(bytes_of(&(ctx.mesh.time.steps as u32)))?;
        // write bounds
        output.write_all(bytes_of(&ctx.mesh.space.lower))?;
        output.write_all(bytes_of(&ctx.mesh.space.upper))?;
        output.write_all(bytes_of(&ctx.mesh.time.lower))?;
        output.write_all(bytes_of(&ctx.mesh.time.upper))?;
        // write method name
        let name = ctx.method.name().as_bytes();
        output.write_all(bytes_of(&(name.len() as u32)))?;
        output.write_all(name)?;

        // marker
        output.write_all(&[0xFF, 0xFF, 0xFF, 0xFF])?;

        // write initial condition
        self.at_each_iteration(ctx)
    }

    fn at_each_iteration(&mut self, ctx: ObsCtx<F>) -> Result<(), SimError> {
        let u = ctx.solution;
        if ctx.space_sampling == 1 {
            // SAFETY: faer stores matrix contiguously in column major order
            let u_slice =
                unsafe { std::slice::from_raw_parts(F::from_group(u.rb().as_ptr()), u.nrows()) };

            self.output
                .write_all(bytemuck::cast_slice(u_slice))
                .map_err(SimError::from)
        } else {
            for chunk in u
                .into_row_chunks(ctx.problem.cl.system_size())
                .step_by(ctx.space_sampling)
            {
                // SAFETY: faer stores matrix contiguously in column major order
                let chunk_slice = unsafe {
                    std::slice::from_raw_parts(F::from_group(chunk.rb().as_ptr()), chunk.nrows())
                };
                self.output
                    .write_all(bytemuck::cast_slice(chunk_slice))
                    .map_err(SimError::from)?
            }
            Ok(())
        }
    }

    fn at_cleanup(&mut self, _ctx: ObsCtx<F>) -> Result<(), SimError> {
        self.output.write_all(&[0xFF, 0xFF, 0xFF, 0xFF])?;
        self.output.flush().map_err(SimError::from)
    }
}
