use faer_core::Mat;
use reborrow::*;

use crate::{method::Method, sim::Simulation, Ctx, SimpleFloat};

pub struct Driver<F: SimpleFloat, M> {
    pub(crate) sim: Simulation<F, M>,
}

impl<F: SimpleFloat, M: Method<F>> Driver<F, M> {
    pub fn new(sim: Simulation<F, M>) -> Self {
        Self { sim }
    }

    pub fn run(&mut self) {
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
                .zip(center.rb_mut().into_row_chunks(problem.cl.m))
            {
                (problem.u0)(x, u)
            }

            v.rb_mut()
                .subrows(left_count, center_count)
                .clone_from(center.rb());

            let ctx = Ctx {
                m: problem.cl.m,
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
        }

        // propagate solution
        for (n, t) in mesh.time.iter().enumerate().skip(1) {
            let ctx = Ctx {
                m: problem.cl.m,
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
            method.apply(ctx, &*problem.cl.flux, u_center.rb(), v_center.rb_mut());
            // apply boundary condition to v
            problem.bc.apply(ctx, v_left, v_center.rb(), v_right);

            // exchange u and v
            std::mem::swap(&mut u, &mut v);
        }
    }
}
