# conlaw

Numerical solver for conservation laws of the form 
$$
\frac{\partial u}{\partial t} + \frac{\partial f \circ u}{\partial x} = 0
$$
where $u:(x,t)\in[x_\min,x_\max]\times[t_\min,t_\max] \mapsto u(x,t)\in\mathbb{R}^m$,
$m \geq 1$ and $f:\mathbb{R}^m\to\mathbb{R}^m$, subject to an initial condition
$u(0,t)=u_0(t)$ and boundary condition(s).