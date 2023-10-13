# conlaw

Numerical solver for conservation laws of the form 
```math
𝜕u/𝜕t + 𝜕(f(u))/𝜕x = 0
```
where `u` is a vector-value function of the real variables `t` and `x`, and `f`
is a vector valued function with domain equal to the range of `u`.