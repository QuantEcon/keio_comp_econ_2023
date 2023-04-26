---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---


# Accelerating Python

[John Stachurski](http://johnstachurski.net)

This notebook demonstrates ways of accelerating plain Python code in
scientific applications.

We begin by importing some libraries that will be discussed below.


```{code-cell} ipython3
import numpy as np
from numpy.random import randn
import numba 
from numba import vectorize, float64
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
```



## Problem 1: A Time Series Model

Consider the time series model

$$ x_{t+1} = \alpha x_t (1 - x_t) $$


Our aim is to generate time series from this model and analyze them.

We will show how to accelerate this operation.

To begin, let's set $\alpha = 4$


```{code-cell} ipython3
α = 4
```

Here's a typical time series:

```{code-cell} ipython3
n = 200
x =  np.empty(n)
x[0] = 0.2
for t in range(n-1):
    x[t+1] = α * x[t] * (1 - x[t])
    
plt.plot(x)
plt.show()
```

### Python Test

+++

Here's a function that iterates forward `n` times, starting from `x0`, and
returns **the final** value:

```{code-cell} ipython3
def quad(x0, n):
    x = x0
    for i in range(n):
        x = α * x * (1 - x)
    return x
```

Let's see how fast this runs:

```{code-cell} ipython3
n = 10_000_000
```

```{code-cell} ipython3
%%time
x = quad(0.2, n)
```

### Fortran Test

+++

Now let's try this in Fortran.

Note --- this step is intended to be a demo and will only execute if

* you have the file `fastquad.f90` in your pwd
* you have a Fortran compiler installed and modify the compilation code below appropriately

```{code-cell} ipython3
%%file fortran_quad.f90

PURE FUNCTION QUAD(X0, N)
 IMPLICIT NONE
 INTEGER, PARAMETER :: DP=KIND(0.d0)                           
 REAL(dp), INTENT(IN) :: X0
 REAL(dp) :: QUAD
 INTEGER :: I
 INTEGER, INTENT(IN) :: N
 QUAD = X0
 DO I = 1, N - 1                                                
  QUAD = 4.0_dp * QUAD * real(1.0_dp - QUAD, dp)
 END DO
 RETURN
END FUNCTION QUAD

PROGRAM MAIN
 IMPLICIT NONE
 INTEGER, PARAMETER :: DP=KIND(0.d0)                          
 REAL(dp) :: START, FINISH, X, QUAD
 INTEGER :: N
 N = 10000000
 X = QUAD(0.2_dp, 10)
 CALL CPU_TIME(START)
 X = QUAD(0.2_dp, N)
 CALL CPU_TIME(FINISH)
 PRINT *,'last val = ', X
 PRINT *,'Elapsed time in milliseconds = ', (FINISH-START) * 1000
END PROGRAM MAIN
```

```{code-cell} ipython3
!gfortran -O3 fortran_quad.f90
```

```{code-cell} ipython3
!./a.out
```

```{code-cell} ipython3
!rm a.out
```

### Codon

Let's try `codon`, an AOT Python compiler

First we install it --- if not yet installed, please uncomment

```{code-cell} ipython3
# !/bin/bash -c "$(curl -fsSL https://exaloop.io/install.sh)"
```

Now we write Python code to a file.


```{code-cell} ipython3
%%file codon_quad.py

from time import time

n = 10_000_000
alpha = 4.0

def quad(x0, n):
    x = x0
    for i in range(1, n):
        x = alpha * x * (1 - x)
    return x


t0 = time()
x = quad(0.1, n)
t1 = time()
print(x)
print("Elapsed time in milliseconds: ", (t1 - t0) * 1000)
```

Next we compile the Python code to build an executable.

```{code-cell} ipython3
!codon build --release --exe codon_quad.py
```

Now let's run it.

```{code-cell} ipython3
!./codon_quad
```

Tidying up:

```{code-cell} ipython3
!rm codon_quad
```

```{code-cell} ipython3

```


### Python + Numba

+++

Now let's replicate the calculations using Numba's JIT compiler.

Here's the Python function we want to speed up


```{code-cell} ipython3
@numba.jit
def quad(x0, n):
    x = x0
    for i in range(1, n):
        x = α * x * (1 - x)
    return x
```

This is the same as before except that we've targeted the function for JIT
compilation with `@numba.jit`.

Let's see how fast it runs.

```{code-cell} ipython3
%%time
x = quad(0.2, n)
```

```{code-cell} ipython3
%%time
x = quad(0.2, n)
```



## Problem 2: Multivariate Optimization

The problem is to maximize the function 

$$ f(x, y) = \frac{\cos \left(x^2 + y^2 \right)}{1 + x^2 + y^2} + 1$$

using brute force --- searching over a grid of $(x, y)$ pairs.

```{code-cell} ipython3
def f(x, y):
    return np.cos(x**2 + y**2) / (1 + x**2 + y**2) + 1
```

```{code-cell} ipython3
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm

gridsize = 50
gmin, gmax = -3, 3
xgrid = np.linspace(gmin, gmax, gridsize)
ygrid = xgrid
x, y = np.meshgrid(xgrid, ygrid)

# === plot value function === #
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x,
                y,
                f(x, y),
                rstride=2, cstride=2,
                cmap=cm.jet,
                alpha=0.4,
                linewidth=0.05)


ax.scatter(x, y, c='k', s=0.6)

ax.scatter(x, y, f(x, y), c='k', s=0.6)

ax.view_init(25, -57)
ax.set_zlim(-0, 2.0)
ax.set_xlim(gmin, gmax)
ax.set_ylim(gmin, gmax)

plt.show()
```

Let's try a few different methods to make it fast.



### Vectorized Numpy 

```{code-cell} ipython3
grid = np.linspace(-3, 3, 10000)

x, y = np.meshgrid(grid, grid)
```

```{code-cell} ipython3
%%time

np.max(f(x, y))
```

### JITTed code


A jitted version

```{code-cell} ipython3
@numba.jit
def compute_max():
    m = -np.inf
    for x in grid:
        for y in grid:
            z = np.cos(x**2 + y**2) / (1 + x**2 + y**2) + 1
            if z > m:
                m = z
    return m
```

```{code-cell} ipython3
compute_max()
```

```{code-cell} ipython3
%%time
compute_max()
```

### Vectorized Numba on the CPU


Numba for vectorization with automatic parallelization;

```{code-cell} ipython3
@vectorize('float64(float64, float64)', target='parallel')
def f_par(x, y):
    return np.cos(x**2 + y**2) / (1 + x**2 + y**2) + 1
```

```{code-cell} ipython3
x, y = np.meshgrid(grid, grid)

np.max(f_par(x, y))
```

```{code-cell} ipython3
%%time
np.max(f_par(x, y))
```



### JAX on the GPU

Now let's try JAX.

This code will work well if you have a GPU and JAX configured to use it.

Let's see what we have available.

```{code-cell} ipython3
!nvidia-smi
```


Warning --- you need a GPU with relatively large memory for this to work.


```{code-cell} ipython3
def f(x, y):
    return jnp.cos(x**2 + y**2) / (1 + x**2 + y**2) + 1
```


```{code-cell} ipython3
grid = np.linspace(-3, 3, 10000)

x, y = jnp.meshgrid(grid, grid)
```

Here's our timing.

```{code-cell} ipython3
%%time

jnp.max(f(x, y))
```


```{code-cell} ipython3
@jax.jit
def f(x, y):
    return jnp.cos(x**2 + y**2) / (1 + x**2 + y**2) + 1
```

Let's JIT-compile the function and see if anything changes.

```{code-cell} ipython3
%%time

jnp.max(f(x, y))
```


```{code-cell} ipython3
%%time

jnp.max(f(x, y))
```


## Problem 3: Monte Carlo


In this section we describe the Monte Carlo method of integration via a simple
example.

### Share Price with Known Distribution

Let's suppose that we are considering buying a share (or many shares) in a
given company.

Our plan is either to 

* buy it now, hold it for one year and then sell it, or
* do something else with our money.

We start by thinking of the share price in one year as a random variable $S$.

(Let's forget about dividends for now, so that our return on holding the share
is the relative change in its price.)

To decide whether or not to go ahead, we need to know some features of the
distribution of $S$.

For example, we might decide to buy if the mean is high and the variance is
low.

(High expected returns and low risk.)

Suppose that, after analyzing the data, we have decided that $S$ is well
represented by a lognormal distribution with parameters $\mu, \sigma$ .

* $S$ has the same distribution as $\exp(\mu + \sigma Z)$ where $Z$ is standard normal.
* we write this statement as $S \sim LN(\mu, \sigma)$.

Any good reference on statistics will tell us that the mean and variance are

$$
    \mathbb E S 
        = \exp \left(\mu + \frac{\sigma^2}{2} \right)
$$

and 

$$ 
    \mathop{\mathrm{Var}} S 
    = [\exp(\sigma^2) - 1] \exp(2\mu + \sigma^2)
$$

So far we have no need for a computer.


### Share Price with Unknown Distribution

But now suppose that we study the distribution of $S$ more carefully, leading
us to decompose the price into multiple factors.

In particular, we conclude that the share price depends on three variables,
with

$$
    S = (X_1 + X_2 + X_3)^p
$$

We assume that

* $p$ is a positive number, which is known to us,
* $X_i \sim LN(\mu_i, \sigma_i)$ for $i=1,2,3$,
* the values of $\mu_i, \sigma_i$ have all been estimated, and
* the random variables $X_1$, $X_2$ and $X_3$ are independent.

How should we compute the mean of $S$?

To do this with pencil and paper is hard (unless, say, $p=1$).

But fortunately there's an easy way to do this, at least approximately:

1. Generate $n$ independent draws of $X_1$, $X_2$ and $X_3$ on a computer,
1. Use these draws to generate $n$ independent draws of $S$, and
1. Take the average value of these draws of $S$.

By the law of large numbers, this average will be close to the true mean when
$n$ is large.

We use the following values for $p$ and each $\mu_i$ and $\sigma_i$.

```{code-cell} ipython3
n = 10_000_000
p = 0.5
μ_1, μ_2, μ_3 = 0.2, 0.8, 0.4
σ_1, σ_2, σ_3 = 0.1, 0.05, 0.2
```

### A Routine using Loops in Python

+++

Here's a routine using native Python loops to calculate the desired mean

$$
    \frac{1}{n} \sum_{i=1}^n S_i
    \approx \mathbb E S
$$


```{code-cell} ipython3
def compute_mean(n=10_000_000):
    S = 0.0
    for i in range(n):
        X_1 = np.exp(μ_1 + σ_1 * randn())
        X_2 = np.exp(μ_2 + σ_2 * randn())
        X_3 = np.exp(μ_3 + σ_3 * randn())
        S += (X_1 + X_2 + X_3)**p
    return(S / n)
```

Let's test it and see how long it takes.

```{code-cell} ipython3
%%time

compute_mean()
```

### Using Numba's JIT Compiler

```{code-cell} ipython3
compute_mean_numba = numba.jit(compute_mean)
```

```{code-cell} ipython3
%%time

compute_mean_numba()
```

```{code-cell} ipython3
%%time

compute_mean_numba()
```

### A Vectorized Routine

+++

Now we implement a vectorized routine using traditional NumPy array processing.

```{code-cell} ipython3

def compute_mean_vectorized(n=10_000_000):
    X_1 = np.exp(μ_1 + σ_1 * randn(n))
    X_2 = np.exp(μ_2 + σ_2 * randn(n))
    X_3 = np.exp(μ_3 + σ_3 * randn(n))
    S = (X_1 + X_2 + X_3)**p
    return(S.mean())
```

```{code-cell} ipython3
%%time

compute_mean_vectorized()
```

### Using Google JAX


Finally, let's try to shift this to the GPU and parallelize it effectively.


```{code-cell} ipython3
!nvidia-smi
```

```{code-cell} ipython3
:tags: []

def compute_mean_jax(n=10_000_000):
    key = jax.random.PRNGKey(1)
    Z = jax.random.normal(key, (3, n))
    X_1 = jnp.exp(μ_1 + σ_1 * Z[0,:])
    X_2 = jnp.exp(μ_2 + σ_2 * Z[1,:])
    X_3 = jnp.exp(μ_3 + σ_3 * Z[2,:])
    S = (X_1 + X_2 + X_3)**p
    return(S.mean())
```

```{code-cell} ipython3
%%time

compute_mean_jax()
```

```{code-cell} ipython3
compute_mean_jax_jitted = jax.jit(compute_mean_jax)
```

```{code-cell} ipython3
%%time

compute_mean_jax_jitted()
```

```{code-cell} ipython3
%%time

compute_mean_jax_jitted()
```
