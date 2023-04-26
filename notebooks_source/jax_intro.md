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

# An Introduction to JAX

#### John Stachurski

+++

[JAX](https://github.com/google/jax) is scientific library within the Python ecosystem that provides data types, functions and a compiler for fast linear algebra operations and automatic differentiation.

Loosely speaking, JAX is like NumPy with the addition of

* automatic differentiation
* automated GPU/TPU support
* a just-in-time compiler

JAX is often used for machine learning and AI, since it can scale to big data operations on GPUs and automatically differentiate loss functions for gradient decent.

However, JAX is sufficiently low-level that it can be used for many purposes.

Here is a short history of JAX:

* 2015: Google open-sources part of its AI infrastructure called TensorFlow.
* 2016: The popularity of TensorFlow grows rapidly.
* 2017: Facebook open-sources PyTorch beta, an alternative AI framework (developer-friendly, more Pythonic)
* 2018: Facebook launches a full production-ready version of PyTorch.
* 2019: PyTorch surges in popularity (adopted by Uber, Airbnb, Tesla, etc.)
* 2020: Google launches JAX as an open-source framework.
* 2021: Google starts to shift away from TPUs to Nvidia GPUs, extends JAX capabilities.
* 2022: Uptake of Google JAX accelerates rapidly

+++

We begin this notebook with some standard imports

```{code-cell} ipython3
import numpy as np
import matplotlib as plt
from numba import jit, njit, float64, vectorize
```


## Installation

JAX can be installed with or without GPU support.

* Follow [the install guide](https://github.com/google/jax)

Note that JAX is pre-installed with GPU support on [Google Colab](https://colab.research.google.com/).

(Colab Pro offers better GPUs.)

```{code-cell} ipython3
!nvidia-smi
```

+++

## JAX as a NumPy Replacement

+++

One way to use JAX is as a plug-in NumPy replacement.  Let's look at the similarities and differences.

### Similarities

+++

The following import is standard, replacing `import numpy as np`:

```{code-cell} ipython3
import jax
import jax.numpy as jnp
```

Now we can use `jnp` in place of `np` for the usual array operations:

```{code-cell} ipython3
a = jnp.asarray((1.0, 3.2, -1.5))
```

```{code-cell} ipython3
print(a)
```

```{code-cell} ipython3
print(jnp.sum(a))
```

```{code-cell} ipython3
print(jnp.mean(a))
```

```{code-cell} ipython3
print(jnp.dot(a, a))
```

However, the array object `a` is not a NumPy array:

```{code-cell} ipython3
a
```

```{code-cell} ipython3
type(a)
```

Even scalar-valued maps on arrays return JAX arrays:

```{code-cell} ipython3
jnp.sum(a)
```

JAX arrays are allocated on the `device`.

Here `device` refers to the hardware accelerator (GPU or TPU), although JAX falls back to the CPU if no accelerator is detected.

(In the terminology of GPUs, the "host" is the machine that launches GPU operations, while the "device" is the GPU itself.)


+++

Operations on higher dimensional arrays is also similar to NumPy:

```{code-cell} ipython3
A = jnp.ones((2, 2))
B = jnp.identity(2)
A @ B
```

```{code-cell} ipython3
from jax.numpy import linalg
```

```{code-cell} ipython3
linalg.solve(B, A)
```

```{code-cell} ipython3
linalg.eigh(B)  # Computes eigenvalues and eigenvectors
```

### Differences

+++

One difference between NumPy and JAX is that, when running on a GPU, JAX uses 32 bit floats by default.  This is standard for GPU computing and can lead to significant speed gains with small loss of precision.

However, for some calculations precision matters.  In these cases 64 bit floats can be enforced via the command

```{code-cell} ipython3
jax.config.update("jax_enable_x64", True)
```

Let's check this works:

```{code-cell} ipython3
jnp.ones(3)
```

As a NumPy replacement, a more significant difference is that arrays are treated as **immutable**.  For example, with NumPy we can write

```{code-cell} ipython3
import numpy as np
a = np.linspace(0, 1, 3)
a
```

and then mutate the data in memory:

```{code-cell} ipython3
a[0] = 1
a
```

In JAX this fails:

```{code-cell} ipython3
a = jnp.linspace(0, 1, 3)
a
```

```{code-cell} ipython3
a[0] = 1
```

In line with immutability, JAX does not support inplace operations:

```{code-cell} ipython3
a = np.array((2, 1))
a.sort()
a
```

```{code-cell} ipython3
a = jnp.array((2, 1))
a.sort()
a
```

The designers of JAX chose to make arrays immutable because JAX uses a functional programming style.  More on this below.  

Note that, while mutation is discouraged, it is in fact possible with `at`, as in

```{code-cell} ipython3
a = jnp.linspace(0, 1, 3)
id(a)
```

```{code-cell} ipython3
a
```

```{code-cell} ipython3
a.at[0].set(1)
```

We can check that the array is mutated by verifying its identity is unchanged:

```{code-cell} ipython3
id(a)
```

## Random Numbers

+++

Random numbers are also a bit different in JAX, relative to NumPy.  Typically, in JAX, the state of the random number generator needs to be controlled explicitly.

```{code-cell} ipython3
import jax.random as random
```

First we produce a key, which seeds the random number generator.

```{code-cell} ipython3
key = random.PRNGKey(1)
```

```{code-cell} ipython3
type(key)
```

```{code-cell} ipython3
print(key)
```

Now we can use the key to generate some random numbers:

```{code-cell} ipython3
x = random.normal(key, (3, 3))
x
```

If we use the same key again, we initialize at the same seed, so the random numbers are the same:

```{code-cell} ipython3
random.normal(key, (3, 3))
```

To produce a (quasi-) independent draw, best practice is to "split" the existing key:

```{code-cell} ipython3
key, subkey = random.split(key)
```

```{code-cell} ipython3
random.normal(key, (3, 3))
```

```{code-cell} ipython3
random.normal(subkey, (3, 3))
```

The function below produces `k` (quasi-) independent random `n x n` matrices using this procedure.

```{code-cell} ipython3
def gen_random_matrices(key, n, k):
    matrices = []
    for _ in range(k):
        key, subkey = random.split(key)
        matrices.append(random.uniform(subkey, (n, n)))
    return matrices
```

```{code-cell} ipython3
matrices = gen_random_matrices(key, 2, 2)
for A in matrices:
    print(A)
```

One point to remember is that JAX expects tuples to describe array shapes, even for flat arrays.  Hence, to get a one-dimensional array of normal random draws we use `(len, )` for the shape, as in

```{code-cell} ipython3
random.normal(key, (5, ))
```

## JIT Compilation


The JAX JIT compiler accelerates logic within functions by fusing linear algebra operations into a single, highly optimized kernel that the host can launch on the GPU / TPU (or CPU if no accelerator is detected).

+++

Consider the following pure Python function.

```{code-cell} ipython3
def f(x, p=1000):
    return sum((k*x for k in range(p)))
```

Let's build an array to call the function on.

```{code-cell} ipython3
n = 50_000_000
x = jnp.ones(n)
```

How long does the function take to execute?

```{code-cell} ipython3
%time f(x).block_until_ready()
```

This code is not particularly fast.  While it is run on the GPU, since `x` is a DeviceArray, each vector `k * x` has to be instantiated before the final sum is computed.

If we JIT-compile the function with JAX, then the operations are fused and no intermediate arrays are created.

```{code-cell} ipython3
f_jit = jax.jit(f)   # target for JIT compilation
```

Let's run once to compile it:

```{code-cell} ipython3
f_jit(x)
```

And now let's time it.

```{code-cell} ipython3
%time f_jit(x).block_until_ready()
```

## Functional Programming

From JAX's documentation:

*When walking about the countryside of Italy, the people will not hesitate to tell you that JAX has "una anima di pura programmazione funzionale".*

+++

In other words, JAX assumes a functional programming style.

The major implication is that JAX functions should be pure:
    
* no dependence on global variables
* no side effects

"A pure function will always return the same result if invoked with the same inputs."

JAX will not usually throw errors when compiling impure functions but execution becomes unpredictable.

+++

Here's an illustration of this fact, using global variables:

```{code-cell} ipython3
a = 1  # global

@jax.jit
def f(x):
    return a + x
```

```{code-cell} ipython3
x = jnp.ones(2)
```

```{code-cell} ipython3
f(x)
```

In the code above, the global value `a=1` is fused into the jitted function.

Even if we change `a`, the output of `f` will not be affected --- as long as the same compiled version is called.

```{code-cell} ipython3
a = 42
```

```{code-cell} ipython3
f(x)
```

Changing the dimension of the input triggers a fresh compilation of the function, at which time the change in the value of `a` takes effect:

```{code-cell} ipython3
x = np.ones(3)
```

```{code-cell} ipython3
f(x)
```

Moral of the story: write pure functions when using JAX!

+++

## Gradients

+++

JAX can use automatic differentiation to compute gradients.

This can be extremely useful in optimization, root finding and other applications.

Here's a very simple illustration, involving the function

```{code-cell} ipython3
def f(x):
    return (x**2) / 2
```

Let's take the derivative:

```{code-cell} ipython3
f_prime = jax.grad(f)
```

```{code-cell} ipython3
f_prime(10.0)
```

Let's plot the function and derivative, noting that $f'(x) = x$.

```{code-cell} ipython3
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

fig, ax = plt.subplots()
x_grid = jnp.linspace(-4, 4, 200)
ax.plot(x_grid, f(x_grid), label="$f$")
ax.plot(x_grid, jax.vmap(f_prime)(x_grid), label="$f'$")
ax.legend(loc='upper center')
plt.show()
```

```{code-cell} ipython3

```


## Exercise

Recall that Newton's method for solving for the root of $f$ involves iterating on

+++

$$ q(x) = x - \frac{f(x)}{f'(x)} $$

Write a function called `newton` that takes a function $f$ plus a guess $x_0$ and returns an approximate fixed point.  Your `newton` implementation should use automatic differentiation to calculate $f'$.

Test your `newton` method on the function shown below.

```{code-cell} ipython3
f = lambda x: jnp.sin(4 * (x - 1/4)) + x + x**20 - 1
x = jnp.linspace(0, 1, 100)

fig, ax = plt.subplots()
ax.plot(x, f(x), label='$f(x)$')
ax.axhline(ls='--', c='k')
ax.set_xlabel('$x$', fontsize=12)
ax.set_ylabel('$f(x)$', fontsize=12)
ax.legend(fontsize=12)
plt.show()
```

```{code-cell} ipython3
# Put your code here
```

solution below

solution below

solution below

solution below

solution below

solution below

solution below

solution below

solution below

solution below




```{code-cell} ipython3
def newton(f, x_0, tol=1e-5):
    f_prime = jax.grad(f)
    def q(x):
        return x - f(x) / f_prime(x)

    error = tol + 1
    x = x_0
    while error > tol:
        y = q(x)
        error = abs(x - y)
        x = y
        
    return x
```

```{code-cell} ipython3
newton(f, 0.2)
```

This number looks good, given the figure.



## Exercise

This exercise uses parallelized gradient ascent to maximize a function.

Here's the function we want to maximize

```{code-cell} ipython3
@jax.jit
def f(x):
    return jnp.cos(x[0]**2 + x[1]**2) / (1 + x[0]**2 + x[1]**2) + 1

```

Here's one update step of gradient ascent

```{code-cell} ipython3
f_grad = jax.grad(f)

def update(x, f, f_grad, alpha=0.01):
    return x + alpha * f_grad(x)

x_0 = jnp.array((0.7, 0.7))

update(x_0, f, f_grad)
```

Let's vectorize it.

```{code-cell} ipython3
update_vec = jax.vmap(update, (0, None, None))
```

Let's test that this works as expected.

```{code-cell} ipython3
key = jax.random.PRNGKey(1)
n = 1000
xs = jax.random.uniform(key, (n, 2), minval=-3.0, maxval=3.0)
xs
```

```{code-cell} ipython3
update_vec(xs, f, f_grad)
```

We have updated every 2-vector in `xs` (every row) using the update rule.

The exercise is to run this in a loop and compute an approximate maximum of
the function.

```{code-cell} ipython3
# Put your code here
```

solution below

solution below

solution below

solution below

solution below

solution below

solution below

solution below

solution below

solution below


Here's a suitable function for the loop phase.

```{code-cell} ipython3
def gradient_ascent(f, f_grad, x_0, tol=1e-8, alpha=1e-2, max_iter=10_000):
    error = tol + 1
    x = x_0
    i = 0
    current_max = - jpn.inf
    while error > tol and i < max_iter:
        y = update_vec(x, f, f_grad)
        new_max = jnp.max(jax.vmap(f)(x))
        error = abs(new_max - current_max)
        current_max = new_max
        x = y
        i += 1
        
    return current_max, i
```

Now let's call it, starting from `xs`


```{code-cell} ipython3
gradient_ascent(f, f_grad, xs)
```
