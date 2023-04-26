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

# Equilibrium

+++

#### Author: [John Stachurski](http://johnstachurski.net/)

+++

In this notebook we solve a very simple market equilibrium problem.

Supply and demand are nonlinear and we use Newton's root-finding algorithm to solve for equilibrium prices.

We use the following imports:

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
```

## Prelude: A Note on Root-Finding

+++

If $f$ maps an interval $(a, b)$ into $\mathbb R$, then a **root** of the function $f$ is an $x^* \in (a,b)$ with $f(x^*)=0$.

A common method for root finding is Newton's algorithm.

We start with a guess $x_0 \in (a, b)$.

Then we replace $f$ with the tangent function $f_a(x) = f(x_0) + f'(x_0)(x - x_0)$ and solve for the root of $f_a$ (which can be done exactly).

Calling the root $x_1$, we have

$$ 
    f_a(x_1)=0
    \quad \iff \quad
    x_1 = x_0 - \frac{f(x_0)}{f'(x_0)} 
$$

This is our update rule:

$$
    x_{k+1} = q(x_k)
    \quad \text{where} \quad
    q(x) := x - \frac{f(x)}{f'(x)} 
$$


+++

The algorithm is implemented in `scipy.optimize`

```{code-cell} ipython3
from scipy.optimize import newton
```

Let's apply this to find the positive root of $f(x) = x^2 - 1$.

```{code-cell} ipython3
def f(x):
    return x**2 - 1

x_grid = np.linspace(-1, 2, 200)
fig, ax = plt.subplots()
ax.plot(x_grid, f(x_grid), label="$f$")
ax.plot(x_grid, np.zeros_like(x_grid), "k--")
ax.legend()
plt.show()

```

Here we call `newton`.

```{code-cell} ipython3
newton(f, 0.5)   # search for root of f starting at x_0 = 0.5
```

In the last call we didn't supply the gradient of $f$, so it was approximated
numerically.  

We can supply it as follows:

```{code-cell} ipython3
def f_prime(x):
    return 2 * x

newton(lambda x: x**2 - 1, 0.5, fprime=f_prime)
```

## The Market

+++

Now let's consider a market for coffee beans.  The price per kilo is $p$.  Total supply at price $p$ is

$$ q_s (p) = b \sqrt{p} $$

and total demand is 

$$ q_d (p) = a \exp(-p) + c, $$

where $a, b$ and $c$ are positive parameters.

+++

Now we write routines to compute supply and demand as functions of price and parameters.

We take $a=1$, $b=0.5$ and $c=1$ as "default" parameter values.

```{code-cell} ipython3
def supply(p, b=0.5):
    return b * np.sqrt(p)

def demand(p, a=1, c=1):
    return a * np.exp(-p) + c
```

Now we can call the functions as follows:

```{code-cell} ipython3
demand(2.0)  # with a and c at defaults
```

```{code-cell} ipython3
demand(2.0, a=0.4)  # a is specified and c remains at its defaults
```

etc.

+++

Note that these functions are automatically NumPy "universal functions":

```{code-cell} ipython3
p_vals = np.array((0.5, 1.0, 1.5))
supply(p_vals)
```

```{code-cell} ipython3
demand(p_vals)
```

### Exercise 1

Plot both supply and demand as functions of $p$ on the interval $[0, 10]$ at the default parameters.

* Put price on the horizonal axis.  
* Use a legend to label the two functions and be sure to label the axes.  
* Make a rough estimate of the equilibrium price, where demand equals supply.

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

solution below

solution below

```{code-cell} ipython3
fig, ax = plt.subplots()
p_grid = np.linspace(0, 10, 200)
ax.plot(p_grid, supply(p_grid), label='supply')
ax.plot(p_grid, demand(p_grid), label='demand')
ax.set_xlabel("price")
ax.set_ylabel("quantity")
ax.legend(frameon=False, loc='upper center')
plt.show()
```

The equilibrium price looks to be about 4.1.

+++

### Exercise 2

Write a function that takes arguments $a, b, c, p$, with default values $a=1$, $b=0.5$ and $c=1$, and returns *excess demand*, which is defined as

$$ e(p) = q_d(p) - q_s(p) $$

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

solution below

solution below


```{code-cell} ipython3
def excess_demand(p, a=1, b=0.5, c=1):
    return demand(p, a, c) - supply(p, b)
```

Now we test it:

```{code-cell} ipython3
excess_demand(1.0)
```

### Organizing our Code

If we have many functions working with the same parameters, it's hard to know where to put the default values.

As such, we normally collect them in a data structure, such as a class or a tuple.

Personally, I normally used `namedtuple` instances, which are lighter than classes but easier to work with than tuples.

Here's an example:

```{code-cell} ipython3
from collections import namedtuple

Params = namedtuple('Params', ('a', 'b', 'c'))

def create_market_params(a=1.0, b=0.5, c=1.0):
    return Params(a=a, b=b, c=c)


def supply(p, params):
    a, b, c = params
    return b * np.sqrt(p)

def demand(p, params):
    a, b, c = params
    return a * np.exp(-p) + c

def excess_demand(p, params):
    a, b, c = params
    return demand(p, params) - supply(p, params)
```

### Exercise 3

Using these functions, plot excess demand over the interval from $0.2$ up to $10$.  Also plot a horizontal line at zero.  The equilibrium price is where excess demand crosses zero.

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

solution below

solution below


```{code-cell} ipython3
params = create_market_params()

fig, ax = plt.subplots()
p_grid = np.linspace(0, 10, 200)
ax.plot(p_grid, excess_demand(p_grid, params), label='excess demand')
ax.plot(p_grid, np.zeros_like(p_grid), 'k--')
ax.set_xlabel("price")
ax.set_ylabel("quantity")
ax.legend()
plt.show()
```

### Exercise 4

+++

Write a function that takes an instance of `Params` (i.e, a parameter vector) and returns a market clearing price via Newton's method.

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

solution below

solution below


```{code-cell} ipython3
def compute_equilibrium(params, price_init=2.0):
    p_star = newton(lambda p: excess_demand(p, params), price_init)
    return p_star
```

```{code-cell} ipython3
params = create_market_params()
compute_equilibrium(params)
```

This looks about right given the figures above.

+++

### Exercise 5

For $b$ in a grid of 200 values between 0.5 and 1.0, plot the equilibrium price for each $b$.

Does the curve that you plotted slope up or down?  Try to provide an explanation for what you see in terms of market equilibrium.

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

solution below

solution below



```{code-cell} ipython3
b_grid = np.linspace(0.5, 1.0, 200)
prices = np.empty_like(b_grid)
for i, b in enumerate(b_grid):
    params = create_market_params(b=b)
    prices[i] = compute_equilibrium(params)
```

```{code-cell} ipython3
fig, ax = plt.subplots()
ax.plot(b_grid, prices, label="equilibrium prices")
ax.set_xlabel("$b$")
ax.set_ylabel("price")
ax.legend()
plt.show()
```

The curve slopes down because an increase in $b$ pushes up supply at any given price.  (In other words, the supply curve shifts up.)  

With greater supply, the price tends to fall.

```{code-cell} ipython3

```
