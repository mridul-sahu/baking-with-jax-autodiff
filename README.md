# Baking with JAX Autodiff 🎂: From Basics to Advanced Recipes

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1p201cehNanuWhh8dc_DzzduuKUmGB6n4) Learn JAX's powerful automatic differentiation (autodiff) system through a fun and relatable cake-baking analogy! 🧑‍🍳

This repository contains a tutorial notebook (`.ipynb`) that walks you through core autodiff concepts and practical JAX implementations. We start with the basics of getting gradients and progressively build up to advanced techniques like Jacobians, Hessians, complex number differentiation, and even defining your own custom "secret recipes" for derivatives.

## Why This Tutorial? 🤔

* **Intuitive Analogy:** Uses cake baking to make potentially complex topics like Jacobians, Hessians, JVPs/VJPs, and custom rules easier to grasp.
* **Comprehensive Coverage:** Goes beyond `jax.grad` to cover a wide range of JAX's autodiff features.
* **Practical Code:** Provides runnable code examples within a cohesive narrative.
* **Target Audience:** Suitable for those new to JAX autodiff or intermediate users looking for a deeper, more practical understanding.

## Tutorial Outline ("Our Baking Course") 📚

1.  **The Basics**
    Getting started with gradients (`jax.grad`), handling different input/output structures (PyTrees), and verifying results. Covers `jax.value_and_grad`.
2.  **Advanced Baking**
    Dealing with multiple cake properties at once (Jacobians with `jax.jacfwd`/`jax.jacrev`), understanding the rate at which improvements change (Hessians with `jax.hessian`), efficiently calculating directional changes (JVPs/VJPs with `jax.jvp`/`jax.vjp`, and HVPs), and controlling gradient flow (`jax.lax.stop_gradient`, `jax.vmap`).
3.  **Exotic Flavors & Vibrations**
    Exploring how JAX differentiates functions involving complex numbers, understanding the difference between 'smooth' (holomorphic) and 'tricky' (non-holomorphic) cases, and using the right tools (`grad` with `holomorphic=True`, Jacobians) accordingly.
4.  **Secret Family Recipes**
    Teaching JAX custom differentiation rules using `jax.custom_jvp` and `jax.custom_vjp` to overcome limitations, such as fixing numerical instability, enforcing specific baking rules (like gradient clipping), or handling complex iterative processes (like dough maturation) that standard autodiff struggles with.

## Key JAX Concepts/Functions Covered ✨

* Automatic Differentiation (Forward & Reverse Modes)
* `jax.grad`: Computing gradients of scalar functions.
* `jax.value_and_grad`: Computing function value and gradient together efficiently.
* Higher-Order Derivatives: Stacking `grad` for second, third, etc. derivatives.
* PyTrees: Differentiating with respect to standard Python containers (dicts, lists, tuples).
* `jax.jacfwd`, `jax.jacrev`: Computing full Jacobian matrices using forward and reverse modes.
* `jax.jvp`, `jax.vjp`: Understanding the core Jacobian-Vector Product and Vector-Jacobian Product primitives.
* `jax.hessian`: Computing full Hessian matrices (second derivatives).
* Hessian-Vector Products (HVPs): Efficiently computing the action of the Hessian on a vector (`H @ v`) without forming the full `H`.
* `jax.lax.stop_gradient`: Preventing gradient flow through specific parts of a computation for algorithmic control.
* `jax.vmap`: Combining with `grad` for efficient per-example or batched gradient calculations.
* Complex Number Differentiation: Handling differentiation involving complex numbers, including the `holomorphic=True` argument for `jax.grad`.
* `jax.custom_jvp`: Defining custom forward-mode differentiation rules (and often getting reverse-mode automatically).
* `jax.custom_vjp`: Defining custom reverse-mode differentiation rules for fine-grained control or handling complex operations.
