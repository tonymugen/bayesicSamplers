# Overview

This library implements Markov chain Monte Carlo samplers I use to fit Bayesian models. There are currently two samplers available: a simple [Metropolis](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm) updating scheme using a Gaussian proposal that must be tuned by hand, and my own implementation of the [No-U-Turn](https://arxiv.org/abs/1111.4246) Sampler (NUTS) with automatic tuning of sampler parameters. I use NUTS for production code, but the Metropolis sampler is useful for debugging and quick model implementation tests.

## Dependencies

The library depends on a C++ compiler that understands the C++-11 standard. It also requires a set of numerical utilities that I collected in the [bayesicUtilities](https://github.com/tonymugen/bayesicUtilities) repository. I assume that the utilities are available in a `bayesicUtilities` directory at the same level as `bayesicSamplers`. This can be changed by modifying `#include` paths in the header files.
