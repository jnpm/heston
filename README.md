# The Heston Model

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/jnpm/heston/blob/main/LICENSE)

## Seminar in Empirical Finance Winter Term 20/21

This repository includes the code and supplementary material for the seminar paper on the Heston model.

## Project structure

The application folder contains the option chain data ``1yopt.xslx`` as well as the Python code ``heston.py``.
Supplementary material such as interactive graphs and the data output of the empirical analysis can be found on [jnpm.github.io/heston](https://jnpm.github.io/heston/).

## Code
All packages needed to run the code base can be installed from the included ``requirements.txt``. The code has been tested on Python 3.8 and 3.9. Older versions might lack used built-in Python functions or are incompatible with some packages.

The code should be run with caution since it includes computationally intensive functions and uses multiple cores (by default **all** available cores) for faster computation. Computation using a single core has been included optionally and has to be uncommented.
