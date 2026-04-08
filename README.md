# Combining Evolutionary Strategies and Novelty Detection to go Beyond the Alignment Limit of the $Z_2\times Z_2$ C3HDM

Repository of the code implementing an Evolutionary Strategy enhanced with Novelty Detection used in the paper [*Unearthing large pseudoscalar Yukawa couplings with Machine Learning*](https://arxiv.org/abs/2505.10625). The methodology builds upon the original work presented in [*Combining Evolutionary Strategies and Novelty Detection to go Beyond the Alignment Limit of the Z<sub>3</sub> 3HDM*](https://arxiv.org/abs/2402.07661)

> Note: This repository does not include the code that computes the $Z_2\times Z_2$ C3HDM observables, but the main Evolutionary Strategy with Novelty Detection loop presented herein can be adapted to other problems.


# Outline of the code

The code is organised in such a way that a single run is executed inside a `docker` container. The container is executed by `run.py`, and the python code that performs the scan is inside the `src` folder.

## `src` folder

The `src` folder includes all the python. The entrypoint is `scan.py` which performs a scan according to the configurations provided (see below). From here, the code picks an algorithm (`cmaes` or `rs` -- random sampler), and a density penalty for the novelty reward. Multiple penalties are provided, but in the papar only `HBOS` was used after initial experimentation.

The interface with $3HDM$ routine happens explicitly in `src/utils/process-points.py` where in `evaluate_population_batch` the points are saved to a file, `in.dat`, processed by `3HDM-Main`, and the outputs are recovered. The parameters are defined in `src/utils/parameters.py`, the constraints in `src/utils/constraints.py`. The file `src/utils/data.py` includes various utilities for handling the inputs and outputs of `3HDM-Main`.

In the future, we intend to adapt the code to be modular and easily adaptable to any balck-box/HEP package.

## Dockerfile

The `Dockerfile` shows how the image running the code was prepared, but notice that the line
```dockerfile
ADD 3HDMZ3.tar.gz /app/
```
will fail as the `3HDM` computational routine is not provided.

Alternatively, one can prepare a `python` virtual environment with the packages and their versions specified in `requirements.txt`:

```bash
$ python -m venv venv
$ source venv/bin/activate
(venv) $ pip install -r requirements.txt
```

## Configuration Files

The `configs` folder has a collection of configuration files. The `constraints-bounds` and `parameter-bounds` define the bounds on constraints and parameter, respectively. The `defaults` file defines the parameters of the scan.

The files with a `-local` suffix can be used to override the values of the configurations without having to change their values. This is useful to perform multiple runs while keep the original paramters.

## Notebooks and analyses

The analyses presented in the paper were performed using the notebooks in the `notebooks` folder.

# Citation

```bibtex
@article{deSouza:2025bpl,
    author = "de Souza, Fernando Abreu and Boto, Rafael and Crispim Rom{\~a}o, Miguel and Figueiredo, Pedro N. and Rom{\~a}o, Jorge C. and Silva, Jo{\~a}o P.",
    title = "{Unearthing large pseudoscalar Yukawa couplings with machine learning}",
    eprint = "2505.10625",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    reportNumber = "IPPP/25/28",
    doi = "10.1007/JHEP07(2025)268",
    journal = "JHEP",
    volume = "07",
    pages = "268",
    year = "2025"
}

```
