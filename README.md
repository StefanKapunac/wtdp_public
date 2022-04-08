# Variable Neighborhood Search for Weighted Total Domination Problem and Its Application in Social Network Information Spreading

This is a public repo containing source code, instances, results and supplementary material from the paper 'Variable Neighborhood Search for the Weighted Total Domination Problem and Its Application in Social Network Information Spreading' by Stefan Kapunac, Aleksandar Kartelj and Marko Djukanovic.

## Contents
- doc - contains the main paper and supplementary pdf;
- instances - contains all four groups of instances used in the paper;
- src - contains Python source code of all algorithms presented in the paper;
- test - contains all results presented in the paper along with some exemplary calls of used methods.

## Requirements
- Python 3
- Networkx
- sortedcontainers

### ILP
- CPLEX
- Docplex

### Visualization
- Matplotlib
- Graphistry (free version)

## Usage
```
python vns.py <DIR_PATH> <NAME_MASK> <KMIN> <KMAX_FUNC (logn/sqrtn/ndiv5/ndiv10)> <KMAX_MAX> <NUM_ITERS> <TIME_SECONDS> <NUM_RUNS> <PERFORM_LS2 (yes/no)>

or

pypy vns.py <DIR_PATH> <NAME_MASK> <KMIN> <KMAX_FUNC (logn/sqrtn/ndiv5/ndiv10)> <KMAX_MAX> <NUM_ITERS> <TIME_SECONDS> <NUM_RUNS> <PERFORM_LS2 (yes/no)>
```

| Parameter | Description |
| --------- | ----------- |
| <DIR_PATH> | path to directory with instances |
| <NAME_MASK> | use only instances whose name contains <NAME_MASK> |
| \<KMIN\> | minimal neighborhood k_min |
| <KMAX_FUNC> | allowed functions are logn, sqrtn, ndiv5 and ndiv10 |
| <KMAX_MAX> | maximal value for k_max, k_max = min(KMAX_FUNC, KMAX_MAX) |
| <NUM_ITERS> | maximal number of iterations |
| <TIME_SECONDS> | maximal execution time in seconds |
| <NUM_RUNS> | number of runs |
| <PERFORM_LS2> | indicator if LocalSearch2 should be used (yes/no) |
