# Small-World Networks: an experimental study on real-world datasets

The present repository showcases the implementation of an experimental examination of the Small-World Networks (SWN) theory. The examination is based on real-world datasets, and the outcomes are presented through a Jupyter Notebook. As there is no corresponding written paper, the notebook serves as the sole source of information regarding the study. It encompasses all relevant theoretical background, details of the experimental design, presentation of results, and derivation of conclusions.

---

Time spent one the project since the beginning of 2023 (add ~20 hours for a real estimate)

[![wakatime](https://wakatime.com/badge/user/a3116382-7adb-43ba-9490-83130c4b22c5/project/3db09d02-0a29-49b1-9a39-08c74e3df4ce.svg)](https://wakatime.com/badge/user/a3116382-7adb-43ba-9490-83130c4b22c5/project/3db09d02-0a29-49b1-9a39-08c74e3df4ce)

---

## Documentation

### Notebook structure

The core component of the project is a Jupyter Notebook named `main.ipynb`. The notebook is comprehensive and provides ample information to comprehend the study and its results.

- The notebook starts with a theoretical background of the study, delving into the theory of random networks. The Erdős-Rényi and Watts-Strogatz models receive particular attention, as their properties form the basis of the study.

- Next, the experimental setup is presented. There is a detail description of the datasets, along with the methodology adopted for data extraction, transformation, and loading. In this section, the graphs are constructed from the raw data obtained from the datasets.

- The properties of the constructed graphs are then calculated, including the average degree, average clustering coefficient, average shortest path length, and average betweenness centrality.

- The results of the study are then presented, followed by a comprehensive analysis and discussion of the outcomes.

- Finally, we attempt to understand whether the networks are small-world or not by employing different approaches and evaluating the consistency of the results.


### Algorithms and functions implemented

In order to maintain a clean and organized notebook, I have integrated all algorithms and functions into the `utils.py` file. This module serves as a centralized repository of functions and is imported into the notebook for use in the study.

A thorough technical examination of the project requires a close examination of this file, as it holds nearly all the code utilized. Each function is accompanied by detailed documentation to ensure clear understanding and ease of use.

### External scripts

The computation of the omega coefficient, a crucial measure of the small-worldness of a network, is one of the most vital functions in this project. However, its application proves to be extremely slow, rendering its usage in the notebook inadvisable. To address this issue, I have developed a separate script, named `omega_sampled_server.py`. The script is intended for execution on a server machine, capable of handling the extensive processing demands for extended periods of time, to compute the omega coefficient for a specified graph.

To run the script, execute the following command:

```bash
./omega_sampled_server.py graph --k --niter --nrand
```

Where:

- `graph` is the name of the graph
- `k` Percentage of nodes to be remove
- `niter` Number of rewiring operations per edge
- `nrand` Number of random graphs to be generated

For further details run `./omega_sampled_server.py --help`

#### Parallel version

However, the computation of the omega coefficient was still very slow. To speed up the process, I developed a parallel version of the script above, called `omega_parallel_server.py`. Its development required substantial effort as the parallelization of a function utilizing a random number generator is a complex task.

To run the script, execute the following command:

```bash
./omega_sampled_parallel.py graph --k --niter --nrand --n_processes --seed
```

Where:

- `graph` is the name of the graph
- `k` Percentage of nodes to be remove
- `niter` Number of rewiring operations per edge
- `nrand` Number of random graphs to be generated
- `n_processes` is an argument that specifies the number of processes to be used
- `seed` is the seed for the random number generator

For further details run `./omega_sampled_parallel.py --help`



## Requirements

The project has been developed in Python 3.10.9, to install the required libraries run the following command:

```bash
pip install -r requirements.txt
```

**NOTE**: I have no access to Windows or MacOS machines, so I cannot guarantee whether the project will function optimally on these platforms. However, I have made a concerted effort to ensure broad compatibility by implementing the project in a manner that should allow for its seamless operation on any platform. All the functions have been test on a Arch Linux machine, with an AMD Ryzen 5 2600 processor and 16GB of RAM. The version of Python used is 3.10.9.

### Experimental

There is a repository named `experimental` that contains a number of scripts written in `C++` with the purpose of speeding up the computation of the omega coefficient. The scripts are not used in the project, but they are included for completeness. This scripts are not documented, not tested, and not guaranteed to work. I suggest to ignore them unless you are a C++ enthusiast that loves to read experimental code.

## References

Here I report a list of the references utilized for the implementation of the project. This information is also available in the notebook.

> _In no particular order_

`[1]` On the evolution of random graphs, P. Erdős, A. Rényi, _Publ. Math. Inst. Hungar. Acad. Sci._, 5, 17-61 (1960).

`[2]` Complex Networks: Structure, Robustness, and Function, R. Cohen, S. Havlin, D. ben-Avraham, H. E. Stanley, _Cambridge University Press, 2009_.

`[3]` Collective dynamics of 'small-world' networks, D. J. Watts and S. H. Strogatz, _Nature_, 393, 440-442, 1998.

`[4]` On random graphs I, P. Erdős and A. Rényi, _Publ. Math. Inst. Hungar. Acad. Sci._, 5, 290-297, 1960.

`[5]` Generalizations of the clustering coefficient to weighted complex networks, M. E. J. Newman, _Physical Review E_, 74, 036104, 2006.

`[6]` The ubiquity of small-world networks. Telesford QK, Joyce KE, Hayasaka S, Burdette JH, Laurienti PJ. _Brain Connect_. 2011;1(5):367-75

`[8]` Humphries and Gurney (2008). “Network ‘Small-World-Ness’: A Quantitative Method for Determining Canonical Network Equivalence”. PLoS One. 3 (4)

`[9]` The brainstem reticular formation is a small-world, not scale-free, network M. D. Humphries, K. Gurney and T. J. Prescott, Proc. Roy. Soc. B 2006 273, 503-511,

`[10]` Sporns, Olaf, and Jonathan D. Zwi. “The small world of the cerebral cortex.” Neuroinformatics 2.2 (2004): 145-162.

`[11]` Maslov, Sergei, and Kim Sneppen. “Specificity and stability in topology of protein networks.” Science 296.5569 (2002): 910-913.

`[13]` B. Bollob ́as, Random Graphs, 1985. London: Academic Press

`[14]` R. Cohen, K. Erez, D. ben-Avraham, and S. Havlin, Resilience of the Internet to
random breakdown, Physical Review Letters 85 (2000), 4626–4628

`[15]` Dingqi Yang, Bingqing Qu, Jie Yang, Philippe Cudre-Mauroux, Revisiting User Mobility and Social Relationships in LBSNs: A Hypergraph Embedding Approach, In Proc. of The Web  Conference (WWW'19). May. 2019, San Francisco, USA.

`[16]` Ulrik Brandes, A Faster Algorithm for Betweenness Centrality, Journal of Mathematical Sociology, 25(2):163-177, 2001._

`[17]` Error and attack tolerance of complex networks, R. Albert, Nature volume 406, pages378–382 (2000)
