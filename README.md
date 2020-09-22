# Point-to-Hyperplane Nearest Neighbor Search beyond the Unit Hypersphere

## Introduction

This source package provides the implementations and experiments of NH and FH for Point-to-Hyperplane Nearest Neighbor Search (P2HNNS) in high-dimensional Euclidean spaces.

We also implement three state-of-the-art hyperplane hashing schemes EH, BH, and MH and two heuristic linear scan methods Random-Scan and Sorted-Scan for comparison.

## Datasets and Queries

We use five real-life datasets `Yelp`, `Music-100`, `GloVe`, `Tiny-1M`, and `Msong` in the experiments. For each dataset, we generate 100 hyperplane queries for evaluations. 

The datasets and queries can be found by the follow link:
https://drive.google.com/drive/folders/1aBFV4feZcLnQkDR7tjC-Kj7g3MpfBqv7?usp=sharing

The statistics of datasets and queries are summarized as follows.

| Datasets  | #Data Objects | Dimensionality | #Queries | Data Size | Type   |
| --------- | ------------- | -------------- | -------- | --------- | ------ |
| Yelp      | 77,079        | 50             | 100      | 14.7 MB   | Rating |
| Music-100 | 1,000,000     | 100            | 100      | 381.5 MB  | Rating |
| GloVe     | 1,183,514     | 100            | 100      | 451.5 MB  | Text   |
| Tiny-1M   | 1,000,000     | 384            | 100      | 1.43 GB   | Image  |
| Msong     | 992,272       | 420            | 100      | 1.55 GB   | Audio  |

## Compilation

This source package requires ```g++-8``` with ```c++17``` support. Before the compilation, please check whether the `g++-8` is installed. If not, please install it first. We provide a way to install `g++-8` in Ubuntu 18.04 as follows:

```bash
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install g++-8
sudo apt-get install gcc-8 (optional)
```

To compile the c++ code, please type commands as follows:

```bash
cd methods
make
```

## Run Experiments

We provide the bash scripts to reproduce all of experiments reported in manuscript `ID=240`. Please first download the datasets and queries and copy them to to directory `data/bin/`. For example, when you get `Yelp.ds` and `Yelp.q` from the `Yelp/` folder, please move them to `data/bin/Yelp/Yelp.ds` and `data/bin/Yelp/Yelp.q`.

Once moving datasets and queries to `data/bin/`, you can run all experiments by simply typing the following command:

```bash
cd methods
bash run_all.sh
```

We provide the dataset Yelp in this package for your reference. A quick example (run NH and FH on Yelp) is shown as follows:

```bash
# generate ground truth results
./plane -alg 0 -n 77079 -qn 100 -d 50 -ds ../data/bin/Yelp/Yelp.ds -qs ../data/bin/Yelp/Yelp.q -ts ../data/bin/Yelp/Yelp.gt

# NH
./plane -alg 10 -n 77079 -qn 100 -d 50 -m 256 -w 1.0 -cf run.conf -dn Yelp -ds ../data/bin/Yelp/Yelp.ds -qs ../data/bin/Yelp/Yelp.q -ts ../data/bin/Yelp/Yelp.gt -of ../results/

# FH
./plane -alg 7 -n 77079 -qn 100 -d 50 -m 16 -b 0.9 -cf run.conf -dn Yelp -ds ../data/bin/Yelp/Yelp.ds -qs ../data/bin/Yelp/Yelp.q -ts ../data/bin/Yelp/Yelp.gt -of ../results/
```

## Draw Figures

Finally, to reproduce all results, we also provide pythom scripts (i.e., `plot.py` and `plot_para.py`) to draw the figures appeared in the manuscript `ID=240`. These two scripts require python 3.7 (or higher verison) with numpy, scipy, and matplotlib installed. 

To draw the figures with the experimental results, please type commands as follows:

```bash
python3 plot.py
python3 plot_para.py
```
