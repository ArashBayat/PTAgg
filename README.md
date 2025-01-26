# Simulation Project

This project implements a simulation for the idea described in [the article](https://www.preprints.org/manuscript/202501.1925/v1).
The idea is to achive privacy and trust together in private data aggregation (focusing on voting systems)

The only input to the simulator (command line argument) is a yaml file that inlude a list of parameter set.
Each parameter set describes one simulation. The program performs all simulation and output each simulation results in a separate files.
There are two formats for the parameter set.
- first format is used when you would like to name candidate and imaginary candidates
- second format is when you specify number of candidate. The program name candidate as `c1`, `c2`, ..., `cN` where `N` is the number of candidate. Similarly imaginary candidate are are named `i1`, `i2`, ..., `iN`

In each parameter set you need to identify the following parameter (see below example that include both format)
- `sim_name`: name of simulation
- `result_file`: result file name without file extention
- `num_repeats`: number of time simulation should be repeated
- `candidate_votes`: number of vote given to each candidate
- `alpha`: as described in [the article](https://www.preprints.org/manuscript/202501.1925/v1)

```yaml
- sim_name: "ExampleSimulation_1"
  result_file: "MyResult_1"
  num_repeats: 3
  candidate_votes:
    - Alice: 65000
    - Bob: 110000
  imaginary_candidate:
    - Red
    - Blue
  alpha: 0.8
- sim_name: "ExampleSimulation_2"
  result_file: "MyResult_2"
  num_repeats: 4
  num_candidates: 3
  candidate_votes:
    - 65000
    - 110000
    - 45000
  alpha: 0.6
```

The program create two output file per simulation:
- A `csv` file including the following columns:
    - `[candidate]_actual`: Number of vote for each `[candidate]`
    - `[imaginary_candidate]_expected`: Expected number of votes per `[imaginary_candidate]` (theoritical probability applies)
    - [candidate]_to_[imaginary_candidate]: Number of vote form each `[candidate]` that transform to vote for each of `[imaginary_candidate]` (experimental probability apply)
    - `[imaginary_candidate]_counted`: Number of votes per `[imaginary_candidate]` (experimental probability apply)
    - `[candidate]_computed`: Number of votes computed for each `[candidate]` given the number of votes for imaginary candidates (experimental probability apply)
    - `[candidate]_percent_error`: Percentage of error comparing actual candidate votes with computed candidate votes. Calculated as `(abs(actual_votes - computed_votes) / actual_votes) * 100`.
    - `max_percent_error`: Maximum percentage error across all candidate.
- A `yaml` file including simulation parameter, probability matrix and its inverted matrix as well as summary statistics for percent error per candidate and over all candidate

Here is an example `csv` output:

| sim_index | Alice_actual | Bob_actual | Red_expected | Blue_expected | Alice_to_Red | Alice_to_Blue | Bob_to_Red | Bob_to_Blue | Red_counted | Blue_counted | Alice_computed | Bob_computed | Alice_percent_error |   Bob_percent_error |   max_percent_error |
|----------:|-------------:|-----------:|-------------:|--------------:|-------------:|--------------:|-----------:|------------:|------------:|-------------:|---------------:|-------------:|--------------------:|--------------------:|--------------------:|
|         0 |        65000 |     110000 |        74000 |        101000 |        52018 |         12982 |      21816 |       88184 |       73834 |       101166 |          64723 |       110277 | 0.42615384615384616 | 0.25181818181818183 | 0.42615384615384616 |
|         1 |        65000 |     110000 |        74000 |        101000 |        52070 |         12930 |      21979 |       88021 |       74049 |       100951 |          65082 |       109918 | 0.12615384615384614 | 0.07454545454545454 | 0.12615384615384614 |
|         2 |        65000 |     110000 |        74000 |        101000 |        52120 |         12880 |      22101 |       87899 |       74221 |       100779 |          65368 |       109632 |  0.5661538461538461 | 0.33454545454545453 |  0.5661538461538461 |

Here is an example of `yaml` output

```yaml
sim_name: ExampleSimulation_1
result_file: MyResult_1
num_repeats: 3
num_candidates: 2
alpha: 0.8
beta: 0.19999999999999996
candidate_vote:
- Alice: 65000
- Bob: 110000
expected_imaginary_votes:
  Red: 74000.0
  Blue: 101000.0
probability:
  Alice:
    Red: 0.8
    Blue: 0.2
  Bob:
    Red: 0.2
    Blue: 0.8
inverted_probability:
  Alice:
    Red: 1.33333
    Blue: -0.33333
  Bob:
    Red: -0.33333
    Blue: 1.33333
json:
  candidates: '["Alice", "Bob"]'
  votes: '[65000, 110000]'
  imaginary_candidate: '["Red", "Blue"]'
  expected_imaginary_vote: '[74000.0, 101000.0]'
  probability: '[[0.8, 0.19999999999999996], [0.19999999999999996, 0.8]]'
  inverted_probability: '[[1.3333333333333333, -0.3333333333333332], [-0.3333333333333332, 1.333333333333333]]'
percent_error_summary_statistics:
  Alice:
    count: 3.0
    mean: 0.3728205128205128
    std: 0.22479620400116487
    min: 0.12615384615384614
    25%: 0.27615384615384614
    50%: 0.42615384615384616
    75%: 0.49615384615384617
    max: 0.5661538461538461
  Bob:
    count: 3.0
    mean: 0.2203030303030303
    std: 0.13283412054614288
    min: 0.07454545454545454
    25%: 0.16318181818181818
    50%: 0.25181818181818183
    75%: 0.2931818181818182
    max: 0.33454545454545453
  AllCandidate:
    count: 6.0
    mean: 0.29656177156177155
    std: 0.18506693250789402
    min: 0.07454545454545454
    25%: 0.15756993006993006
    50%: 0.2931818181818182
    75%: 0.40325174825174825
    max: 0.5661538461538461
```

## Requirements

- Python 3.x
- `numpy`
- `pandas`
- `pyyaml`

## Installation

To install the required packages, run:

```sh
pip install numpy pandas pyyaml
```

## Experimental Results
The parameter file and experimental simulation results of the experiment mentioned in [the article](https://www.preprints.org/manuscript/202501.1925/v1) are stored in [experiment](./experiment) directory