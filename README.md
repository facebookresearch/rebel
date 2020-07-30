# ReBeL

Implementation of [ReBeL](https://arxiv.org/abs/2007.13544), an algorithm that generalizes the paradigm of self-play reinforcement learning and search to imperfect-information games.
This repository contains implementation only for [Lair's Dice](https://en.wikipedia.org/wiki/Liar%27s_dice) game.

## Installation

The recommended way to install ReBeL is via conda env.

First, install dependencies:

```
pip install -r requirements.txt
conda install cmake
git submodule update --init
```

Then, compile the C++ part:

```
make
```

## Training a value net

Use the following command to train a value net:

```
python run.py --adhoc --cfg conf/c02_selfplay/liars_sp.yaml \
    env.num_dice=1 \
    env.num_faces=4 \
    env.fp.use_cfr=true \
    selfplay.cpu_gen_threads=60
```

Check the config [conf/c02_selfplay/liars_sp.yaml](conf/c02_selfplay/liars_sp.yaml) for all possible parameters. If use use Slurm to manage the cluster, add `launcher=slurm` to run the job on the cluster.


## Evaluating a value net

The trainer saves checkpoints every 10 epochs as state dictionaries and as TorchScript modules. You can use the latter to compute exploitability of strategy produced with such a model using the following command:

```
build/recursive_eval \
    --net path/to/model.torchscript \
    --mdp_depth 2 \
    --num_faces 4 \
    --num_dice 1 \
    --subgame_iters 1024 \
    --num_repeats 4097 \
    --num_threads 10 \
    --cfr
```

Setting `--num_repeats` to a positive value enables evaluation of a sampled policy, i.e., when we use a randomly selected iteration of the underlying subgame algorithm for the subgame. Computing the exact full policy produced by such a process is intractable. Therefore, we average `num_repeats` such policies to get an upper bound for the exploitability.

The script reports exploitability for both full tree solving and recursive solving.


## Pretrained checkpoints

We release checkpoints of value function for games 1x4f, 1x5f, 1x6f, and 2x3f. We report the average exploitability of these checkpoints in the paper. Use [eval_all.py](https://github.com/facebookresearch/rebel/blob/master/scripts/eval_all.py) script to download and evaluate all the models.

## Code structure

The training loop is implemented in Python and located in [cfvpy/selfplay.py](cfvpy/selfplay.py). The actual data generation part happens in C++ and could be found in [csrc/liarc_dice](csrc/liars_dice).

## License
Rebel is released under the Apache license. See [LICENSE](LICENSE) for additional details.


## Citation

```bibtex
@article{brown2020combining,
    title={Combining Deep Reinforcement Learning and Search for Imperfect-Information Games},
    author={Noam Brown and Anton Bakhtin and Adam Lerer and Qucheng Gong},
    year={2020},
    journal={arXiv:2007.13544}
}
```
