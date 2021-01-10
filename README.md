# ReBeL

Implementation of [ReBeL](https://arxiv.org/abs/2007.13544), an algorithm that generalizes the paradigm of self-play reinforcement learning and search to imperfect-information games.
This repository contains implementation only for [Liar's Dice](https://en.wikipedia.org/wiki/Liar%27s_dice) game.

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

Use the following command to train a value net with data generation placed on CPU:

```
python run.py --adhoc --cfg conf/c02_selfplay/liars_sp.yaml \
    env.num_dice=1 \
    env.num_faces=4 \
    env.subgame_params.use_cfr=true \
    selfplay.cpu_gen_threads=60
```

As CFR requires evaluation of the value function for several nodes at each iteration, the code above will be pretty slow. If you have multiple GPUs on your machine, use these flags instead. First GPU will be used for training and the others will be used for data generation with 8 CPU threads per each GPU:

```
python run.py --adhoc --cfg conf/c02_selfplay/liars_sp.yaml \
    env.num_dice=1 \
    env.num_faces=4 \
    env.subgame_params.use_cfr=true \
    selfplay.cpu_gen_threads=0  \
    selfplay.threads_per_gpu=8
```

Check the config [conf/c02_selfplay/liars_sp.yaml](conf/c02_selfplay/liars_sp.yaml) for all possible parameters. If use use Slurm to manage the cluster, add `launcher=slurm_8gpus launcher.num_gpus=NUM_GPUS` to run the job on the cluster. If you specify `NUM_GPUS > 8`, the code will assume that you are launching on several machines with 8 GPUs each.


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
@article{brown2020rebel,
  title={Combining deep reinforcement learning and search for imperfect-information games},
  author={Brown, Noam and Bakhtin, Anton and Lerer, Adam and Gong, Qucheng},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```
