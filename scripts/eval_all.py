# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import json
import itertools
import os
import pathlib
import subprocess
import urllib.request

try:
    import pandas as pd
except ImportError:
    print("Install pandas for pretty tables")
    pd = None


import torch

Game = collections.namedtuple("Game", "num_dice,num_faces")
GAMES = (Game(1, 4), Game(1, 5), Game(1, 6), Game(2, 3))
SOLVERS = ("fp", "cfr")
EPOCHS = (980, 1000, 1020)
NUM_REPEATS = 1024

URL_PREFIX = "http://dl.fbaipublicfiles.com/rebel/liarsdice_ckpt/"
FILE_PATTERN = "{num_dice}x{num_faces}f_{solver}_{epoch}.ckpt"


# Where to save checkpoints and results.
EVAL_ROOT = pathlib.Path(__file__).parent.parent.resolve() / "exps" / "eval"
BINARY_PATH = pathlib.Path(__file__).parent.parent.resolve() / "build" / "recursive_eval"


def cache_eval(eval_f):
    def cached_eval_f(net_path, *, num_threads, **kwargs):
        suffix = "_".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        if suffix:
            suffix = f".{suffix}"
        cache_path = f"{net_path}.eval{suffix}"
        if not os.path.exists(cache_path):
            result = eval_f(net_path, num_threads=num_threads, **kwargs)
            torch.save(result, cache_path)
        return torch.load(cache_path)

    return cached_eval_f


@cache_eval
def run_eval(
    net_path,
    *,
    num_threads,
    num_faces,
    num_dice,
    solver,
    depth=2,
    subgame_iters=1024,
    num_repeats,
):
    assert BINARY_PATH.exists(), "Cannot find the eval binary. Try `make`"
    args = [
        str(BINARY_PATH),
        "--net",
        str(net_path),
        "--mdp_depth",
        str(depth),
        "--num_faces",
        str(num_faces),
        "--num_dice",
        str(num_dice),
        "--subgame_iters",
        str(subgame_iters),
        "--num_repeats",
        str(num_repeats),
        "--num_threads",
        str(num_threads),
    ]
    assert solver in ("cfr", "fp")
    if solver == "cfr":
        args.append("--cfr")

    try:
        out = subprocess.check_output(args)
    except:
        print(*args)
        raise
    out = out.decode("utf8")
    res = json.JSONDecoder(object_pairs_hook=collections.OrderedDict).decode(
        [line.split("XXX")[1] for line in out.split("\n") if "XXX" in line][0]
    )
    return res


def download_all():
    EVAL_ROOT.mkdir(exist_ok=True, parents=True)
    downloaded = []
    for game, epoch, solver in itertools.product(GAMES, EPOCHS, SOLVERS):
        params = dict(
            num_dice=game.num_dice, num_faces=game.num_faces, solver=solver, epoch=epoch
        )
        dst = EVAL_ROOT / FILE_PATTERN.format(**params)
        if not dst.exists():
            url = URL_PREFIX + FILE_PATTERN.format(**params)
            print("Downloading %s to %s" % (url, dst))
            urllib.request.urlretrieve(url, dst)
        downloaded.append((params, str(dst)))

    return downloaded


def download_and_eval_all(num_threads):
    data = []
    downloaded = download_all()
    for params, net_path in downloaded:
        epoch = params.pop("epoch")
        metrics = run_eval(
            net_path, num_threads=num_threads, num_repeats=NUM_REPEATS, **params
        )
        data.append(
            dict(
                eval_type="full",
                exploitability=float(metrics["full_tree"]),
                epoch=epoch,
                **params,
            )
        )
        data.append(
            dict(
                eval_type="rebel",
                exploitability=float(metrics[f"repeated toleaf {NUM_REPEATS}"]),
                epoch=epoch,
                **params,
            )
        )
    if pd is not None:
        df = pd.DataFrame(data)
        print("Average exploitability over 3 epochs")
        print(
            df.pivot_table(
                "exploitability", ["eval_type", "solver"], ["num_dice", "num_faces"]
            )
        )
    else:
        for row in data:
            print(row)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_threads", type=int, default=10)
    download_and_eval_all(**vars(parser.parse_args()))
