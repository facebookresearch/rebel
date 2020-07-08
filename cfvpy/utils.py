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

from typing import Dict, Optional
import collections
import importlib
import logging
import pathlib
import subprocess
import time

import torch

import heyhi


class StopWatchTimer:
    def __init__(self, auto_start=True):
        self._elapsed: float = 0
        self._start: Optional[float] = None
        if auto_start:
            self.start()

    def start(self) -> None:
        self._start = time.time()

    @property
    def elapsed(self) -> float:
        if self._start is not None:
            return self._elapsed + time.time() - self._start
        else:
            return self._elapsed

    def pause(self) -> None:
        self._elapsed = self.elapsed
        self._start = None


class MultiStopWatchTimer:
    def __init__(self):
        self._start: Optional[float] = None
        self._name = None
        self._timings = collections.defaultdict(float)

    def start(self, name) -> None:
        now = time.time()
        if self._name is not None:
            self._timings[self._name] += now - self._start
        self._start = now
        self._name = name

    @property
    def timings(self) -> Dict[str, float]:
        if self._name is not None:
            self.start(self._name)
        return self._timings


def TimedContext(*args, **kwargs):
    import cfvpy.rela

    class TimedContextInner(cfvpy.rela.Context):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._timer = StopWatchTimer(auto_start=False)

        @property
        def running_time(self) -> float:
            """Returns the time in second the context was in working state."""
            return self._timer.elapsed

        def start(self):
            super().start()
            self._timer.start()

        def resume(self):
            super().resume()
            self._timer.start()

        def pause(self):
            super().pause()
            self._timer.pause()

    return TimedContextInner(*args, **kwargs)


def compute_exploitability(model_path, cfg_cfr_eval, cfr_binary="build/cfr"):
    root_dir = pathlib.Path(__file__).parent.parent.resolve()
    if cfg_cfr_eval.args.num_threads:
        num_threads = cfg_cfr_eval.args.num_threads
    else:
        num_threads = 10 if heyhi.is_on_slurm() else 40
    cmd = ("%s -t %d -linear -alternate -decompose -cfr_ave -cfvnet ") % (
        cfr_binary,
        num_threads,
    )
    for k, v in cfg_cfr_eval.args.items():
        if k == "num_threads":
            continue
        if v is True:
            cmd += f" -{k}"
        elif v is False:
            pass
        else:
            cmd += f" -{k} {v}"
    logging.debug("Going to run: %s", cmd)
    output = subprocess.check_output(
        cmd.split() + ["-model", str(model_path.resolve())], cwd=root_dir
    )
    values = []
    for line in output.decode("utf8").split("\n"):
        line = line.strip()
        if line.startswith("Summed Exploitability:"):
            values.append(float(line.split()[-1]))
    return values


def cfg_instantiate(cfg, *args, **kwargs):
    """Create a instance of a class specified in cfg."""
    package_name, classname = cfg.classname.rsplit(".", 1)
    package = importlib.import_module(package_name)
    constructor = getattr(package, classname)
    obj = constructor(*args, **(cfg.kwargs or {}), **kwargs)
    return obj


def _sanitize(value):
    if isinstance(value, torch.Tensor):
        return value.detach().item()
    return value


class FractionCounter:
    def __init__(self):
        self.numerator = self.denominator = 0

    def update(self, top, bottom=1.0):
        self.numerator += _sanitize(top)
        self.denominator += _sanitize(bottom)

    def value(self):
        return self.numerator / max(self.denominator, 1e-6)


def get_travertser_beliefs(player_id, beliefs):
    p1, p2 = torch.chunk(beliefs, 2, -1)
    traverser_is_first = (player_id % 2) == 0
    return torch.where(traverser_is_first, p1, p2)


class MaxCounter:
    def __init__(self, default=0):
        self._value = default

    def update(self, value):
        self._value = max(value, self._value)

    def value(self):
        return self._value
