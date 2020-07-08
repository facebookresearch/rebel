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

from typing import Sequence, Optional
import argparse
import logging
import os
import resource
import pathlib
import pprint

import torch

import cfvpy.tasks
import heyhi

EXP_DIR: pathlib.Path = pathlib.Path(os.environ.get("HH_EXP_DIR", "exps/"))
CFG_PATH: pathlib.Path = pathlib.Path(__file__).resolve().parent / "conf" / "common"


@heyhi.save_result_in_cwd
def main(cfg):
    heyhi.setup_logging()
    logging.info("CWD: %s", os.getcwd())
    logging.info("cfg:\n%s", cfg.pretty())
    resource.setrlimit(
        resource.RLIMIT_CORE, (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
    )
    logging.info("resource.RLIMIT_CORE: %s", resource.RLIMIT_CORE)
    heyhi.log_git_status()
    logging.info("Is AWS: %s", heyhi.is_aws())
    logging.info("is on slurm:%s", heyhi.is_on_slurm())
    if heyhi.is_on_slurm():
        logging.info("Slurm job id: %s", heyhi.get_slurm_job_id())
    logging.info("Is master: %s", heyhi.is_master())

    task = getattr(cfvpy.tasks, cfg.task)
    return task(cfg)


def run(
    overrides: Sequence[str],
    cfg: pathlib.Path,
    mode: heyhi.ModeType,
    adhoc: bool = False,
    force_override_exp_id: Optional[str] = None,
    force_override_tag: Optional[str] = None,
) -> heyhi.ExperimentDir:
    """Computes the task locally of remotely if neeeded in the mode.

    The function checks the exp_handle first to detect whether the experiment
    is running, dead, or dead. Depending on that and the mode the function
    may kill the job, wipe the exp_handle, start a computation or do none of
    this.

    See heyhi.handle_dst for how the modes are handled.

    The computation may run locally or on the cluster depending on the
    launcher config section. In both ways main(cfg) with me executed with the
    final config with all overrides and substitutions.
    """
    heyhi.setup_logging()
    logging.info("Config: %s", cfg)
    logging.info("Overrides: %s", overrides)

    if not CFG_PATH.exists():
        CFG_PATH.mkdir(exist_ok=True, parents=True)

    exp_handle, need_run = heyhi.handle_dst(
        EXP_DIR,
        mode,
        cfg,
        overrides,
        adhoc,
        force_override_exp_id=force_override_exp_id,
        force_override_tag=force_override_tag,
    )
    logging.info("Exp dir: %s", exp_handle.exp_path)
    logging.info("Job status [before run]: %s", exp_handle.get_status())
    if need_run:
        heyhi.run_with_config(main, exp_handle, cfg, overrides, [CFG_PATH])
    if exp_handle.is_done():
        result = torch.load(exp_handle.result_path)
        if result is not None:
            simple_result = {
                k: v for k, v in result.items() if isinstance(v, (int, float, str))
            }
            pprint.pprint(simple_result, indent=2)
    return exp_handle


def parse_args_and_run():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cfg", required=True, type=pathlib.Path)
    parser.add_argument("--adhoc", action="store_true")
    parser.add_argument("--mode", choices=heyhi.MODES, default="gentle_start")
    args, overrides = parser.parse_known_args()
    return run(overrides, **vars(args))


if __name__ == "__main__":
    parse_args_and_run()
