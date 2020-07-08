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
import logging
import pathlib
import os
import time

import pytorch_lightning.logging as pl_logging
import omegaconf
import torch
import tqdm

import cfvpy.models
import cfvpy.rela
import cfvpy.utils
import heyhi


def _build_model(device, env_cfg, cfg, state_dict=None, half=False, jit=False):
    assert cfg is not None
    model_name = cfg.name
    kwargs = cfg.kwargs
    model_class = getattr(cfvpy.models, model_name)
    model = model_class(
        num_faces=env_cfg.num_faces, num_dice=env_cfg.num_dice, **kwargs
    )
    if state_dict is not None:
        model.load_state_dict(state_dict)
    if half:
        model = model.half()
    model.to(device)
    if jit:
        model = torch.jit.script(model)
    logging.info("Built a model: %s", model)
    logging.info("Params: %s", [x.dtype for x in model.parameters()])
    return model


class CFVExp:
    def __init__(self, cfg):

        self.cfg = cfg
        self.device = cfg.device or "cuda"
        ckpt_path = "."
        if heyhi.is_on_slurm():
            self.rank = int(os.environ["SLURM_PROCID"])
            self.is_master = self.rank == 0
            n_nodes = int(os.environ["SLURM_JOB_NUM_NODES"])
        else:
            self.rank = 0
            self.is_master = True
            n_nodes = 1
        logging.info(
            "Setup: is_master=%s n_nodes=%s rank=%s ckpt_path=%s",
            self.is_master,
            n_nodes,
            self.rank,
            ckpt_path,
        )

        self.num_actions = cfg.env.num_dice * cfg.env.num_faces * 2 + 1

        self.net = _build_model(self.device, self.cfg.env, self.cfg.model)
        if self.is_master:
            if self.cfg.load_checkpoint:
                logging.info("Loading checkpoint: %s", self.cfg.load_checkpoint)
                self.net.load_state_dict(
                    torch.load(self.cfg.load_checkpoint),
                    strict=not self.cfg.load_checkpoint_loose,
                )
        if self.cfg.selfplay.data_parallel:
            logging.info("data parallel")
            assert self.cfg.selfplay.num_master_threads == 0
            self.net = torch.nn.DataParallel(self.net)
        else:
            logging.info("Single machine mode")

        self.train_timer = cfvpy.utils.MultiStopWatchTimer()

        if cfg.seed:
            logging.info("Setting pytorch random seed to %s", cfg.seed)
            torch.manual_seed(cfg.seed)

    def configure_scheduler(self, optimizer):
        sched_cfg = self.cfg.optimizer.scheduler
        if not sched_cfg:
            return None
        assert not self.cfg.train_policy, "Not supported"
        if "." not in sched_cfg.classname:
            sched_cfg.classname = "torch.optim.lr_scheduler." + sched_cfg.classname
        scheduler = cfvpy.utils.cfg_instantiate(sched_cfg, optimizer)
        logging.info("Scheduler: %s", scheduler)
        return scheduler

    def get_value_params(self):
        if self.cfg.train_policy:
            return self.net.value_net.parameters()
        else:
            return self.net.parameters()

    def get_policy_params(self):
        if self.cfg.train_policy:
            return self.net.policy_net.parameters()
        else:
            return None

    def configure_optimizers(self):
        def build(params):
            if params is None:
                return None
            optim_cfg = self.cfg.optimizer
            if "." not in optim_cfg.classname:
                optim_cfg.classname = "torch.optim." + optim_cfg.classname
            optimizer = cfvpy.utils.cfg_instantiate(optim_cfg, params)
            logging.info("Optimizer: %s", optimizer)
            return optimizer

        optimizer = build(self.get_value_params())
        policy_optimizer = build(self.get_policy_params())
        return optimizer, policy_optimizer

    def loss_func(self, x):
        if self.cfg.loss == "huber":
            return (x.abs() > 1).float() * (x.abs() * 2 - 1) + (
                x.abs() <= 1
            ).float() * x.pow(2)
        elif self.cfg.loss == "mse":
            return x.pow(2)
        else:
            raise ValueError(f"Unknown loss: {self.cfg.loss}")

    def _compute_loss_dict(self, data, device, use_policy_net, timer_prefix=None):
        query = data.query.to(device)
        cf_vals = data.values.to(device, non_blocking=True)
        cf_vals_pred = self.net.forward(query)
        loss_per_example = (self.loss_func(cf_vals - cf_vals_pred)).mean(-1)

        # Shape: scalar.
        loss = loss_per_example.mean()
        losses = {"loss": loss, "partials": {}}

        if timer_prefix:
            self.train_timer.start(f"{timer_prefix}forward-stats")

        action_id = get_last_action_index(data.query, self.num_actions)
        for count in range(self.num_actions + 1):
            mask = action_id == count
            loss_select = loss_per_example[mask]
            val_select = cf_vals[mask]
            if count == self.num_actions:
                count = "initial"
            losses["partials"][count] = {
                "count": mask.long().sum(),
                "loss_sum": loss_select.sum(),
                "val_sum": val_select.sum(),
            }

        return losses

    def run(self):
        return self.run_trainer()

    def get_model(self):
        if hasattr(self.net, "module"):
            return self.net.module
        else:
            return self.net

    def initialize_datagen(self):
        # Need to preserve ownership of the ref models!
        ref_models = []
        model_lockers = []
        assert torch.cuda.device_count() >= 2, torch.cuda.device_count()
        if self.cfg.selfplay.cpu_gen_threads:
            num_threads = self.cfg.selfplay.cpu_gen_threads
            act_devices = ["cpu"] * num_threads
            logging.info("Will generate on CPU with %d threads", num_threads)
            assert self.cfg.selfplay.models_per_gpu == 1
        else:
            act_devices = [f"cuda:{i}" for i in range(1, torch.cuda.device_count())]
            if self.is_master and self.cfg.selfplay.num_master_threads is not None:
                num_threads = self.cfg.selfplay.num_master_threads
            else:
                num_threads = self.cfg.selfplay.threads_per_gpu * len(act_devices)

        # Don't need mode deviced than threads.
        act_devices = act_devices[:num_threads]

        logging.info("Gpus for generations: %s", act_devices)
        logging.info("Threads: %s", num_threads)
        for act_device in act_devices:
            ref_model = [
                _build_model(
                    act_device,
                    self.cfg.env,
                    self.cfg.model,
                    self.get_model().state_dict(),
                    half=self.cfg.half_inference,
                    jit=True,
                )
                for _ in range(self.cfg.selfplay.models_per_gpu)
            ]
            for model in ref_models:
                model.eval()
            ref_models.extend(ref_model)
            model_locker = cfvpy.rela.ModelLocker(ref_model, act_device)
            model_lockers.append(model_locker)

        replay_params = dict(
            capacity=2 ** 20,
            seed=10001 + self.rank,
            alpha=1.0,
            beta=0.4,
            prefetch=3,
            use_priority=True,
        )
        if self.cfg.replay:
            replay_params.update(self.cfg.replay)
        logging.info("Replay params (per buffer): %s", replay_params)
        replay = cfvpy.rela.ValuePrioritizedReplay(
            **replay_params, compressed_values=False
        )
        if self.cfg.train_policy:
            policy_replay = cfvpy.rela.ValuePrioritizedReplay(
                **replay_params, compressed_values=self.cfg.compress_policy_values
            )
        else:
            policy_replay = None

        context = cfvpy.utils.TimedContext()
        cfr_cfg = create_mdp_config(self.cfg.env)
        for i in range(num_threads):
            thread = cfvpy.rela.create_cfr_thread(
                model_lockers[i % len(model_lockers)],
                replay,
                cfr_cfg,
                self.rank * 1000 + i,
            )
            context.push_env_thread(thread)

        return dict(
            ref_models=ref_models,
            model_lockers=model_lockers,
            replay=replay,
            policy_replay=policy_replay,
            context=context,
        )

    def run_trainer(self):
        # Fix version so that training always continues.
        if self.is_master:
            logger = pl_logging.TestTubeLogger(save_dir=os.getcwd(), version=0)

        # Storing the whole dict to preserve ref_models.
        datagen = self.initialize_datagen()
        context = datagen["context"]
        replay = datagen["replay"]
        policy_replay = datagen["policy_replay"]

        if self.cfg.data.train_preload:
            # Must preload data before starting generators to avoid deadlocks.
            _preload_data(self.cfg.data.train_preload, replay)
            preloaded_size = replay.size()
        else:
            preloaded_size = 0

        self.opt, self.policy_opt = self.configure_optimizers()
        self.scheduler = self.configure_scheduler(self.opt)

        context.start()

        if self.cfg.benchmark_data_gen:
            # Benchmark generation speed and exit.
            time.sleep(self.cfg.benchmark_data_gen)
            context.terminate()
            size = replay.num_add()
            logging.info(
                "BENCHMARK size %s speed %.2f", size, size / context.running_time
            )
            return

        train_size = self.cfg.data.train_epoch_size or 128 * 1000
        logging.info("Train set size (forced): %s", train_size)

        assert self.cfg.data.train_batch_size
        batch_size = self.cfg.data.train_batch_size
        epoch_size = train_size // batch_size

        if self.is_master:
            val_datasets = []

        logging.info(
            "model size is %s",
            sum(p.numel() for p in self.net.parameters() if p.requires_grad),
        )
        save_dir = pathlib.Path("ckpt")
        if self.is_master and not save_dir.exists():
            logging.info(f"Creating savedir: {save_dir}")
            save_dir.mkdir(parents=True)

        burn_in_frames = batch_size * 2
        while replay.size() < burn_in_frames or (
            policy_replay is not None and policy_replay.size() < burn_in_frames
        ):
            logging.info(
                "warming up replay buffer: %d/%d", replay.size(), burn_in_frames
            )
            if policy_replay is not None:
                logging.info(
                    "warming up POLICY replay buffer: %d/%d",
                    policy_replay.size(),
                    burn_in_frames,
                )
            time.sleep(30)

        def compute_gen_bps():
            return (
                (replay.num_add() - preloaded_size) / context.running_time / batch_size
            )

        def compute_gen_bps_policy():
            return policy_replay.num_add() / context.running_time / batch_size

        metrics = None
        num_decays = 0
        for epoch in range(self.cfg.max_epochs):
            self.train_timer.start("start")
            if (
                epoch % self.cfg.decrease_lr_every == self.cfg.decrease_lr_every - 1
                and self.scheduler is None
            ):
                if (
                    not self.cfg.decrease_lr_times
                    or num_decays < self.cfg.decrease_lr_times
                ):
                    for param_group in self.opt.param_groups:
                        param_group["lr"] /= 2
                    num_decays += 1
            if (
                self.cfg.create_validation_set_every
                and self.is_master
                and epoch % self.cfg.create_validation_set_every == 0
            ):
                logging.info("Adding new validation set")
                val_batches = [
                    replay.sample(batch_size, "cpu")[0]
                    for _ in range(512 * 100 // batch_size)
                ]
                val_datasets.append((f"valid_snapshot_{epoch:04d}", val_batches))

            if (
                self.cfg.selfplay.dump_dataset_every_epochs
                and epoch % self.cfg.selfplay.dump_dataset_every_epochs == 0
                and (not self.cfg.data.train_preload or epoch > 0)
            ):
                dataset_folder = pathlib.Path("dumped_data").resolve()
                dataset_folder.mkdir(exist_ok=True, parents=True)
                dataset_path = dataset_folder / f"data_{epoch:03d}.dat"
                logging.info(
                    "Saving replay buffer as supervised dataset to %s", dataset_path
                )
                replay.save(str(dataset_path))

            metrics = {}
            metrics["optim/lr"] = next(iter(self.opt.param_groups))["lr"]
            metrics["epoch"] = epoch
            counters = collections.defaultdict(cfvpy.utils.FractionCounter)
            if self.cfg.grad_clip:
                counters["optim/grad_max"] = cfvpy.utils.MaxCounter()
                if self.cfg.train_policy:
                    counters["optim_policy/grad_max"] = cfvpy.utils.MaxCounter()
            use_progress_bar = not heyhi.is_on_slurm() or self.cfg.show_progress_bar
            train_loader = range(epoch_size)
            train_device = self.device
            train_iter = tqdm.tqdm(train_loader) if use_progress_bar else train_loader
            training_start = time.time()

            if self.cfg.train_gen_ratio:
                while True:
                    if replay.num_add() * self.cfg.train_gen_ratio >= train_size * (
                        epoch + 1
                    ):
                        break
                    logging.info(
                        "Throttling to satisfy |replay| * ratio >= train_size * epochs:"
                        " %s * %s >= %s %s",
                        replay.num_add(),
                        self.cfg.train_gen_ratio,
                        train_size,
                        epoch + 1,
                    )
                    time.sleep(60)
            assert self.cfg.replay.use_priority is False, "Not supported"

            value_loss = policy_loss = 0  # For progress bar.
            for iter_id in train_iter:
                self.train_timer.start("train-get_batch")
                use_policy_net = iter_id % 2 and policy_replay is not None
                if use_policy_net:
                    batch, _ = policy_replay.sample(batch_size, train_device)
                    suffix = "_policy"
                else:
                    batch, _ = replay.sample(batch_size, train_device)
                    suffix = ""
                self.train_timer.start("train-forward")
                self.net.train()
                loss_dict = self._compute_loss_dict(
                    batch, train_device, use_policy_net, timer_prefix="train-"
                )
                self.train_timer.start("train-backward")
                loss = loss_dict["loss"]
                opt = self.policy_opt if use_policy_net else self.opt
                params = (
                    self.get_policy_params()
                    if use_policy_net
                    else self.get_value_params()
                )
                opt.zero_grad()
                loss.backward()

                if self.cfg.grad_clip:
                    g_norm = clip_grad_norm_(params, self.cfg.grad_clip)
                else:
                    g_norm = None
                opt.step()
                loss.item()  # Force sync.
                self.train_timer.start("train-rest")
                if g_norm is not None:
                    g_norm = g_norm.item()
                    counters[f"optim{suffix}/grad_max"].update(g_norm)
                    counters[f"optim{suffix}/grad_mean"].update(g_norm)
                    counters[f"optim{suffix}/grad_clip_ratio"].update(
                        int(g_norm >= self.cfg.grad_clip - 1e-5)
                    )
                counters[f"loss{suffix}/train"].update(loss)
                for num_cards, partial_data in loss_dict["partials"].items():
                    counters[f"loss{suffix}/train_{num_cards}"].update(
                        partial_data["loss_sum"], partial_data["count"],
                    )
                    counters[f"val{suffix}/train_{num_cards}"].update(
                        partial_data["val_sum"], partial_data["count"],
                    )
                    counters[f"shares{suffix}/train_{num_cards}"].update(
                        partial_data["count"], batch_size
                    )

                if use_progress_bar:
                    if use_policy_net:
                        policy_loss = loss.detach().item()
                    else:
                        value_loss = loss.detach().item()
                    pbar_fields = dict(
                        policy_loss=policy_loss,
                        value_loss=value_loss,
                        buffer_size=replay.size(),
                        gen_bps=compute_gen_bps(),
                    )
                    if policy_replay is not None:
                        pbar_fields["pol_buffer_size"] = policy_replay.size()
                    train_iter.set_postfix(**pbar_fields)
                if self.cfg.fake_training:
                    # Generation benchmarking mode in which training is
                    # skipped. The goal is to measure generation speed withot
                    # sample() calls..
                    break
            if self.cfg.fake_training:
                # Fake training epoch takes a minute.
                time.sleep(60)

            if len(train_loader) > 0:
                metrics["bps/train"] = len(train_loader) / (
                    time.time() - training_start
                )
                metrics["bps/train_examples"] = metrics["bps/train"] * batch_size
            logging.info(
                "[Train] epoch %d complete, avg error is %f",
                epoch,
                counters["loss/train"].value(),
            )
            if self.scheduler is not None:
                self.scheduler.step()
            for name, counter in counters.items():
                metrics[name] = counter.value()
            metrics["buffer/size"] = replay.size()
            metrics["buffer/added"] = replay.num_add()
            metrics["bps/gen"] = compute_gen_bps()
            metrics["bps/gen_examples"] = metrics["bps/gen"] * batch_size
            if policy_replay is not None:
                metrics["buffer/policy_size"] = policy_replay.size()
                metrics["buffer/policy_added"] = policy_replay.num_add()
                metrics["bps/gen_policy"] = compute_gen_bps_policy()
                metrics["bps/gen_policy_examples"] = (
                    metrics["bps/gen_policy"] * batch_size
                )

            if (epoch + 1) % self.cfg.selfplay.network_sync_epochs == 0 or epoch < 15:
                logging.info("Copying current network to the eval network")
                for model_locker in datagen["model_lockers"]:
                    model_locker.update_model(self.get_model())
            if self.cfg.purging_epochs and (epoch + 1) in self.cfg.purging_epochs:
                new_size = max(
                    burn_in_frames,
                    int((self.cfg.purging_share_keep or 0.0) * replay.size()),
                )
                logging.info(
                    "Going to purge everything but %d elements in the buffer", new_size,
                )
                replay.pop_until(new_size)

            if self.is_master and epoch % 10 == 0:
                with torch.no_grad():
                    for i, (name, val_loader) in enumerate(val_datasets):
                        self.train_timer.start("valid-acc-extra")
                        eval_errors = []
                        val_iter = (
                            tqdm.tqdm(val_loader, desc="Eval")
                            if use_progress_bar
                            else val_loader
                        )
                        for data in val_iter:
                            self.net.eval()
                            loss = self._compute_loss_dict(
                                data, train_device, use_policy_net=False
                            )["loss"]
                            eval_errors.append(loss.detach().item())
                        current_error = sum(eval_errors) / len(eval_errors)
                        logging.info(
                            "[Eval] epoch %d complete, data is %s, avg error is %f",
                            epoch,
                            name,
                            current_error,
                        )
                        metrics[f"loss/{name}"] = current_error

                self.train_timer.start("valid-trace")
                ckpt_path = save_dir / f"epoch{epoch}.ckpt"
                torch.save(self.get_model().state_dict(), ckpt_path)
                bin_path = ckpt_path.with_suffix(".torchscript")
                torch.jit.save(torch.jit.script(self.get_model()), str(bin_path))

                self.train_timer.start("valid-exploit")
                if self.cfg.exploit and epoch % 20 == 0:
                    bin_path = pathlib.Path("tmp.torchscript")
                    torch.jit.save(torch.jit.script(self.get_model()), str(bin_path))
                    (
                        exploitability,
                        mse_net_traverse,
                        mse_fp_traverse,
                    ) = cfvpy.rela.compute_stats_with_net(
                        create_mdp_config(self.cfg.env), str(bin_path)
                    )
                    logging.info(
                        "Exploitability to leaf (epoch=%d): %.2f", epoch, exploitability
                    )
                    metrics["exploitability_last"] = exploitability
                    metrics["eval_mse/net_reach"] = mse_net_traverse
                    metrics["eval_mse/fp_reach"] = mse_fp_traverse

            if len(train_loader) > 0:
                metrics["bps/loop"] = len(train_loader) / (time.time() - training_start)
            total = 1e-5
            for k, v in self.train_timer.timings.items():
                metrics[f"timing/{k}"] = v / (epoch + 1)
                total += v
            for k, v in self.train_timer.timings.items():
                metrics[f"timing_pct/{k}"] = v * 100 / total
            logging.info("Metrics: %s", metrics)
            if self.is_master:
                logger.log_metrics(metrics)
                logger.save()
        return metrics


def create_mdp_config(cfr_yaml_cfg):
    cfg_dict: dict
    if cfr_yaml_cfg is None:
        cfg_dict = {}
    else:
        cfg_dict = dict(cfr_yaml_cfg)
    logging.info(
        "Using the following kwargs to create RecursiveSolvingParams: %s", cfr_yaml_cfg
    )

    def recusive_set(cfg, cfg_dict):
        for key, value in cfg_dict.items():
            if not hasattr(cfg, key):
                raise RuntimeError(
                    f"Cannot find key {key} in {cfg}. It's either not definied"
                    " or not imposed via pybind11"
                )
            if isinstance(value, (dict, omegaconf.dictconfig.DictConfig)):
                recusive_set(getattr(cfg, key), value)
            else:
                setattr(cfg, key, value)
        return cfg

    return recusive_set(cfvpy.rela.RecursiveSolvingParams(), cfg_dict)


def _preload_data(cfg_preload, replay):
    """Load supervised dataset into the replay buffer."""
    logging.info("Going to preload data from %s to the buffer", cfg_preload.path)
    replay.load(
        cfg_preload.path,
        cfg_preload.priority or 1.0,
        cfg_preload.max_size or -1,
        cfg_preload.stride or 1,
    )


def get_last_action_index(query, num_actions):
    with torch.no_grad():
        action_one_hot = torch.cat(
            [
                query[:, 2 : 2 + num_actions],
                torch.full((len(query), 1), 0.1, device=query.device),
            ],
            -1,
        )
        return action_one_hot.max(-1).indices


def clip_grad_norm_(parameters, max_norm, norm_type=2):
    """Copied from Pytorch 1.5. Faster version for grad norm."""
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]),
        norm_type,
    )
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.detach().mul_(clip_coef)
    return total_norm
