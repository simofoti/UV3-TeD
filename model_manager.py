import os
import torch
import torch.profiler
import torch_geometric.data

import mitsuba as mi

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup

import utils
import rendering

from network import DiffusionNet


class ModelManager(torch.nn.Module):
    def __init__(self, config: dict, len_train_loader: int | None = None):
        super(ModelManager, self).__init__()
        # create dummy params to always get correct device
        self.__dummy_param = torch.nn.Parameter(torch.empty(0))

        self._num_train_timesteps = config["num_train_timesteps"]
        self._noise_scheduler = DDPMScheduler(self._num_train_timesteps)

        net_config = config["net"]
        self._net = DiffusionNet(
            in_channels=net_config["channels"]["net_in"],
            out_channels=net_config["channels"]["net_out"],
            io_mlp_channels=net_config["channels"]["blocks_mlp_io"],
            attention_inner_channels=net_config["channels"]["attention"],
            blocks_depth=net_config["blocks_depth"],
            last_activation=net_config["last_activation"],
            mlp_hidden_dims=net_config["channels"]["blocks_mlp_intermediate"],
            dropout=net_config["dropout"],
            time_freq_shift=net_config["time_frequency_shift"],
            time_flip_sin_to_cos=net_config["time_flip_sin_to_cos"],
            k_eig=config["data"]["transforms_config"]["eigen_number"],
            n_hks=utils.in_or_default(
                config["data"]["transforms_config"], "hks_number", 0
            ),
        )

        self._optimizer = torch.optim.AdamW(
            self._net.parameters(), lr=float(config["lr"])
        )

        if len_train_loader is not None and config["lr_warmup_steps"] > 0:
            self._lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=self._optimizer,
                num_warmup_steps=config["lr_warmup_steps"],
                num_training_steps=(len_train_loader * config["epochs"]),
            )
        else:
            print(
                "No learning rate scheduler is going to be used. If you want",
                "to use 'cosine_schedule_with_warmup' set 'len_train_loader'",
                "when initialising the ModelManager and 'lr_warmup_steps' > 0",
                "in the config file.",
            )
            self._lr_scheduler = None

        self._epoch_loss = 0
        self._logger = None
        self._shapes_for_logging = None
        self._rendering_config = config["rendering"]
        self._scene_dict = self._set_rendering_scene_dict()
        all_transforms = (
            config["data"]["pre_transforms_list"]
            + config["data"]["transforms_list"]
        )
        self._render_pcl_texure = any(["sample" in t for t in all_transforms])
        self._profiler = None

    @property
    def device(self):
        # with DataParallel the device can be accessed with -> *.module.device
        return self.__dummy_param.device

    def forward(self, data: torch_geometric.data.Data):
        return self.generate(data)

    @torch.no_grad()
    def generate(
        self,
        data: torch_geometric.data.Data,
        to_01: bool = True,
        nest_bar: bool = False,
        bar_msg: str = "",
    ) -> torch_geometric.data.Data:
        data = data.to(self.device)
        data.x = torch.randn_like(data.x)

        for t in tqdm(
            self._noise_scheduler.timesteps,
            desc="Generating" + bar_msg + ": ",
            position=1 if nest_bar else None,
            leave=not nest_bar,
        ):
            if "to_padded_mask" in data.keys():
                t = t.expand(data.num_graphs)
            data.x = self._denoising_step(data, t.to(self.device))
        if to_01:
            data.x = ((data.x / 2) + 0.5).clamp(0, 1)
        return data

    @torch.no_grad()
    def _denoising_step(
        self, data: torch_geometric.data.Data, timestep: torch.IntTensor
    ) -> torch.Tensor:
        self._net.eval()

        # Predict noise model_output
        model_output = self._net(data, timestep).sample

        if "to_padded_mask" in data.keys():
            timestep = timestep[0]
        # Compute previous sample: x_t -> x_t-1
        previous_sample = self._noise_scheduler.step(
            model_output, timestep, data.x, generator=None
        ).prev_sample
        return previous_sample

    def _noising_step(
        self, data: torch_geometric.data.Data, timestep: torch.IntTensor
    ) -> torch.Tensor:
        noise = torch.randn_like(data.x)
        return self._noise_scheduler.add_noise(data.x, noise, timestep)

    def run_epoch(self, data_loader, train: bool = True, nest_bar: bool = True):
        if train:
            self._net.train()
            bar_desc = "Train iterations: "
        else:
            self._net.eval()
            bar_desc = "Val iterations: "

        b_p = 1 if nest_bar else None  # bar position
        b_l = not nest_bar  # leave bar when completed
        # If the bar has to be nested, set position=0 in outer loop

        it, avg_loss = 0, 0
        with tqdm(data_loader, desc=bar_desc, position=b_p, leave=b_l) as bar:
            for it, data in enumerate(bar):
                if train:
                    loss = self._run_iteration(data, train=True)
                else:
                    with torch.no_grad():
                        loss = self._run_iteration(data, train=False)
                bar.set_postfix(loss=loss)
                avg_loss += loss
        avg_loss /= it + 1
        self._epoch_loss = avg_loss

    def _run_iteration(
        self, data: torch_geometric.data.Data, train: bool = True
    ) -> float:
        data = data.detach().to(self.device)

        b = (
            data.num_graphs
            if "to_padded_mask" in data.keys()
            else data.x.shape[0]
        )

        # Sample a random timestep for each sample in the batch
        timesteps = torch.randint(
            low=0,
            high=self._noise_scheduler.config.num_train_timesteps,
            size=(b,),
            device=data.x.device,
        ).long()

        # Add noise to the clean samples according to the noise magnitude
        # at each timestep (this is the forward diffusion process)
        if "to_padded_mask" in data.keys():
            gt_col = utils.packed_to_padded(
                data.x, data.to_padded_mask, data.max_verts, data.num_graphs
            )

            noise = torch.randn_like(gt_col)
            data.x = self._noise_scheduler.add_noise(gt_col, noise, timesteps)

            noise = utils.padded_to_packed(noise, data.to_packed_idx)
            data.x = utils.padded_to_packed(data.x, data.to_packed_idx)
        else:
            gt_col = data.x.clone()
            noise = torch.randn_like(gt_col)
            data.x = self._noise_scheduler.add_noise(gt_col, noise, timesteps)

        # Predict the noise residual
        noise_pred = self._net(data, timesteps).sample
        mse_loss = torch.nn.functional.mse_loss(noise_pred, noise)

        # Clamp the loss to avoid exploding gradients
        # gradients when the loss is too high should go to zero, as the
        # threshold is quite high this should not affect the training
        mse_loss = torch.clamp(mse_loss, max=10_000)

        if train:
            mse_loss.backward()
            torch.nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
            self._optimizer.step()
            if self._lr_scheduler is not None:
                self._lr_scheduler.step()
            self._optimizer.zero_grad()
        return mse_loss.item()

    def _set_rendering_scene_dict(
        self, rend_config: dict | None = None
    ) -> dict:
        if rend_config is None:
            rend_config = self._rendering_config

        camera_config = rend_config["camera"]
        return {
            "type": "scene",
            "integrator": rendering.define_integrator(),
            "camera": rendering.define_camera(
                camera_config["distance"],
                camera_config["azimuth_deg"],
                camera_config["elevation_deg"],
                camera_config["camera_type"],
                camera_config["img_width"],
                camera_config["img_height"],
                camera_config["sampler_type"],
                camera_config["sample_count"],
                camera_config["fov"],
            ),
            "emitter": rendering.define_emitter(
                rend_config["emitter"]["envmap_path"]
            ),
        }

    def reset_rendering_camera_params(self):
        self._scene_dict = self._set_rendering_scene_dict()

    def change_rendering_camera_param(self, param_name, param_value):
        rend_config = self._rendering_config.copy()
        rend_config["camera"][param_name] = param_value
        self._scene_dict = self._set_rendering_scene_dict(rend_config)

    def render(self, data: torch_geometric.data.Data):
        scene_dict = self._scene_dict.copy()
        sided = utils.in_or_default(self._rendering_config, "twosided", False)
        if self._render_pcl_texure:
            scene_dict["mesh"] = rendering.data_coloured_points_to_mitsuba(
                data, twosided=sided
            )
        else:  # render with vertex colours
            scene_dict["mesh"] = rendering.data_coloured_verts_to_mitsuba(
                data, twosided=sided
            )
        scene = mi.load_dict(scene_dict)
        return torch.Tensor(mi.render(scene))

    def run_profiling_epoch(self, data_loader):
        assert self._profiler is not None
        avg_loss = 0
        self._profiler.start()
        for it, data in enumerate(data_loader):
            if it >= 200:
                break
            loss = self._run_iteration(data, train=True)
            avg_loss += loss
            self._profiler.step()
        print(avg_loss / (it + 1))
        self._profiler.stop()

    def create_profiler(self, log_dir: str):
        self._profiler = torch.profiler.profile(
            schedule=torch.profiler.schedule(
                wait=1, warmup=1, active=3, repeat=2
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
            record_shapes=True,
            with_stack=True,
            profile_memory=True,
        )

    def create_tb_logger(self, output_directory: str):
        self._logger = SummaryWriter(os.path.join(output_directory, "logs"))

    def log_loss(self, epoch: int, phase: str = "train"):
        self._logger.add_scalar(phase + "/mse", self._epoch_loss, epoch + 1)

    def pick_shapes_for_logging(
        self, dict_loaders: dict, number_of_shapes: int = 5
    ):
        data = next(iter(dict_loaders["train"]))
        batch_size = data.num_graphs if "to_padded_mask" in data.keys() else 1

        shapes_for_logging = {}
        for k, loader in dict_loaders.items():
            shapes_for_logging[k] = []
            for i, data in enumerate(loader):
                shapes_for_logging[k].append(data)
                if (i + 1) * batch_size >= number_of_shapes:
                    break
        self._shapes_for_logging = shapes_for_logging

    def get_shapes_for_logging(self, phase: str | None = None):
        if phase is None:
            shapes = self._shapes_for_logging
        else:
            shapes = self._shapes_for_logging[phase]
        return shapes

    def log_generated_images(
        self,
        epoch: int,
        phase: str = "train",
        n_variants: int = 3,
    ):
        assert self._shapes_for_logging is not None
        n_shapes_to_render = len(self._shapes_for_logging[phase]) * n_variants
        render_counter = 0

        rendering.mega_kernel(state=False)

        all_renders = []
        for data in self._shapes_for_logging[phase]:
            current_shape_renders = []
            for _ in range(n_variants):
                render_counter += 1
                bar_msg = f" ({str(render_counter)}/{str(n_shapes_to_render)})"
                data = self.generate(
                    data, to_01=False, nest_bar=True, bar_msg=bar_msg
                )
                if "to_padded_mask" in data.keys():
                    data_r = data.clone()
                    data_r.grad_x, data_r.grad_y = None, None
                    batch_renders = []
                    for i in range(data_r.num_graphs):
                        batch_renders.append(
                            self.render(data_r.get_example(i)).detach().cpu()
                        )
                    current_shape_renders.append(
                        torch.cat(batch_renders, dim=-2)
                    )
                else:
                    current_shape_renders.append(
                        self.render(data.clone()).detach().cpu()
                    )
            all_renders.append(torch.cat(current_shape_renders, dim=-3))
        log_images = torch.cat(all_renders, dim=-2).permute(2, 0, 1)
        self._logger.add_image(
            tag=phase, img_tensor=log_images, global_step=epoch + 1
        )
        rendering.flush_cache()

    def save(self, checkpoint_dir: str, epoch: int):
        net_name = os.path.join(checkpoint_dir, "model_%08d.pt" % (epoch + 1))
        torch.save({"model": self._net.state_dict()}, net_name)

        opt_name = os.path.join(checkpoint_dir, "optimizer.pt")
        opt = {"optimizer": self._optimizer.state_dict()}
        if self._lr_scheduler is not None:
            opt["scheduler"] = self._lr_scheduler.state_dict()
        torch.save(opt, opt_name)

    def resume(self, checkpoint_dir: str) -> int:
        last_model_name = utils.get_model_list(checkpoint_dir, "model")
        state_dict = torch.load(last_model_name)
        self._net.load_state_dict(state_dict["model"])
        epochs = int(last_model_name[-11:-3])
        opt_dict = torch.load(os.path.join(checkpoint_dir, "optimizer.pt"))
        self._optimizer.load_state_dict(opt_dict["optimizer"])
        if self._lr_scheduler is not None:
            self._lr_scheduler.load_state_dict(opt_dict["scheduler"])
        print(f"Resuming from epoch {epochs}")
        return epochs
