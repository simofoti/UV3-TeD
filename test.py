import os
import torch
import drjit
import mitsuba as mi
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from cleanfid import fid

import utils
import rendering
from transforms import get_transforms


class ModelTester:
    def __init__(
        self,
        model_manager,
        data_loaders,
        configs,
        device,
        output_directory,
        phase="val",
    ) -> None:
        self._manager = model_manager
        self._loaders = data_loaders
        self._configs = configs
        self._device = device
        self._output_directory = output_directory
        self._sample_count_rendering = 48

        assert phase in ["val", "test"]
        self._phase = phase

    def __call__(self) -> None:
        self.render_learned_heat_diffusion_times(
            param_name="diffusion", n_heat_sources=8
        )

        # self.render_more_gen_textures(n_shapes=8, n_variants=8)

        self.render_gt(num_views=5)

        for n in range(3):
            self.generate_pcltex_for_whole_set(run_n=n)
            self.render_pregenerated_pcltex(run_n=n, num_views=1)
            self.render_pregenerated_pcltex(run_n=n, num_views=5)

        self.compute_fid_and_kid(
            os.path.join(self._output_directory, "rendered_gts"),
            os.path.join(self._output_directory, "rendered_all_multiview_0"),
            "proposed",
        )

        # p = self._output_directory
        # self.parse_lpips_files_and_save_average(
        #     [os.path.join(p, f) for f in ("1vs2.txt", "2vs3.txt", "1vs3.txt")]
        # )

    def change_sampling_density(self, n_poisson_samples: int = 20_000):
        transforms_configs = self._configs["data"]["transforms_config"]
        if "n_poisson_samples" in transforms_configs:
            transforms_configs["n_poisson_samples"] = n_poisson_samples

        transforms = get_transforms(
            list_transforms=self._configs["data"]["transforms_list"],
            transforms_config=transforms_configs,
            root=self._configs["data"]["root"],
        )
        self._loaders[self._phase].dataset.transform = transforms

    def render_more_gen_textures(self, n_shapes: int, n_variants: int) -> None:
        self._manager.change_rendering_camera_param(
            "sample_count", self._sample_count_rendering
        )
        self._manager.pick_shapes_for_logging(self._loaders, n_shapes)
        shapes = self._manager.get_shapes_for_logging(self._phase)

        n_shapes_to_render = len(shapes) * n_variants
        render_counter = 0

        rendering.mega_kernel(state=False)

        # Add also GTs for reference
        gt_renders = []
        for data in shapes:
            if "to_padded_mask" in data.keys():
                for i in range(data.num_graphs):
                    gt_renders.append(
                        self._manager.render(data.get_example(i)).detach().cpu()
                    )
            else:
                gt_renders.append(
                    self._manager.render(data.clone()).detach().cpu()
                )

        all_renders = [gt_renders]
        for _ in range(n_variants):
            current_variant_renders = []
            for data in shapes:
                render_counter += 1
                bar_msg = f" ({str(render_counter)}/{str(n_shapes_to_render)})"
                data = self._manager.generate(
                    data, to_01=False, nest_bar=True, bar_msg=bar_msg
                )
                if "to_padded_mask" in data.keys():
                    data_r = data.clone()
                    data_r.grad_x, data_r.grad_y = None, None
                    for i in range(data_r.num_graphs):
                        current_variant_renders.append(
                            self._manager.render(data_r.get_example(i))
                            .detach()
                            .cpu()
                        )
                else:
                    current_variant_renders.append(
                        self._manager.render(data.clone()).detach().cpu()
                    )
            all_renders.append(current_variant_renders)
            rendering.flush_cache()

        out_img_dir = os.path.join(
            self._output_directory, "renderings_" + self._phase
        )
        utils.mkdir(out_img_dir)

        all_renders_transposed = list(map(list, zip(*all_renders)))
        for i, sh_variants in enumerate(all_renders_transposed):
            im = torch.cat(sh_variants, dim=-2)
            mi.util.write_bitmap(os.path.join(out_img_dir, str(i) + ".png"), im)

    def generate_pcltex_for_whole_set(self, run_n: int | None = None):
        out_dir = os.path.join(self._output_directory, "generated")
        if run_n is not None:
            out_dir = out_dir + f"_{str(run_n)}"
        utils.mkdir(out_dir)

        def _save_pcl_tex(d):
            m = utils.load_mesh(
                os.path.join(self._configs["data"]["root"], d.raw_path)
            )
            path = os.path.join(out_dir, d.raw_path.split("/")[1] + ".ply")

            if "verts" in d.keys():
                verts = d.verts
            else:
                torch.Tensor(m.vertices)

            utils.save_pcltex(
                verts,
                torch.Tensor(m.faces.T),
                d.pos,
                d.x,
                path,
                save_also_mesh=True,
            )

        for data in tqdm(self._loaders[self._phase], desc="Generating all"):
            data = self._manager.generate(data, to_01=False, nest_bar=True)
            if "to_padded_mask" in data.keys():
                data_r = data.clone()
                data_r.grad_x, data_r.grad_y = None, None
                for i in range(data_r.num_graphs):
                    d = data_r.get_example(i)
                    _save_pcl_tex(d)
            else:
                _save_pcl_tex(data)

    def render_pregenerated_pcltex(
        self,
        dir_path=None,
        run_n=None,
        normalise_scale=False,
        num_views=1,
    ):
        if dir_path is None and run_n is None:
            dir_path = os.path.join(self._output_directory, "generated")
        elif run_n is not None:
            dir_path = os.path.join(
                self._output_directory, f"generated_{run_n}"
            )

        if num_views > 1:
            dir_name = "rendered_all_multiview"
        else:
            dir_name = "rendered_all"

        out_dir = dir_path.replace("generated", dir_name)
        utils.mkdir(out_dir)

        all_pclt_paths = []
        for root, _, files in os.walk(dir_path):
            for fn in files:
                if fn.endswith(".ply") and "mesh" not in fn:
                    all_pclt_paths.append(os.path.join(root, fn))

        rendering.mega_kernel(state=False)
        for pcl_fpath in tqdm(all_pclt_paths, desc="Rendering all"):
            mesh_fpath = pcl_fpath.replace(".ply", "_mesh.ply")
            data = utils.load_mesh_with_pcltex(pcl_fpath, mesh_fpath)
            mi_mesh = rendering.mesh_with_pcltex_to_mitsuba(
                data, normalise_scale, twosided=True
            )
            if num_views == 1:
                images = [self._render_single_view(mi_mesh)]
            else:
                images = self._render_random_views(mi_mesh, num_views)

            for i, im in enumerate(images):
                f_path = pcl_fpath.replace("generated", dir_name)
                f_path = f_path.replace(".ply", f"_{str(i)}.png")
                mi.util.write_bitmap(f_path, im)
        rendering.flush_cache()
    def render_gt(self, num_views=5):
        out_dir = os.path.join(self._output_directory, "rendered_gts")
        utils.mkdir(out_dir)

        def _render_and_save(d):
            mi_mesh = rendering.data_original_texture_to_mitsuba(
                d, merge_tex=False, twosided=True, normalize_scale=False
            )

            images = self._render_random_views(mi_mesh, num_views)
            for i, im in enumerate(images):
                f_path = d.raw_path.split("/")[1] + f"_{str(i)}.png"
                f_path = os.path.join(out_dir, f_path)
                mi.util.write_bitmap(f_path, im)

        for data in tqdm(self._loaders[self._phase], desc="Rendering GTs"):
            if "to_padded_mask" in data.keys():
                for i in range(data.num_graphs):
                    d = data.get_example(i)
                    _render_and_save(d)
            else:
                _render_and_save(data)
    def _render_random_views(self, mi_mesh, view_num=5, denoise=False):
        camera_config = self._configs["rendering"]["camera"]
        rend_config = self._configs["rendering"]
        scene_dict = {
            "type": "scene",
            "integrator": rendering.define_integrator(),
            "emitter": rendering.define_emitter(
                rend_config["emitter"]["envmap_path"]
            ),
            "mesh": mi_mesh,
        }
        scene = mi.load_dict(scene_dict)

        if denoise:
            denoiser = mi.OptixDenoiser(
                input_size=[
                    camera_config["img_width"],
                    camera_config["img_height"],
                ]
            )

        azimuth_angles = np.random.rand(view_num) * 360
        elevation_angles = np.random.rand(view_num) * 60

        images = []
        for azimuth, elevation in zip(azimuth_angles, elevation_angles):
            camera = mi.load_dict(
                rendering.define_camera(
                    camera_config["distance"],
                    azimuth,
                    elevation,
                    camera_config["camera_type"],
                    camera_config["img_width"],
                    camera_config["img_height"],
                    camera_config["sampler_type"],
                    self._sample_count_rendering,
                    camera_config["fov"],
                )
            )
            with drjit.suspend_grad():
                img = mi.render(scene, sensor=camera)
            if denoise:
                img = denoiser(img)
            images.append(torch.Tensor(img))
        return images

    def _render_single_view(self, mi_mesh, denoise=False):
        camera_config = self._configs["rendering"]["camera"]
        rend_config = self._configs["rendering"]
        scene_dict = {
            "type": "scene",
            "integrator": rendering.define_integrator(),
            "emitter": rendering.define_emitter(
                rend_config["emitter"]["envmap_path"]
            ),
            "mesh": mi_mesh,
        }
        scene = mi.load_dict(scene_dict)

        camera = mi.load_dict(
            rendering.define_camera(
                camera_config["distance"],
                camera_config["azimuth_deg"],
                camera_config["elevation_deg"],
                camera_config["camera_type"],
                camera_config["img_width"],
                camera_config["img_height"],
                camera_config["sampler_type"],
                self._sample_count_rendering,
                camera_config["fov"],
            )
        )

        with drjit.suspend_grad():
            image = torch.Tensor(mi.render(scene, sensor=camera))

        if denoise:
            denoiser = mi.OptixDenoiser(
                input_size=[
                    camera_config["img_width"],
                    camera_config["img_height"],
                ]
            )
            image = denoiser(image)
        return image
    def compute_fid_and_kid(self, gt_img_save_dir, gen_img_save_dir, id):
        score_fid = fid.compute_fid(gt_img_save_dir, gen_img_save_dir)
        score_kid = fid.compute_kid(gt_img_save_dir, gen_img_save_dir)
        print(f"FID Score: {score_fid}, KID Score: {score_kid}")

        # Save the scores in a file
        fname = f"scores_{id}.txt"
        with open(os.path.join(self._output_directory, fname), "w") as f:
            f.write(f"FID Score: {score_fid}\n")
            f.write(f"KID Score: {score_kid}\n")

    def parse_lpips_files_and_save_average(self, lst_files):
        """
        Lpips can be used launching a simple script from the LPIPS library.
        Instructions are provided in the README.md file.
        """
        all_lpips = []
        for f in lst_files:
            with open(f, "r") as fp:
                lines = fp.readlines()
                for line in lines:
                    line = line.strip()
                    number = float(line.split(":")[1].strip())
                    all_lpips.append(number)
        average_lpips = sum(all_lpips) / len(all_lpips)
        with open(
            os.path.join(os.path.dirname(lst_files[0]), "average_lpips.txt"),
            "w",
        ) as f:
            f.write(f"Average LPIPS: {average_lpips}")

    def print_net_parameter_stats(self, param_name: str = "out_weight"):
        for name, param in self._manager.named_parameters():
            if param.requires_grad and param_name in name:
                mean = utils.truncate(
                    param.data.mean().cpu().numpy(), decimals=3
                )
                std = utils.truncate(param.data.std().cpu().numpy(), decimals=3)
                print(f"{name} = {mean} +- {std}")

    def render_learned_heat_diffusion_times(
        self, param_name: str = "diffusion", n_heat_sources: int = 5
    ):
        self._manager.change_rendering_camera_param(
            "sample_count", self._sample_count_rendering
        )
        self._manager.pick_shapes_for_logging(
            self._loaders, self._loaders[self._phase].batch_size
        )
        batch = self._manager.get_shapes_for_logging(self._phase)[0]
        assert "to_padded_mask" in batch.keys()  # they are really batched

        def _heat_diffusion(d, t):
            x_spec = utils.to_basis(d.x, d.evecs, d.massvec)
            diffusion_coefs = torch.exp(-d.evals.unsqueeze(-1) * t.unsqueeze(0))
            x_diffuse_spec = diffusion_coefs * x_spec
            return utils.from_basis(x_diffuse_spec, d.evecs)

        out_img_dir = os.path.join(
            self._output_directory, "renderings_learned_heat_diffusions"
        )
        utils.mkdir(out_img_dir)

        base_color = utils.get_rgb_color("lightgrey")
        base_color = base_color.to(self._device)

        rendering.mega_kernel(state=False)
        all_renderings = defaultdict(list)
        for i in range(batch.num_graphs):
            data = batch.get_example(i)
            data.evals = batch.evals[i, :]

            # select n_heat_sources farthest samples
            mask = utils.farthest_point_sampling(data.pos, n_heat_sources)
            indices = torch.argwhere(mask).squeeze()
            # sampled_points = data.pos[mask, :]

            for layer_name, param in self._manager.named_parameters():
                if param.requires_grad and param_name in layer_name:
                    renders = []
                    ts = [
                        # param.data.min(),
                        param.data.mean(),
                        # param.data.max(),
                    ]
                    for t in ts:
                        end_colors_iterator = utils.get_color_iterator()
                        combined_map = torch.zeros_like(
                            data.x, device=self._device, requires_grad=False
                        )
                        # diffuse all point sources separately so that they can
                        # be assigned a different colour.
                        for i in indices:
                            d = data.clone().to(self._device)
                            d.x = torch.zeros_like(data.x, device=self._device)
                            d.x = d.x[:, 0:1]
                            d.x[i, :] = 1.0
                            diffused = _heat_diffusion(d, t)
                            cmap = ["lightgrey", next(end_colors_iterator)]

                            single_source_map = utils.values_to_cmap(
                                diffused, cmap, None, None
                            ).squeeze()
                            # remove base colour from everywhere and add it back
                            # after combining all diffused source maps
                            combined_map += single_source_map - base_color

                        combined_map += base_color
                        # will go back to [0,1] in rendering
                        d.x = (combined_map - 0.5) * 2
                        renders.append(self._manager.render(d).detach().cpu())
                    all_renderings[layer_name].append(
                        torch.cat(renders, dim=-2)
                    )

        rendering.flush_cache()
        for name, list_rends in all_renderings.items():
            name = name.replace(".", "_")
            im = torch.cat(list_rends, dim=-3)
            mi.util.write_bitmap(os.path.join(out_img_dir, name + ".png"), im)


if __name__ == "__main__":
    import os
    import argparse
    import torch

    import utils
    from data_loading import get_all_data_loaders
    from model_manager import ModelManager

    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, default="026", help="experiment ID")
    parser.add_argument("--output_path", type=str, default="./outputs")
    parser.add_argument("--processed_dir_name", type=str, default="processed")
    parser.add_argument("--batch_size", type=int)
    parsed = parser.parse_args()

    experiment_name = parsed.id
    output_directory = os.path.join(parsed.output_path, experiment_name)
    checkpoint_directory = os.path.join(output_directory, "checkpoints")
    config = utils.get_config(os.path.join(output_directory, "config.yaml"))
    data_config = config["data"]

    # Use GPU if available
    if not torch.cuda.is_available():
        device = torch.device("cpu")
        print("GPU not available, running on CPU")
    else:
        device = torch.device("cuda")

    if parsed.batch_size is not None:
        data_config["batch_size"] = parsed.batch_size

    pre_transform = get_transforms(
        list_transforms=data_config["pre_transforms_list"],
        transforms_config=data_config["transforms_config"],
        root=data_config["root"],
    )
    transform = get_transforms(
        list_transforms=data_config["transforms_list"],
        transforms_config=data_config["transforms_config"],
        root=data_config["root"],
    )

    try:
        custom_eval_list_path = os.path.join(
            data_config["root"], parsed.processed_dir_name, "custom_eval.txt"
        )
        with open(custom_eval_list_path, "r") as f:
            custom_eval_list = f.readlines()
        custom_eval_list = [fn.strip() for fn in custom_eval_list]
    except FileNotFoundError:
        custom_eval_list = None

    loaders = get_all_data_loaders(
        data_config,
        transform,
        pre_transform,
        None,
        parsed.processed_dir_name,
        list_of_files_to_use=custom_eval_list,
    )

    model_manager = ModelManager(config, len_train_loader=len(loaders["train"]))
    model_manager = model_manager.to(device)
    model_manager.resume(checkpoint_directory)

    # TODO: change to test on final model
    tester = ModelTester(
        model_manager, loaders, config, device, output_directory, phase="test"
    )
    tester()
    # tester.change_sampling_density(n_poisson_samples=30_000)
    # tester.render_learned_heat_diffusion_times(
    #     param_name="diffusion", n_heat_sources=8
    # )
    # tester.print_net_parameter_stats(param_name="out_weight")
    # tester.print_net_parameter_stats(param_name="diffusion_time")
    # tester.print_net_parameter_stats(param_name="diffusion_in_time")
    # tester.print_net_parameter_stats(param_name="diffusion_out_time")

    # tester.change_sampling_density(n_poisson_samples=5_000)
    # tester.render_more_gen_textures(n_shapes=6, n_variants=3)

