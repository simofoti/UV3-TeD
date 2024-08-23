import os
import json
from abc import abstractmethod
from typing import Any, Callable, List, Tuple, Dict

import func_timeout
import tqdm
import trimesh
import torch
import torch.utils.data
import torch_geometric.data

import numpy as np
import pyvista as pv
from PIL import Image
from torch.utils.data.dataloader import default_collate
from torch_geometric.data.collate import collate as geom_collate

import utils


def get_all_data_loaders(
    data_config: dict,
    transform: Callable[..., Any] | None = None,
    pre_transform: Callable[..., Any] | None = None,
    pre_filter: Callable[..., Any] | None = None,
    processed_dir_name: str = "processed",
    list_of_files_to_use: List[str] | None = None,
    **kwargs,
) -> Dict:
    if "AmazonBerkeleyObjects" in data_config["root"]:
        dataset_class = AmazonBerkeleyObjectsDataset
        category = None  # The whole dataset is used
    elif "ShapeNet" in data_config["root"]:
        dataset_class = ShapeNetDataset
        category = utils.in_or_default(data_config, "category", "all")
    elif "celeba" in data_config["root"]:
        dataset_class = SyntheticMeshCelebADataset
        category = None
    else:
        avail_datasets = ["AmazonBerkeleyObjects", "ShapeNet"]
        raise NotImplementedError(
            "Which dataset are you using? The available options are:",
            f"{avail_datasets}. Make sure that the name of the",
            "dataset is contained in the 'root' parameter of the config file.",
            "For a new dataset you may attemp to create a new dataset class",
            "inheriting from 'BaseMeshDataset' and overriding the 'download'",
            "method (PS. you can just use a pass statement)!",
        )

    loaders = {}
    for dt in ["train", "val", "test"]:
        dataset = dataset_class(
            root=data_config["root"],
            dataset_type=dt,
            pre_transform=pre_transform,
            transform=transform,
            pre_filter=pre_filter,
            filter_only_files_with=utils.in_or_default(
                data_config, "filter_files_with", []
            ),
            processed_dir_name=processed_dir_name,
            category=category,
            num_workers=utils.in_or_default(data_config, "num_workers", 8),
            list_of_files_to_use=list_of_files_to_use,
        )
        loaders[dt] = MeshLoader(
            dataset,
            data_config["batch_size"],
            shuffle=True,
            pad=utils.in_or_default(data_config, "pad", False),
            num_workers=utils.in_or_default(data_config, "num_workers", 8),
            **kwargs,
        )

    print(f"Files not processed: {len(dataset.all_processing_failure_files)}")
    return loaders


# Load and batch meshes ########################################################
class MeshLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        pad=False,
        set_affinity_manually: bool = True,
        **kwargs,
    ):
        collater = MeshCollater(pad, batch_size)
        worker_init = worker_init_fn if set_affinity_manually else None
        super(MeshLoader, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=collater,
            worker_init_fn=worker_init,
            **kwargs,
        )


def worker_init_fn(worker_id):
    # Compensate for unexpected modification to CPU affinity of Dataloader
    # workers in some environments where pytorch or a dependency appears to
    # unexpectedly modify CPU affinity
    os.sched_setaffinity(0, range(os.cpu_count()))


class MeshCollater:
    def __init__(self, pad: bool = False, batch_size: int = 1) -> None:
        self._pad = pad
        self._batch_size = batch_size

    def __call__(self, data_list):
        if self._batch_size > 1 and not self._pad:
            return self.custom_collate(data_list)
        elif self._batch_size > 1 and self._pad:
            return self.pad_collate(data_list)
        else:
            return self.equal_sized_collate(data_list)

    def equal_sized_collate(
        self, data_list: List[torch_geometric.data.Data]
    ) -> torch_geometric.data.Data:
        if not isinstance(data_list[0], torch_geometric.data.Data):
            raise TypeError(
                f"DataLoader found invalid type: {type(data_list[0])}. "
                f"Expected torch_geometric.data.Data instead"
            )

        keys = [set(data.keys()) for data in data_list]
        keys = list(set.union(*keys))
        batched_data = torch_geometric.data.Data()
        for key in keys:
            attribute_list = [data[key] for data in data_list]
            try:
                batched_data[key] = default_collate(attribute_list)
            except (TypeError, RuntimeError):
                # attributes with incompatible sizes or unknow types are batched
                # as a list
                batched_data[key] = attribute_list
        return batched_data

    def custom_collate(
        self, data_list: List[torch_geometric.data.Data]
    ) -> torch_geometric.data.Batch:
        """
        Custom collation that mostly pack collates (batching of torch_geometric)
        but evals are batched on dim 0. Also values to convert from padded to
        packed batches are precomputed.
        """
        x_sizes = []
        # pytorch geometric batches automatically on the diagonal attributes
        # with 'adj' in their name. Temporarily rename them to avoid changing
        # Data class or pytorch geometric collater.
        for data in data_list:
            self._rename_attribute(data, "grad_x", "grad_x_adj")
            self._rename_attribute(data, "grad_y", "grad_y_adj")
            x_sizes.append(data.x.shape[0])

        batch, slice_dict, inc_dict = geom_collate(
            torch_geometric.data.Batch,
            data_list=data_list,
            increment=True,
            add_batch=True,
            exclude_keys=["evals", "farthest_sampling_mask"],
        )

        self._rename_attribute(batch, "grad_x_adj", "grad_x")
        self._rename_attribute(batch, "grad_y_adj", "grad_y")
        slice_dict["grad_x"] = slice_dict.pop("grad_x_adj")
        slice_dict["grad_y"] = slice_dict.pop("grad_y_adj")
        inc_dict["grad_x"] = inc_dict.pop("grad_x_adj")
        inc_dict["grad_y"] = inc_dict.pop("grad_y_adj")

        # Note that get_example won't work converting tensors to coo
        # batch.grad_x = batch.grad_x.to_torch_sparse_coo_tensor()
        # batch.grad_y = batch.grad_y.to_torch_sparse_coo_tensor()

        batch._num_graphs = len(data_list)
        batch._slice_dict = slice_dict
        batch._inc_dict = inc_dict

        # Precompute info to convert from padded to packed and vice-versa
        batch.max_verts = max(x_sizes)
        to_padded_mask = torch.zeros(
            (batch._num_graphs, batch.max_verts), device=batch.x.device
        ).bool()
        for i, y in enumerate(x_sizes):
            to_padded_mask[i, slice(0, y)] = True
        batch.to_padded_mask = to_padded_mask.flatten()

        batch.to_packed_idx = torch.cat(
            [
                torch.arange(v, device=batch.x.device) + i * batch.max_verts
                for (i, v) in enumerate(x_sizes)
            ],
            dim=0,
        ).long()

        # batch stack attributes that are going to be used in heat-diffusion.
        # Note that evecs are fed also as input, conversion is performed online.
        batch.evals = torch.stack([d.evals for d in data_list], dim=0)

        if "farthest_sampling_mask" in data_list[0].keys():
            farthest_masks_list = [d.farthest_sampling_mask for d in data_list]
            batch.farthest_sampling_mask = farthest_masks_list[0].new_full(
                (batch._num_graphs, batch.max_verts), False
            )
            for i, data in enumerate(data_list):
                slices = [i, slice(0, farthest_masks_list[i].shape[0])]
                batch.farthest_sampling_mask[slices] = farthest_masks_list[i]

        return batch

    def pad_collate(
        self, data_list: List[torch_geometric.data.Data]
    ) -> torch_geometric.data.Data:
        if not isinstance(data_list[0], torch_geometric.data.Data):
            raise TypeError(
                f"DataLoader found invalid type: {type(data_list[0])}. "
                f"Expected torch_geometric.data.Data instead"
            )

        keys = [set(data.keys()) for data in data_list]
        keys = list(set.union(*keys))

        batched_data = torch_geometric.data.Data()

        pos_list = [data.pos for data in data_list]

        padded_dims = list(pos_list[0].shape)
        padded_dims[0] = max([y.shape[0] for y in pos_list if len(y) > 0])
        padded = pos_list[0].new_full((len(pos_list), *padded_dims), 0.0)

        padded_dims_evecs = list(data_list[0].evecs.shape)
        padded_dims_evecs[0] = padded_dims[0]

        batched_data.pos = padded.clone()
        batched_data.x = padded.clone()
        batched_data.evecs = data_list[0].evecs.new_full(
            (len(pos_list), *padded_dims_evecs), 0.0
        )
        batched_data.massvec = data_list[0].massvec.new_full(
            (len(pos_list), padded_dims[0]), 0.0
        )

        for i, data in enumerate(data_list):
            slices = [i, slice(0, pos_list[i].shape[0])]
            batched_data.pos[slices] = pos_list[i]
            batched_data.x[slices] = data.x
            batched_data.evecs[slices] = data.evecs
            batched_data.massvec[slices] = data.massvec

        for key in keys:
            if key in ["pos", "x", "evecs", "massvec"]:
                continue

            attribute_list = [data[key] for data in data_list]
            try:
                batched_data[key] = default_collate(attribute_list)
            except (TypeError, RuntimeError):
                # attributes with incompatible sizes or unknow types are batched
                # as a list
                batched_data[key] = attribute_list
        return batched_data

    @staticmethod
    def _rename_attribute(data, old_name, new_name):
        data._store._mapping[new_name] = data._store._mapping.pop(old_name)


# Mesh Datasets ################################################################
class BaseMeshDataset(torch_geometric.data.Dataset):
    def __init__(
        self,
        root: str,
        dataset_type: str = "train",
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        pre_filter: Callable[..., Any] | None = None,
        processed_dir_name: str = "processed",
        filter_only_files_with: str | List[str] | None = None,
        list_of_files_to_use: List[str] | None = None,
        **kwargs,
    ):
        self._root = root
        if not hasattr(self, "_category"):
            self._category = ""
        self._dataset_type = dataset_type
        self._processed_dir_name = processed_dir_name

        torch_geometric.data.makedirs(self.processed_dir)

        self._split_fpath = os.path.join(self.processed_dir, "data_split.json")
        self._train_names, self._test_names, self._val_names = self._split_data(
            os.path.join(self.processed_dir, "data_split.json")
        )

        if isinstance(filter_only_files_with, str):
            self._only_files_with(filter_only_files_with)
        elif isinstance(filter_only_files_with, list):
            for filter in filter_only_files_with:
                self._only_files_with(filter)

        if list_of_files_to_use is not None:
            self._train_names = list_of_files_to_use
            self._test_names = list_of_files_to_use
            self._val_names = list_of_files_to_use

        self._processed_files = [f[:-4] + ".pt" for f in self.raw_file_names]

        super().__init__(root, transform, pre_transform, pre_filter)

    @abstractmethod
    def download(self):
        r"""Downloads the dataset to the :obj:`self.raw_dir` folder."""
        raise NotImplementedError

    @property
    def raw_file_names(self) -> List[str]:
        if self._dataset_type == "train":
            file_names = self._train_names
        elif self._dataset_type == "test":
            file_names = self._test_names
        elif self._dataset_type == "val":
            file_names = self._val_names
        else:
            raise Exception("train, val and test are supported dataset types")
        return file_names

    @property
    def processed_file_names(self) -> List[str]:
        return self._processed_files

    @property
    def processed_dir(self) -> str:
        return os.path.join(self._root, self._processed_dir_name)

    @property
    def all_processing_failure_files(self) -> List[str]:
        all_raw_fnames = self._find_filenames()
        all_processed = self._train_names + self._test_names + self._val_names
        return [f for f in all_raw_fnames if f not in all_processed]

    @property
    def raw_dir(self) -> str:
        return self._root

    def get(self, idx: int) -> torch_geometric.data.Data:
        filename = self.processed_file_names[idx]
        return torch.load(os.path.join(self.processed_dir, filename))

    def process(self):
        fnames_failed_transforms = []
        for fname in tqdm.tqdm(
            self.raw_file_names, desc=f"Processing {self._dataset_type} data"
        ):
            out_path = self._get_corresponding_out_fname(fname)

            if not os.path.exists(out_path):
                is_processed = self._process_single_file(fname, out_path)
                if not is_processed:
                    fnames_failed_transforms.append(fname)

        self._update_files_lists(fnames_to_remove=fnames_failed_transforms)
        print(
            f"The following files were not transformed and are not goint to ",
            f"be used: {fnames_failed_transforms}",
        )

    def _process_single_file(self, fname: str, out_path: str) -> bool:
        try:
            file_path = os.path.join(self._root, fname)
            mesh = utils.load_mesh(file_path)

            verts = torch.tensor(
                mesh.vertices, dtype=torch.float, requires_grad=False
            ).contiguous()

            faces = torch.tensor(
                mesh.faces.T, dtype=torch.long, requires_grad=False
            ).contiguous()

            data = torch_geometric.data.Data(
                pos=verts, face=faces, original_trimesh=mesh, raw_path=fname
            )

            # Handle pre_transform failure (e.g. when eigendecomposition
            # fails or takes too long)
            if self.pre_transform is not None:
                data = self.pre_transform(data)
        except (
            TypeError,
            ValueError,
            func_timeout.FunctionTimedOut,
            RuntimeError,
        ):
            return False

        fname_split = fname.split(os.sep)
        if len(fname_split) > 1:
            utils.mkdir(os.path.join(self.processed_dir, *fname_split[:-1]))
        torch.save(data, out_path)
        return True

    def _get_corresponding_out_fname(self, fname: str) -> str:
        return os.path.join(self.processed_dir, fname[:-4] + ".pt")

    def _find_filenames(
        self, file_ext: str | Tuple[str] | None = None
    ) -> List[str]:
        if file_ext is None:
            file_ext = (".ply", ".obj", ".glb")
        root_l = len(self._root)
        files = []
        path_to_walk = os.path.join(self._root, self._category)
        for dirpath, _, fnames in os.walk(path_to_walk):
            for f in fnames:
                if f.endswith(file_ext):
                    absolute_path = os.path.join(dirpath, f)
                    f = absolute_path[dirpath.index(self._root) + root_l + 1 :]
                    files.append(f)
        return files

    def _split_data(
        self, data_split_list_path: str
    ) -> Tuple[List[str], List[str], List[str]]:
        try:
            with open(data_split_list_path, "r") as fp:
                data = json.load(fp)
            train_list = data["train"]
            test_list = data["test"]
            val_list = data["val"]
        except FileNotFoundError:
            all_file_names = self._find_filenames()

            train_list, test_list, val_list = [], [], []
            for i, fname in enumerate(all_file_names):
                if i % 100 <= 5:
                    test_list.append(fname)
                elif i % 100 <= 10:
                    val_list.append(fname)
                else:
                    train_list.append(fname)

            data = {"train": train_list, "test": test_list, "val": val_list}
            with open(data_split_list_path, "w") as fp:
                json.dump(data, fp)
        return train_list, test_list, val_list

    def _only_files_with(self, attribute_condition: str = "base_color_tex"):
        not_used_path = os.path.join(self.processed_dir, "not_used.json")
        unused_key = f"{self._dataset_type}_no_{attribute_condition}"
        # All unused files are stored in a list to avoid recomputing. Try to see
        # if it has already been created with the desired filter.
        try:
            with open(not_used_path, "r") as fp:
                not_used_all = json.load(fp)
            unused = not_used_all[unused_key]
            for f in unused:
                try:
                    self.raw_file_names.remove(f)
                except ValueError:  # not in raw_file_names list already
                    pass

        # The desired filtering has not been done yet. Do it and save it.
        except (FileNotFoundError, KeyError):
            unused = []
            s = f"Filter {self._dataset_type} files with {attribute_condition}"

            # TODO: iterate through data only once and save table with all
            # metadata that can be quickly filtered without reading all files

            for f in tqdm.tqdm(list(self.raw_file_names), desc=s):
                raw_path = os.path.join(self._root, f)
                try:
                    if not os.path.isfile(raw_path):
                        raise FileNotFoundError
                    mesh = utils.load_mesh(raw_path)
                    if attribute_condition == "base_color_tex":
                        base_texture = mesh.visual.material.baseColorTexture
                        if base_texture is None:
                            raise AttributeError
                    elif attribute_condition == "a_texture":
                        material = mesh.visual.material
                        if isinstance(
                            material, trimesh.visual.material.SimpleMaterial
                        ):
                            texture = material.image
                        else:
                            texture = material.baseColorTexture
                        if texture is None:
                            raise AttributeError
                    elif attribute_condition[:4] == "less":
                        # ex. less_100k_verts -> convert 100k in a number
                        mv = int(attribute_condition.split("_")[1][:-1]) * 1000
                        if mesh.vertices.shape[0] > mv:
                            raise ValueError
                    elif attribute_condition[:4] == "more":
                        # ex. more_0.1k_verts -> convert 0.1k in a number
                        m = float(attribute_condition.split("_")[1][:-1]) * 1000
                        if mesh.vertices.shape[0] < m:
                            raise ValueError
                    else:
                        raise NotImplementedError(
                            "only 'base_color_tex' and 'less_Nk_verts'",
                            "currently implemented...",
                        )
                except (
                    AttributeError,
                    ValueError,
                    # TypeError,  # ShapeNet often had NoneTypes (w/o material)
                    # FileNotFoundError,  # ShapeNet, not all files in split exist
                    # IndexError,  # ShapeNet, faces and vertices not compatible
                    # AssertionError,  # shapenet-glb, glb views != byte length
                ):
                    unused.append(f)
                    self.raw_file_names.remove(f)

            # Save/append to avoid recomputing
            if os.path.exists(not_used_path):
                with open(not_used_path, "r") as fp:
                    not_used_all = json.load(fp)
                not_used_all[unused_key] = unused
            else:
                not_used_all = {unused_key: unused}
            with open(not_used_path, "w") as fp:
                json.dump(not_used_all, fp)

        print(
            f"The {len(unused)} {self._dataset_type} files with no",
            f"{attribute_condition} were filtered out and will not be used.",
            "If you changed data split you may need to delete 'not_used.json'.",
        )

    def _update_files_lists(self, fnames_to_remove: List[str]):
        if len(fnames_to_remove) > 0:
            with open(self._split_fpath, "r") as fp:
                data = json.load(fp)

            for fname in fnames_to_remove:
                if self._dataset_type == "train":
                    self._train_names.remove(fname)
                elif self._dataset_type == "test":
                    self._test_names.remove(fname)
                elif self._dataset_type == "val":
                    self._val_names.remove(fname)
                self.processed_file_names.remove(fname[:-4] + ".pt")
                data[self._dataset_type].remove(fname)

            with open(self._split_fpath, "w") as fp:
                json.dump(data, fp)

    def len(self) -> int:
        return len(self.processed_file_names)


class AmazonBerkeleyObjectsDataset(BaseMeshDataset):
    def download(self):
        tarname = "abo-3dmodels.tar"
        url = "https://amazon-berkeley-objects.s3.amazonaws.com/archives/"
        root = self._root

        flag_print_message = True
        if "AmazonBerkeleyObjects/original" in root:
            root = root.replace("/AmazonBerkeleyObjects/original", "")
            flag_print_message = False

        print(
            "You are now about to download the AmazonBerkeleyObjects dataset",
            "The compressed archive weights 155 GB. Once extracted it will",
            "occupy approximately the same space. Once processed additional",
            "175 GB will be required to store the processed information. ",
            "Ensure you have at least 330 GB available if you wish to remove",
            "the compressed archive or 505 GB if you wish to keep it. ",
        )

        # Internally, 'download_url' already checks if file exists
        tar_path = torch_geometric.data.download_url(url + tarname, root)

        extraction_folder = os.path.join(root, "AmazonBerkeleyObjects")
        if not os.path.isdir(extraction_folder):
            torch_geometric.data.makedirs(extraction_folder)
            torch_geometric.data.extract_tar(tar_path, extraction_folder)
            print(
                f"Dataset downloaded and extracted in {extraction_folder}.",
                f"You can delete {tar_path} if you wish.",
            )
        else:
            print(
                "It looks like the dataset has already been downloaded.",
                f"If {extraction_folder} does not contain the desired data,",
                "change the 'root' parameter in your config file or delete",
                f"the folder: {extraction_folder}",
            )

        if flag_print_message:
            print(
                "Please, from the next execution modify the 'root' parameter",
                "in your config file to:",
                f"'{os.path.join(extraction_folder, 'original')}'.",
            )


class ShapeNetDataset(BaseMeshDataset):
    def __init__(
        self,
        root: str,
        dataset_type: str = "train",
        category: str | None = None,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        pre_filter: Callable[..., Any] | None = None,
        processed_dir_name: str = "processed",
        filter_only_files_with: str | None = None,
        list_of_files_to_use: List[str] | None = None,
        **kwargs,
    ):
        self._category = "" if category in ["all", None] else category
        super().__init__(
            root,
            dataset_type,
            transform,
            pre_transform,
            pre_filter,
            processed_dir_name,
            filter_only_files_with,
            list_of_files_to_use,
            **kwargs,
        )

    def download(self):
        # Automatic download would require extra packages and permissions
        # to access the repo. Provide instructions instead
        print(
            "Request permissions to access the following hugging face repo:",
            "https://huggingface.co/datasets/ShapeNet/ShapeNetCore. ",
            "Then, make sure git-lfs is installed (git lfs install) and clone:",
            "git clone git@hf.co:datasets/ShapeNet/ShapeNetCore.",
        )


class SyntheticMeshCelebADataset(BaseMeshDataset):
    def __init__(
        self,
        root: str,
        dataset_type: str = "train",
        base_mesh_path: str | None = None,
        transform: Callable[..., Any] | None = None,
        pre_transform: Callable[..., Any] | None = None,
        pre_filter: Callable[..., Any] | None = None,
        processed_dir_name: str = "processed",
        filter_only_files_with: str | None = None,
        list_of_files_to_use: List[str] | None = None,
        **kwargs,
    ):
        super().__init__(
            root,
            dataset_type,
            transform,
            pre_transform,
            pre_filter,
            processed_dir_name,
            filter_only_files_with,
            list_of_files_to_use,
            **kwargs,
        )

        if base_mesh_path is None:
            base_mesh_path = os.path.join(root, "wave_plane.obj")

            # create wave-like surface mesh on which to apply textures
            steps = np.arange(-10, 10, 0.5)
            x, y = np.meshgrid(steps, steps)
            z = np.sin(np.sqrt(x**2 + y**2))
            surface = pv.StructuredGrid(x, y, z)
            surface.texture_map_to_plane(inplace=True)
            surface = surface.extract_surface().triangulate()

            faces_as_array = surface.faces.reshape((surface.n_cells, 4))[:, 1:]
            base_mesh = trimesh.Trimesh(surface.points / 10, faces_as_array)
            base_mesh.visual = trimesh.visual.TextureVisuals(
                uv=surface.active_texture_coordinates
            )
            base_mesh.export(base_mesh_path)
        else:
            base_mesh = utils.load_mesh(base_mesh_path)
            base_mesh.visuals, base_mesh.uv = None, None
            surface = pv.wrap(base_mesh)
            surface.texture_map_to_plane(inplace=True)  # could use sphere map

        self._uv = surface.active_texture_coordinates

        verts = torch.tensor(
            base_mesh.vertices, dtype=torch.float, requires_grad=False
        ).contiguous()
        faces = torch.tensor(
            base_mesh.faces.T, dtype=torch.long, requires_grad=False
        ).contiguous()

        base_data = torch_geometric.data.Data(
            pos=verts,
            face=faces,
            original_trimesh=base_mesh,
            raw_path=base_mesh_path,
        )

        if self.pre_transform is not None:
            base_data = self.pre_transform(base_data)

        self._base_data = base_data

    @property
    def all_processing_failure_files(self) -> List[str]:
        return []

    def download(self):
        # Skip automatic download for simplicity, but still provide manual steps
        print(
            "Download and unzip the Align&Croped Images from:",
            "https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html",
        )

    def process(self):
        # Not needed as all meshes are the same
        pass

    @staticmethod
    def pil_loader(path: str) -> Image.Image:
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def _find_filenames(self) -> List[str]:
        return super()._find_filenames((".jpg", ".jpeg", ".png"))

    def get(self, idx: int) -> torch_geometric.data.Data:
        filename = self.raw_file_names[idx]
        im = self.pil_loader(os.path.join(self._root, self._category, filename))
        material = trimesh.visual.texture.SimpleMaterial(image=im)
        color_visuals = trimesh.visual.TextureVisuals(
            uv=self._uv, image=im, material=material
        )
        data = self._base_data.clone()
        data.original_trimesh.visual = color_visuals
        data.texture = im
        return data

    def _only_files_with(self, attribute_condition):
        pass


################################################################################


if __name__ == "__main__":
    import rendering
    import mitsuba as mi
    import drjit

    import cProfile, pstats

    from transforms import get_transforms

    data_config = {
        "root": "/data/AmazonBerkeleyObjects/original",
        "batch_size": 1,
        "num_workers": 0,
        "filter_files_with": ["base_color_tex", "less_60k_verts"],
        "pre_transforms_list": [
            "normalise_scale",
            "laplacian_eigendecomposition",
            "tangent_gradients",
            "drop_trimesh",
        ],
        "transforms_list": [
            "sample_everything_poisson",
            "sample_farthest",
            "drop_laplacian",
            "drop_edges",
            "drop_faces",
        ],
        "transforms_config": {
            "n_poisson_samples": 5_000,
            "n_farthest_samples": 100,
            "resize_texture": True,
            "mix_laplacian_w": 0.05,
            "lapl_as_cloud": False,
            "eigen_number": 128,
            "eigen_eps": 1e-8,
            "eigen_timeout_seconds": 300,
            "store_lapl": False,
            "store_massvec": False,
            "grads_as_cloud": False,
            "save_edges": True,
            "save_normals": True,
        },
    }

    pre_transform = get_transforms(
        data_config["pre_transforms_list"],
        data_config["transforms_config"],
        data_config["root"],
    )
    transform = get_transforms(
        data_config["transforms_list"],
        data_config["transforms_config"],
        data_config["root"],
    )

    # dataset = ShapeNetDataset(
    #     root=data_config["root"],
    #     dataset_type="train",
    #     category=None,
    #     pre_transform=pre_transform,
    #     transform=transform,
    #     pre_filter=None,
    #     filter_only_files_with=data_config["filter_files_with"],
    #     processed_dir_name="processed",
    # )

    profiler = cProfile.Profile()
    profiler.enable()

    loaders = get_all_data_loaders(data_config, transform, pre_transform)
    batch = next(iter(loaders["train"]))

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("ncalls")
    stats.dump_stats("outputs/profiled_dataload")
    # NOTE: read it with snakeviz -s profiled_dataload (can be pip installed)

    # scene_dict = {
    #     "type": "scene",
    #     "integrator": rendering.define_integrator(),
    #     "camera": rendering.define_camera(3.5, 30, 60, "perspective", 512, 512),
    #     "emitter": rendering.define_emitter(None),
    #     "mesh": rendering.data_coloured_points_to_mitsuba(batch),
    # }
    # scene = mi.load_dict(scene_dict)

    # rendering.mega_kernel(False)
    # rendered_image = mi.render(scene)
    # rendering.flush_cache()

    # cols = batch.x[0]
    print("done")
