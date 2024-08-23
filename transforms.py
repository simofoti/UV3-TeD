import os
from typing import Any, Callable, List, Dict

import func_timeout
import trimesh
import torch
import torch_geometric.data
import torch_geometric.transforms

import numpy as np
import point_cloud_utils as pcu

import utils


def get_transforms(
    list_transforms: List[str],
    transforms_config: Dict[str, Any] | None,
    root: str | None,
) -> Callable[..., Any] | None:
    if not list_transforms:
        return None
    else:
        cfg = transforms_config if transforms_config is not None else {}
        transforms = []
        for tname in list_transforms:
            if tname == "drop_trimesh":
                transforms.append(DropTrimesh())
            elif tname == "drop_laplacian":
                transforms.append(DropLaplacian())
            elif tname == "drop_edges":
                transforms.append(DropEdges())
            elif tname == "drop_faces":
                transforms.append(DropFaces())
            elif tname == "vertex_colours_from_base_texture":
                transforms.append(
                    VertexColoursFromBaseTexture(root, drop_trimesh=False)
                )
            elif tname == "laplacian_eigendecomposition":
                transforms.append(
                    LaplacianEigendecomposition(
                        mix_lapl_w=utils.in_or_default(cfg, "mix_lapl_w", 0.05),
                        k_eig=utils.in_or_default(cfg, "eigen_number", 10),
                        eps=float(utils.in_or_default(cfg, "eigen_eps", 1e-8)),
                        as_cloud=utils.in_or_default(
                            cfg, "lapl_as_cloud", False
                        ),
                        drop_trimesh=False,
                        store_lapl=utils.in_or_default(
                            cfg, "store_lapl", False
                        ),
                        store_massvec=utils.in_or_default(
                            cfg, "store_massvec", True
                        ),
                        timeout_seconds=utils.in_or_default(
                            cfg, "eigen_timeout_seconds", 300
                        ),
                        on_verts=utils.in_or_default(
                            cfg, "lapl_on_verts", True
                        ),
                    )
                )
            elif tname == "scale_invariant_hks":
                transforms.append(
                    ScaleInvariantHeatKernelSignatures(
                        signatures_number=cfg["hks_number"],
                        max_time=utils.in_or_default(cfg, "hks_max_t", 25.0),
                        increment=utils.in_or_default(cfg, "hks_inc_t", 1 / 16),
                        time_scaler=utils.in_or_default(
                            cfg, "hks_scale_t", 0.01
                        ),
                    )
                )
            elif tname == "tangent_gradients":
                transforms.append(
                    TangentGradients(
                        as_cloud=utils.in_or_default(
                            cfg, "grads_as_cloud", False
                        ),
                        save_edges=utils.in_or_default(
                            cfg, "save_edges", False
                        ),
                        save_normals=utils.in_or_default(
                            cfg, "save_normals", False
                        ),
                    )
                )
            elif tname == "tangent_gradients_to_sparse_np":
                transforms.append(TangentGradientsToSparseNp())
            elif tname in ["normalise_scale", "normalize_scale"]:
                transforms.append(torch_geometric.transforms.NormalizeScale())
            elif tname == "normals":
                transforms.append(
                    torch_geometric.transforms.GenerateMeshNormals()
                )
            elif tname == "edges":
                transforms.append(
                    torch_geometric.transforms.FaceToEdge(remove_faces=False)
                )
            elif tname == "sample_everything_poisson":
                transforms.append(
                    SampleEverything(
                        root,
                        utils.in_or_default(cfg, "n_poisson_samples", 20_000),
                        utils.in_or_default(cfg, "resize_texture", False),
                        utils.in_or_default(cfg, "store_original_verts", True),
                        utils.in_or_default(cfg, "sample_evecs_mass", True),
                    )
                )
            elif tname == "sample_farthest":
                transforms.append(
                    FarthestPointSampling(
                        utils.in_or_default(cfg, "n_farthest_samples", 200)
                    )
                )
            else:
                raise NotImplementedError(f"{tname} not implemented yet.")

        if "drop_trimesh" in cfg:
            transforms.append(DropTrimesh())

        return torch_geometric.transforms.Compose(transforms)


# Mesh transformations #########################################################
class DropTrimesh(torch_geometric.transforms.BaseTransform):
    def __call__(
        self, data: torch_geometric.data.Data
    ) -> torch_geometric.data.Data:
        data.original_trimesh = None
        return data


class DropLaplacian(torch_geometric.transforms.BaseTransform):
    def __call__(
        self, data: torch_geometric.data.Data
    ) -> torch_geometric.data.Data:
        data.lapl = None
        return data


class DropEdges(torch_geometric.transforms.BaseTransform):
    def __call__(
        self, data: torch_geometric.data.Data
    ) -> torch_geometric.data.Data:
        data.edge_index = None
        return data


class DropFaces(torch_geometric.transforms.BaseTransform):
    def __call__(
        self, data: torch_geometric.data.Data
    ) -> torch_geometric.data.Data:
        data.face = None
        return data


class MixinColourSampling:
    def _initialise_mixin(self, root: str):
        self._root = root

    def _load_trimesh_and_base_texture(
        self, data: torch_geometric.data.Data, merge_tex: bool
    ) -> trimesh.Trimesh:
        if (
            "original_trimesh" not in data.keys()
            or data.original_trimesh is None
        ):
            raw_path = os.path.join(self._root, data.raw_path)
            original_trimesh = utils.load_mesh(raw_path, merge_tex=merge_tex)
        else:
            original_trimesh = data.original_trimesh

        try:
            base_texture = original_trimesh.visual.material.baseColorTexture
            # Note: objects with a trimesh SimpleMaterial are discarded because
            # they can hold a single texture image, which is more likely to have
            # shadows baked in (i.e. it is less likely to be an albedo).
            if base_texture is None:
                raise AttributeError

        except AttributeError:
            if any(x in self._root for x in ["ShapeNet", "shapenet", "averse"]):
                # ShapeNet should only have albedos, which may also be
                # contained in a SimpleMaterial. Be more flexible and try to
                # to get the texture of a SimpleMaterial too.
                try:
                    base_texture = original_trimesh.visual.material.image
                    if base_texture is None:
                        raise AttributeError
                except AttributeError:
                    print(
                        "Filter the data setting:"
                        "'filter_only_files_with: a_texture' ",
                        "in the dataset constructor or config file.",
                    )
            else:
                print(
                    "Filter the data setting:"
                    "'filter_only_files_with: base_color_tex' ",
                    "in the dataset constructor or config file.",
                )
        return original_trimesh, base_texture

    def _colours_for_training(self, colours):
        # get only RGB and normalise them in [0, 1]
        colours = colours[:, :3] / 255
        # shift colours in [-1, 1]
        colours = (colours - 0.5) * 2
        return colours


class VertexColoursFromBaseTexture(
    MixinColourSampling, torch_geometric.transforms.BaseTransform
):
    def __init__(self, root: str, drop_trimesh: bool = False) -> None:
        self._drop_trimesh = drop_trimesh
        self._initialise_mixin(root)

    def __call__(
        self, data: torch_geometric.data.Data
    ) -> torch_geometric.data.Data:
        if not all(c in data.keys() for c in ("original_trimesh", "texture")):
            original_trimesh, tex_img = self._load_trimesh_and_base_texture(
                data, merge_tex=True
            )
        else:
            original_trimesh, tex_img = data.original_trimesh, data.texture

        # Store absolute path to mesh in case need to retrieve vertices and
        # faces for rendering.
        data.raw_abs_path = os.path.join(self._root, data.raw_path)

        uv = original_trimesh.visual.uv % 1.0
        tex = trimesh.visual.texture.TextureVisuals(uv, image=tex_img)

        cols = self._colours_for_training(tex.to_color().vertex_colors)

        assert cols.shape[0] == data.pos.shape[0]

        data.x = torch.tensor(
            cols, dtype=torch.float, requires_grad=False
        ).contiguous()

        # Add also a surface area that would be computed with SampleEverything
        total_area = utils.compute_tot_area(data.pos, data.face)
        data.surface_area = total_area.to(torch.float)

        data.verts = data.pos  # they are the same when texture defined on verts

        if self._drop_trimesh:
            data.original_trimesh = None
        return data


class LaplacianEigendecomposition(torch_geometric.transforms.BaseTransform):
    def __init__(
        self,
        mix_lapl_w: float = 0.05,
        k_eig: int = 10,
        eps: float = 1e-8,
        as_cloud: bool = False,
        drop_trimesh: bool = False,
        store_lapl: bool = False,
        store_massvec: bool = True,
        timeout_seconds: int = 300,
        on_verts: bool = True,
    ) -> None:
        self._mix_lapl_w = mix_lapl_w
        self._k_eig = k_eig
        self._eps = eps
        self._as_cloud = as_cloud
        self._drop_trimesh = drop_trimesh
        self._store_lapl = store_lapl
        self._store_massvec = store_massvec
        self._timeout_seconds = timeout_seconds
        self._on_verts = on_verts

    def __call__(
        self, data: torch_geometric.data.Data
    ) -> torch_geometric.data.Data:
        if self._on_verts:
            verts = np.array(data.original_trimesh.vertices)
            faces = np.array(data.original_trimesh.faces)
        else:
            verts = data.pos.cpu().detach().numpy()
            faces = None

        evals, evecs, lapl, massvec = func_timeout.func_timeout(
            timeout=self._timeout_seconds,
            func=utils.compute_eig_laplacian,
            kwargs={
                "verts": verts,
                "faces": faces,
                "mix_lapl_w": self._mix_lapl_w,
                "k_eig": self._k_eig,
                "eps": self._eps,
                "as_cloud": self._as_cloud,
            },
        )
        data.evals = torch.tensor(evals, dtype=torch.float, requires_grad=False)
        data.evecs = torch.tensor(evecs, dtype=torch.float, requires_grad=False)

        if self._store_lapl:
            data.lapl = utils.sparse_np_to_torch(lapl)
        if self._store_massvec:
            data.massvec = torch.tensor(
                massvec, dtype=torch.float, requires_grad=False
            )

        if self._drop_trimesh:
            data.original_trimesh = None
        return data


class ScaleInvariantHeatKernelSignatures(
    torch_geometric.transforms.BaseTransform
):
    def __init__(
        self,
        signatures_number: int,
        max_time: float = 25.0,
        increment: float = 1 / 16,
        time_scaler: float = 0.01,
    ) -> None:
        """Compute scale invariant heat kernel signatures as described in
        'Scale-invariant heat kernel signatures for non-rigid shape recognition'
        by Bronstein and Kokkinos (2010). The default parameters are the same as
        those suggested in the paper.
        """
        self._signatures_number = signatures_number
        self._max_time = max_time
        self._increment = increment
        self._time_scaler = time_scaler
        self._tau_step = int(max_time / increment) + 1
        assert self._tau_step > signatures_number

    def __call__(self, data: Any) -> Any:
        taus = torch.linspace(
            0.0,
            self._max_time,
            steps=self._tau_step,
            device=data.evals.device,
            dtype=data.evals.dtype,
        )
        times = self._time_scaler * 2**taus

        hks = utils.compute_hks(data.evals, data.evecs, times)
        hks += 1e-8  # add small bias to prevent log(0)=-inf

        log_hks = torch.log(hks)
        derivative = log_hks.narrow(
            dim=1, start=1, length=log_hks.size(1) - 1
        ) - log_hks.narrow(dim=1, start=0, length=log_hks.size(1) - 1)

        fft = torch.fft.fft(derivative, dim=1)
        data.hks = torch.abs(fft)[:, : self._signatures_number]
        return data


class TangentGradients(torch_geometric.transforms.BaseTransform):
    def __init__(
        self,
        as_cloud: bool = False,
        save_edges: bool = False,
        save_normals: bool = False,
    ) -> None:
        self._face_to_edge_transform = torch_geometric.transforms.FaceToEdge(
            remove_faces=False
        )
        self._vertex_normals_transform = (
            torch_geometric.transforms.GenerateMeshNormals()
        )
        self._as_cloud = as_cloud
        self._save_edges = save_edges
        self._save_normals = save_normals

    def __call__(
        self, data: torch_geometric.data.Data
    ) -> torch_geometric.data.Data:
        if data.edge_index is None:
            data = self._face_to_edge_transform(data)
        if "norm" not in data.keys() or data.norm is None:
            data = self._vertex_normals_transform(data)

        data.grad_x, data.grad_y = utils.get_grad_operators(
            data.pos, data.face, data.edge_index, data.norm, self._as_cloud
        )

        if not self._save_edges:
            data.edge_index = None
        if not self._save_normals:
            data.norm = None
        return data


class TangentGradientsToSparseNp(torch_geometric.transforms.BaseTransform):
    # Pytorch doesn't load data with multiple workers if gradients are sparse
    def __call__(
        self, data: torch_geometric.data.Data
    ) -> torch_geometric.data.Data:
        data.grad_x = utils.sparse_torch_to_np(data.grad_x.coalesce())
        data.grad_y = utils.sparse_torch_to_np(data.grad_y.coalesce())
        return data


class FarthestPointSampling(torch_geometric.transforms.BaseTransform):
    def __init__(self, n_points: int = 100) -> None:
        self._n_points = n_points

    def __call__(
        self, data: torch_geometric.data.Data
    ) -> torch_geometric.data.Data:
        sampling_mask = utils.farthest_point_sampling(data.pos, self._n_points)
        # sampled_points = data.pos[sampling_mask, :]
        data.farthest_sampling_mask = sampling_mask
        return data


class SampleEverything(
    MixinColourSampling, torch_geometric.transforms.BaseTransform
):
    def __init__(
        self,
        root: str,
        n_samples: int = 20_000,
        resize_texture: bool = False,
        save_original_vertices: bool = True,
        sample_evecs_and_mass: bool = True,
    ) -> None:
        self._initialise_mixin(root)
        self._n_samples = n_samples
        self._resize_texture = resize_texture
        self._save_original_vertices = save_original_vertices
        self._sample_evecs_and_mass = sample_evecs_and_mass

    def __call__(
        self, data: torch_geometric.data.Data
    ) -> torch_geometric.data.Data:
        if not all(c in data.keys() for c in ("original_trimesh", "texture")):
            original_trimesh, tex_img = self._load_trimesh_and_base_texture(
                data, merge_tex=True
            )
        else:
            original_trimesh, tex_img = data.original_trimesh, data.texture

        # data.v_pos = data.pos  # save vertex pos as they can be transformed
        # verts = data.v_pos.cpu().detach().numpy()
        verts = np.array(original_trimesh.vertices)
        faces = np.array(original_trimesh.faces)
        uv = original_trimesh.visual.uv

        # Store absolute path to mesh in case need to retrieve vertices and
        # faces for rendering.
        data.raw_abs_path = os.path.join(self._root, data.raw_path)

        # Disk Poisson sampling (which returns a list of face_ids
        # and barycentric coordinates)
        num_samples = self._n_samples
        fid, bc = pcu.sample_mesh_poisson_disk(
            verts, faces, num_samples=num_samples
        )
        if fid.shape[0] < num_samples / 10:
            # print(
            #     f"Increasing sample density for {data.raw_path}"
            #     f"as it only had {fid.shape[0]} points"
            # )
            num_samples *= 10
            fid, bc = pcu.sample_mesh_poisson_disk(
                verts, faces, num_samples=num_samples
            )

        # Compute approximate squared radius used in Poisson disk sampling.
        # Note that barycentric coordinates are sampled from original mesh,
        # but we need to estimate radius in preprocessed mesh as the new_pos
        # are also interpolated from preprocessed positions.
        total_area = utils.compute_tot_area(data.pos, data.face)
        # squared_radius = total_area / (0.7 * torch.pi * num_samples)
        squared_radius = (2 * total_area * 0.7**2) / (num_samples * np.sqrt(3))

        # Estimate texture scaling
        if self._resize_texture:
            new_w, new_h = utils.estimate_poisson_scaled_texture_size(
                faces,
                torch.tensor(uv),
                data.pos,
                squared_radius,
                tex_img.size,
            )
            tex_img = tex_img.resize((new_w, new_h))

        if self._save_original_vertices:
            data.verts = data.pos.clone()

        # Get new positions
        new_pos = pcu.interpolate_barycentric_coords(faces, fid, bc, data.pos)
        data.pos = new_pos.to(torch.float).contiguous()

        # Gather scale information
        data.surface_area = total_area.to(torch.float)
        # data.bbox_sides = utils.compute_bounding_box_sides(new_pos)[None, :]

        # Get new colours
        new_uvs = pcu.interpolate_barycentric_coords(faces, fid, bc, uv)
        # UVs can go outside [0, 1] in shapenet, trimesh should now consider it
        new_uvs = new_uvs % 1.0
        new_cols = trimesh.visual.uv_to_color(new_uvs, tex_img)
        # new_cols = trimesh.visual.uv_to_interpolated_color(new_uvs, tex_img)
        new_cols = self._colours_for_training(new_cols)
        data.x = torch.tensor(
            new_cols, dtype=torch.float, requires_grad=False
        ).contiguous()

        if self._sample_evecs_and_mass:
            # Get new eigenvectors
            new_evecs = pcu.interpolate_barycentric_coords(
                faces, fid, bc, data.evecs
            )
            data.evecs = new_evecs.to(torch.float).contiguous()

            # Massvectors in heat diffusion equation effectively scale the
            # signal on the mesh to diffuse more towards neighbouring vertices
            # when the area of the triangle defined between them is bigger.
            # Since we are using Poisson Disk sampling, points should be
            # approximately at the same distance. Therefore we set the mass to
            # the approximate area of traingles that could be obtained from new
            # samples. We suppose triangles to be equilater with sides equal to
            # the radius used in Poisson sampling.
            data.massvec = torch.tensor(
                squared_radius * torch.sin(torch.pi / torch.tensor(3)) / 2
            )  # * torch.ones(data.x.shape[0], device=data.x.device)
            mean_n_incident_faces = np.mean(
                np.unique(faces, return_counts=True)[1]
            )
            data.massvec *= mean_n_incident_faces / 3

        # Get new normals, which are used for gradients computation
        new_norm = pcu.interpolate_barycentric_coords(faces, fid, bc, data.norm)
        data.norm = new_norm.to(torch.float).contiguous()

        # Get new gradients
        # TODO: potentially cache and then interpolate tangent frames if faster.
        data.grad_x, data.grad_y = utils.get_grad_operators(
            data.pos, data.face, data.edge_index, data.norm, as_cloud=True
        )
        return data
