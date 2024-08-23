import drjit
import mitsuba as mi

mi.set_variant("cuda_ad_rgb")

import torch
import torch_geometric.data
import torch_geometric.transforms
import numpy as np

from torch_geometric.nn import knn_interpolate

import utils


def define_integrator(hide_emitters: bool = False, type: str = "path") -> dict:
    # approach for solving the light transport equation
    return {"type": type, "hide_emitters": hide_emitters}


def define_camera(
    camera_distance: float,
    azimuth_deg: float,
    elevation_deg: float,
    camera_type: str = "perspective",
    img_width: int = 1024,
    img_height: int = 1024,
    sampler_type: str = "multijitter",  # default was "independent"
    sample_count: int = 16,
    fov: float = 40,
    aperture_radius: float | None = None,
    focus_distance: float | None = None,
) -> dict:
    camera_pos = mi.ScalarTransform4f.rotate([0, 0, 1], elevation_deg).rotate(
        [0, 1, 0], azimuth_deg
    ) @ mi.ScalarPoint3f([0, 0, camera_distance])
    camera = {
        "type": camera_type,
        "fov": fov,
        "near_clip": 0.01,
        "far_clip": 1000.0,
        "to_world": mi.ScalarTransform4f.look_at(
            origin=camera_pos, target=[0, 0, 0], up=[0, 1, 0]
        ),
        "film": {
            "type": "hdrfilm",
            "rfilter": {"type": "box"},
            "width": img_width,
            "height": img_height,
        },
        "sampler": {
            "type": sampler_type,
            "sample_count": sample_count,
        },
    }
    if camera_type == "thinlens":
        camera["aperture_radius"] = aperture_radius
        camera["focus_distance"] = focus_distance
    return camera


def define_emitter(envmap_path: str | None = None, scale: float = 1.0) -> dict:
    # Other emitters are possible, but require positiong the lights in
    # the correct position
    if envmap_path is None:
        emitter = {"type": "constant"}
    else:
        assert envmap_path.endswith(".exr")
        emitter = {"type": "envmap", "filename": envmap_path, "scale": scale}
    return emitter


def add_default_ground_plane(
    rotation_axis: list[float] = [1, 0, 0],
    rotation_angle: int = -90,
    scale=10,
    translation: list[float] = [0, 0, -0.1],
    checkerboard: bool = False,
) -> dict:
    transformation = (
        mi.ScalarTransform4f.rotate(axis=rotation_axis, angle=rotation_angle)
        .scale(scale)
        .translate(translation)
    )
    plane = {
        "type": "rectangle",
        "to_world": transformation,
        "material": {"type": "diffuse"},
    }
    if checkerboard:
        plane["material"]["reflectance"] = {
            "type": "checkerboard",
            "to_uv": mi.ScalarTransform4f.scale([50, 50, 1]),
        }
    else:
        plane["material"]["reflectance"] = {
            "type": "rgb",
            "value": [1.0, 1.0, 1.0],
        }
    return plane


def data_original_texture_to_mitsuba(
    data: torch_geometric.data.Data,
    twosided: bool = True,
    merge_tex: bool = True,
    normalize_scale: bool = True,
) -> mi.Mesh:
    data = data.detach().cpu()
    path = data.raw_abs_path
    path = path[0] if isinstance(path, list) else path

    original_trimesh = utils.load_mesh(path, merge_tex=merge_tex)
    try:
        img_texture = original_trimesh.visual.material.baseColorTexture
    except AttributeError:
        img_texture = original_trimesh.visual.material.image

    img_texture = np.asarray(img_texture, dtype=np.float32) / 255
    uv = np.array(original_trimesh.visual.uv)

    original = torch_geometric.data.Data(
        pos=torch.tensor(
            original_trimesh.vertices, dtype=torch.float, requires_grad=False
        ).contiguous()
    )
    if normalize_scale:
        original = torch_geometric.transforms.NormalizeScale()(original)
    v_pos = original.pos.squeeze().numpy()

    # NOTE:  other attributes can be added to bsdf_dict like for base_color
    # (e.g., roughness, metallic, anisotropic).
    bsdf_dict = {
        "type": "principled",
        "base_color": {
            "type": "bitmap",
            "bitmap": mi.Bitmap(img_texture),
        },
    }

    if twosided:
        bsdf_dict = {"type": "twosided", "material": bsdf_dict}

    bsdf_prop = mi.Properties()
    bsdf_prop["mesh_bsdf"] = mi.load_dict(bsdf_dict)

    mi_mesh = mi.Mesh(
        "mesh",
        vertex_count=v_pos.shape[0],
        face_count=original_trimesh.faces.shape[0],
        has_vertex_texcoords=True,
        props=bsdf_prop,
    )

    # "Traverse" the mesh to get its updateable parameters
    mesh_params = mi.traverse(mi_mesh)
    mesh_params["vertex_positions"] = np.array(v_pos).flatten()
    mesh_params["faces"] = np.array(original_trimesh.faces).flatten()
    mesh_params["vertex_texcoords"] = np.subtract(
        1.0, uv, out=uv, where=[False, True]
    ).flatten()
    return mi_mesh


def data_coloured_verts_to_mitsuba(
    data: torch_geometric.data.Data, twosided: bool = True
) -> mi.Mesh:
    data = data.detach().cpu()
    path = data.raw_abs_path
    path = path[0] if isinstance(path, list) else path
    original_trimesh = utils.load_mesh(path, merge_tex=True)

    bsdf_dict = {
        "type": "principled",
        "base_color": {"type": "mesh_attribute", "name": "vertex_color"},
    }
    if twosided:
        bsdf_dict = {"type": "twosided", "material": bsdf_dict}

    bsdf_prop = mi.Properties()
    bsdf_prop["mesh_bsdf"] = mi.load_dict(bsdf_dict)

    mi_mesh = mi.Mesh(
        "mesh",
        vertex_count=data.pos.squeeze().shape[0],
        face_count=original_trimesh.faces.shape[0],
        props=bsdf_prop,
    )

    # Vertex colours were normalised in [-1, 1], bring them back to [0, 1]
    data.x = ((data.x / 2) + 0.5).clamp(0, 1)

    # Vertex color is not a 'built-in' attribute. Needs to be added.
    mi_mesh.add_attribute("vertex_color", 3, data.x.squeeze().numpy().flatten())

    # "Traverse" the mesh to get its updateable parameters
    mesh_params = mi.traverse(mi_mesh)
    mesh_params["vertex_positions"] = data.pos.squeeze().numpy().flatten()
    mesh_params["faces"] = np.array(original_trimesh.faces).flatten()
    return mi_mesh


class PclColoursTexture(mi.Texture):
    """
    Python plugin for mitsuba 3. It allows to store a texture as a point cloud
    instead of as an image. Rays intersecting the surface search for the 3
    nearest neighbours on the point cloud and interpolate their colours to
    determine the colour at the ray intersection.

    This plugin clushes with the efficient mitsuba implementation. Therefore,
    before rendering the megakernel needs to be shut down: call
    mega_kernel(state=False) before rendering.

    The main disadvantage of disabling the megakernel is the GPU memory
    consumption which increases significantly (and remains high even after
    rendering). Flush the cache with flush_cache() after rendering. Given the
    high memory consumption, you may have to flush the torch cache even before
    rendering.
    """

    def __init__(self, props: mi.Properties) -> None:
        mi.Texture.__init__(self, props)
        self._grad_activator = mi.Vector3f(0)
        self.pcl_torch_pos = None
        self.pcl_mi_cols = None

    def traverse(self, callback):
        callback.put_parameter(
            "grad_activator", self._grad_activator, mi.ParamFlags.Differentiable
        )
        callback.put_parameter(
            "pcltex_pos", self.pcl_torch_pos, mi.ParamFlags.NonDifferentiable
        )
        callback.put_parameter(
            "pcltex_color", self.pcl_mi_cols, mi.ParamFlags.Differentiable
        )

    def eval(self, si, active=True, dirs=None, norms=None, albedo=None):
        surface_intersection_position = vec_to_tens_safe(si.p)
        mi_out = self._eval_in_torch(
            surface_intersection_position, self.pcl_mi_cols
        )
        return drjit.unravel(mi.Vector3f, mi_out)

    @drjit.wrap_ad(source="drjit", target="torch")
    def _eval_in_torch(self, pts, pcl_cols):
        # Find k-NN of pcl_torch_pos to pts with k=3 and interpolate colour
        # from colour of 3-NN

        interpolated_cols_torch = knn_interpolate(
            pcl_cols.to(pts.device),
            self.pcl_torch_pos.to(pts.device),
            pts,
            k=3,
        )
        return interpolated_cols_torch

    def eval_1(self, si, active=True):
        return mi.Float(self.eval(si)[0])

    def eval_1_grad(self, *args, **kwargs):
        raise NotImplementedError()

    def eval_3(self, *args, **kwargs):
        raise NotImplementedError()

    def mean(self, *args, **kwargs):
        raise NotImplementedError()

    def to_string(self):
        return "PclColoursTexture"


mi.register_texture("pcl_colours_texture", lambda p: PclColoursTexture(p))


def vec_to_tens_safe(vec):
    # A utility function that converts a Vector3f to a TensorXf safely in
    # mitsuba while keeping the gradients;
    # a regular type cast mi.TensorXf(vector) detaches the gradients
    return mi.TensorXf(
        drjit.ravel(vec), shape=[drjit.shape(vec)[1], drjit.shape(vec)[0]]
    )


def mega_kernel(state: bool = False):
    drjit.set_flag(drjit.JitFlag.LoopRecord, state)
    drjit.set_flag(drjit.JitFlag.VCallRecord, state)
    drjit.set_flag(drjit.JitFlag.VCallOptimize, state)


def flush_cache():
    for _ in range(5):  # Not sure why but calling it once is not enough
        drjit.flush_malloc_cache()


def data_coloured_points_to_mitsuba(
    data: torch_geometric.data.Data, twosided: bool = True
) -> mi.Mesh:
    data = data.detach().cpu()

    path = data.raw_abs_path
    path = path[0] if isinstance(path, list) else path
    original_trimesh = utils.load_mesh(path, merge_tex=True)
    if "verts" in data.keys():
        v_pos = data.verts.squeeze().numpy()
    else:
        original = torch_geometric.data.Data(
            pos=torch.tensor(
                original_trimesh.vertices,
                dtype=torch.float,
                requires_grad=False,
            ).contiguous()
        )
        original = torch_geometric.transforms.NormalizeScale()(original)
        v_pos = original.pos.squeeze().numpy()

    # Vertex colours were normalised in [-1, 1], bring them back to [0, 1]
    pcl_cols = ((data.x / 2) + 0.5).clamp(0, 1)

    pcl_colours_texture = mi.load_dict({"type": "pcl_colours_texture"})
    pcl_colours_texture.pcl_torch_pos = data.pos.squeeze()

    if "cuda" in mi.variant():
        pcl_cols = pcl_cols.cuda()

    pcl_colours_texture.pcl_mi_cols = mi.TensorXf(
        drjit.ravel(mi.TensorXf(pcl_cols.squeeze())),
        shape=pcl_cols.squeeze().shape,
    )

    # May be unnecessary....
    pcl_colours_texture.pcl_torch_pos.requires_grad = True
    drjit.enable_grad(pcl_colours_texture.pcl_mi_cols)
    print(
        f"pcl_mi_cols has grads enabled?",
        f"{drjit.grad_enabled(pcl_colours_texture.pcl_mi_cols)}",
    )
    print(
        f"pcl_torch_pos has grads enabled?",
        f"{pcl_colours_texture.pcl_torch_pos.requires_grad}",
    )

    bsdf_dict = {
        "type": "principled",
        "base_color": pcl_colours_texture,
    }
    if twosided:
        bsdf_dict = {"type": "twosided", "material": bsdf_dict}

    bsdf_prop = mi.Properties()
    bsdf_prop["mesh_bsdf"] = mi.load_dict(bsdf_dict)

    mi_mesh = mi.Mesh(
        "mesh",
        vertex_count=v_pos.shape[0],
        face_count=original_trimesh.faces.shape[0],
        props=bsdf_prop,
    )

    # "Traverse" the mesh to get its updateable parameters
    mesh_params = mi.traverse(mi_mesh)
    drjit.enable_grad(mesh_params["bsdf.brdf_0.base_color.grad_activator"])
    mesh_params["vertex_positions"] = np.array(v_pos).flatten()
    mesh_params["faces"] = np.array(original_trimesh.faces).flatten()
    mesh_params.update()
    return mi_mesh


def mesh_with_pcltex_to_mitsuba(
    data: torch_geometric.data.Data,
    normalise_scale: bool = False,
    twosided: bool = True,
) -> mi.Mesh:
    data = data.detach().cpu()

    if normalise_scale:
        data = torch_geometric.transforms.NormalizeScale()(data)

    verts = utils.to_np(data.verts.squeeze())
    faces = utils.to_np(data.face.squeeze().T)

    # Vertex colours were normalised in [-1, 1], bring them back to [0, 1]
    pcl_cols = ((data.x / 2) + 0.5).clamp(0, 1)

    pcl_colours_texture = mi.load_dict({"type": "pcl_colours_texture"})
    pcl_colours_texture.pcl_torch_pos = data.pos.squeeze()

    if "cuda" in mi.variant():
        pcl_cols = pcl_cols.cuda()

    pcl_colours_texture.pcl_mi_cols = mi.TensorXf(
        drjit.ravel(mi.TensorXf(pcl_cols.squeeze())),
        shape=pcl_cols.squeeze().shape,
    )

    bsdf_dict = {
        "type": "principled",
        "base_color": pcl_colours_texture,
    }
    if twosided:
        bsdf_dict = {"type": "twosided", "material": bsdf_dict}

    bsdf_prop = mi.Properties()
    bsdf_prop["mesh_bsdf"] = mi.load_dict(bsdf_dict)

    mi_mesh = mi.Mesh(
        "mesh",
        vertex_count=verts.shape[0],
        face_count=faces.shape[0],
        props=bsdf_prop,
    )

    # "Traverse" the mesh to get its updateable parameters
    mesh_params = mi.traverse(mi_mesh)
    mesh_params["vertex_positions"] = verts.flatten()
    mesh_params["faces"] = faces.flatten()
    mesh_params.update()
    return mi_mesh


if __name__ == "__main__":
    import os
    import torch

    import mitsuba as mi

    mi.set_variant("cuda_ad_rgb")

    import rendering
    from transforms import VertexColoursFromBaseTexture

    root = "/data/AmazonBerkeleyObjects/original"
    data_path = os.path.join(root, "processed/J/B07BWMSM1J.pt")
    data = torch.load(data_path)
    data.raw_abs_path = os.path.join(root, data.raw_path)
    # data = VertexColoursFromBaseTexture(root)(data)
    # mitsuba_mesh = rendering.data_coloured_verts_to_mitsuba(data)
    mitsuba_mesh = rendering.data_original_texture_to_mitsuba(data)

    scene = mi.load_dict(
        {
            "type": "scene",
            "integrator": define_integrator(),
            "camera": define_camera(2, 30, 60),
            "emitter": define_emitter(),
            "mesh": mitsuba_mesh,
        }
    )
    image = mi.render(scene)
    # plt.axis("off")
    # plt.imshow(image)

    # import mitsuba as mi
    # import matplotlib.pyplot as plt

    # import utils
    # import rendering

    # data = utils.load_mesh_with_pcltex(
    #     pcltex_path="/data/home/sf3018/shapenet/chair_23/multiple_generated/generated_1/c2d0bea1edd835b6e874cd29a3bc467c.ply",
    #     mesh_path="/data/home/sf3018/shapenet/chair_23/multiple_generated/generated_1/c2d0bea1edd835b6e874cd29a3bc467c_mesh.ply"
    # )

    # scene_dict = {
    #     "type": "scene",
    #     "integrator": rendering.define_integrator(),
    #     "camera": rendering.define_camera(
    #         3.5, 210, -50, "perspective", 512, 512, sample_count=64
    #     ),
    #     "emitter": rendering.define_emitter(),
    #     "mesh": rendering.mesh_with_pcltex_to_mitsuba(data),
    # }
    #
    # scene = mi.load_dict(scene_dict)
    # rendering.mega_kernel(False)
    # rendered = mi.render(scene)
    # plt.axis("off")
    # plt.imshow(rendered)
    # rendering.flush_cache()
