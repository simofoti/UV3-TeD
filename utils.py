from typing import Tuple, Any, Dict, List
import warnings

import os
import shutil
import yaml
import math
import trimesh
import torch
import torch_sparse
import robust_laplacian
import scipy.sparse
import scipy.sparse.linalg
import sklearn.neighbors
import torch_geometric.data
import torch_geometric.utils.sparse
import matplotlib.cm
import matplotlib.colors
import numpy as np


# Suppress warnings triggered by Sparse CSR tensor support being in beta state
warnings.filterwarnings("ignore", category=UserWarning)


def get_config(path: str):
    with open(path, "r") as stream:
        config = yaml.safe_load(stream)
    return convert_none(config)


def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        return None
    models = [
        os.path.join(dirname, f)
        for f in os.listdir(dirname)
        if os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f
    ]
    if models is None:
        return None
    models.sort()
    last_model_name = models[-1]
    return last_model_name


def convert_none(data):
    if isinstance(data, list):
        data[:] = [convert_none(i) for i in data]
    elif isinstance(data, dict):
        for k, v in data.items():
            data[k] = convert_none(v)
    return None if data == "None" else data


def in_or_default(ref_dict: Dict[str, Any], keyname: str, default: Any) -> Any:
    val = default
    if keyname in ref_dict:
        val = ref_dict[keyname]
    return val


def mkdir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def rmdir(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)


def load_mesh(
    file_path: str, show: bool = False, merge_tex: bool = True
) -> trimesh.Trimesh:
    scene = trimesh.load(file_path, process=False)

    if hasattr(scene, "graph"):
        geometries = []
        for node_name in scene.graph.nodes_geometry:
            transform, geometry_name = scene.graph[node_name]
            # get a copy of the geometry
            current = scene.geometry[geometry_name].copy()
            if isinstance(current, trimesh.Trimesh):
                # move the geometry vertices into the requested frame
                try:
                    current.apply_transform(transform)
                except RuntimeWarning:
                    print(f"troubles with {file_path}")

                # If there are pre-existing uvs in regions with a uniform colour
                # and no texture the visual concatenation fails.
                # Delete those uvs!
                try:
                    if current.visual.material.baseColorTexture is None:
                        current.visual.uv = None
                except AttributeError:
                    if current.visual.material.image is None:
                        current.visual.uv = None

                # save to our list of meshes
                geometries.append(current)

        if len(geometries) > 1:
            mesh = trimesh.util.concatenate(geometries)
        else:
            mesh = geometries[0]
    else:
        mesh = scene

    trimesh.grouping.merge_vertices(mesh, merge_tex=merge_tex, merge_norm=True)

    if show:
        mesh.show()
    return mesh


def save_pcltex(
    vertices: torch.Tensor,
    face: torch.Tensor,
    pcltex_pos: torch.Tensor,
    pcltex_colours: torch.Tensor,
    out_path: str,
    save_also_mesh: bool = True,
):
    vertices = vertices.squeeze()
    face = face.squeeze()
    pcltex_pos = pcltex_pos.squeeze()
    pcltex_colours = pcltex_colours.squeeze()

    # Generated colours are normalised in [-1, 1], bring them back to [0, 1]
    pcltex_colours = ((pcltex_colours / 2) + 0.5).clamp(0, 1)

    if out_path.endswith(".pt"):
        if save_also_mesh:
            data = torch_geometric.data.Data(
                verts=vertices, face=face, pos=pcltex_pos, x=pcltex_colours
            )
        else:
            data = torch_geometric.data.Data(pos=pcltex_pos, x=pcltex_colours)
        torch.save(data, out_path)
    elif out_path.endswith(".ply"):
        pcl = trimesh.PointCloud(
            vertices=to_np(pcltex_pos),
            colors=to_np(pcltex_colours) * 255,
        )
        pcl.export(out_path)
        if save_also_mesh:
            mesh = trimesh.Trimesh(
                vertices=to_np(vertices), faces=to_np(face.T)
            )
            mesh.export(out_path[:-4] + "_mesh.ply")
    else:
        raise NotImplementedError(
            "The output path should point to a '.pt' or '.ply' file.",
            "The format you selected is not available yet.",
        )


def load_mesh_with_pcltex(
    pcltex_path: str, mesh_path: str | None
) -> torch_geometric.data.Data:
    if pcltex_path.endswith(".pt"):
        data = torch.load(pcltex_path)
    elif pcltex_path.endswith((".ply", ".obj", ".glb")):
        pcl = trimesh.load_mesh(pcltex_path)
        colours = (pcl.colors / 255 - 0.5) * 2  # shift colours in [-1, 1]
        colours = colours[:, :3]
        data = torch_geometric.data.Data(
            pos=to_torch(pcl.vertices), x=to_torch(colours)
        )
    else:
        raise NotImplementedError(
            "Please, provide a pointcloud texture in either '.pt' or",
            "'.ply' format",
        )

    if "verts" not in data.keys():
        if mesh_path is None:
            raise ImportError(
                "The pcltex file did not contain the vertices and faces ",
                "of the mesh on which to render the pcltexture. Please,",
                "provide a path to the mesh",
            )
        if mesh_path.endswith(".pt"):
            mesh_data = torch.load(mesh_path)
            data.verts = mesh_data.verts
            data.face = mesh_data.face
        elif mesh_path.endswith((".ply", ".obj", ".glb")):
            mesh = load_mesh(mesh_path)
            data.verts = to_torch(mesh.vertices)
            data.face = to_torch(mesh.faces).T
        else:
            raise NotImplementedError(
                "Please, provide a mesh in either '.pt'",
                "'.ply', '.obj', or '.glb' format.",
            )
    return data


def to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def to_torch(x: np.ndarray) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float, requires_grad=False).contiguous()


def truncate(values: np.ndarray, decimals: int = 0) -> np.ndarray:
    return np.trunc(values * 10**decimals) / (10**decimals)


def order_of_magnitude(number: float) -> int:
    return math.floor(math.log(number, 10))


def compute_tot_area(pos: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    side_1 = pos[faces[1]] - pos[faces[0]]
    side_2 = pos[faces[2]] - pos[faces[0]]
    return side_1.cross(side_2).norm(p=2, dim=1).abs().sum() / 2


def compute_mesh_laplacian(
    verts: np.ndarray, faces: np.ndarray
) -> Tuple[scipy.sparse.csc_matrix, np.ndarray]:
    lapl, mass = robust_laplacian.mesh_laplacian(verts, faces)
    return lapl, mass.diagonal()


def compute_point_cloud_laplacian(
    points: np.ndarray,
) -> Tuple[scipy.sparse.csc_matrix, np.ndarray]:
    lapl, mass = robust_laplacian.point_cloud_laplacian(points)
    return lapl, mass.diagonal()


def compute_eig_laplacian(
    verts: np.ndarray,
    faces: np.ndarray | None,
    mix_lapl_w: float = 0.05,
    k_eig: int = 10,
    eps: float = 1e-8,
    as_cloud: bool = False,
) -> Tuple[np.ndarray, np.ndarray, scipy.sparse.csc_matrix, np.ndarray]:
    """Compute the eigendecomposition of the Laplacian

    Args:
        verts (np.ndarray): [N x 3] N vertices of the mesh
        faces (np.ndarray | None): [F x 3] F triangular faces of the mesh
        mix_lapl_w (float, optional): if > 0 the point cloud and mesh Laplacians
            are mixed to enable comunication between disconnected components and
            still take into account mesh topology. Defaults to 0.05.
        k_eig (int, optional): number of eigenvalues and eigenvectors desired.
            Defaults to 10.
        eps (float, optional): constant used to perturb Laplacian during
            eigendecomposition. Defaults to 1e-8.
        as_cloud (bool, optional): computes point cloud Laplacian even if a
            mesh is provided. Defaults to False.

    Raises:
        ValueError: although multiple attempts were made, the eigendecomposition
            failed.

    Returns:
        Tuple[np.ndarray, np.ndarray, scipy.sparse.csc_matrix, np.ndarray]:
            k eigenvalues, [k x N] eigenvectors, [N x N] Laplacian, and
            [N] mass vector
    """

    if faces is None or as_cloud or mix_lapl_w == 1:
        lapl, mass = robust_laplacian.point_cloud_laplacian(verts)
        massvec = mass.diagonal()
    else:
        lapl, mass = robust_laplacian.mesh_laplacian(verts, faces)
        massvec = mass.diagonal()

        # Mix mesh and pcl laplacians to enable comunication between
        # disconnected components
        if mix_lapl_w is not None and mix_lapl_w > 0:
            lapl_pcl, mass_pcl = robust_laplacian.point_cloud_laplacian(verts)
            massvec_pcl = mass_pcl.diagonal()

            lapl = (1 - mix_lapl_w) * lapl + mix_lapl_w * lapl_pcl
            massvec = (1 - mix_lapl_w) * massvec + mix_lapl_w * massvec_pcl

    # Prepare matrices for eigendecomposition like in DiffusionNet code
    lapl_eigsh = (lapl + scipy.sparse.identity(lapl.shape[0]) * eps).tocsc()
    mass_mat = scipy.sparse.diags(massvec)
    eigs_sigma = eps

    failcount = 0
    while True:
        try:
            evals, evecs = scipy.sparse.linalg.eigsh(
                lapl_eigsh, k=k_eig, M=mass_mat, sigma=eigs_sigma
            )
            evals = np.clip(evals, a_min=0.0, a_max=float("inf"))
            break
        except RuntimeError as exc:
            if failcount > 3:
                raise ValueError("failed to compute eigendecomp") from exc
            failcount += 1
            print("--- decomp failed; adding eps ===> count: " + str(failcount))
            lapl_eigsh = lapl_eigsh + scipy.sparse.identity(lapl.shape[0]) * (
                eps * 10**failcount
            )
    return evals, evecs, lapl, massvec


def build_grad(
    edge_index: torch.Tensor,
    edge_tangent_vectors: torch.Tensor,
) -> scipy.sparse.coo_matrix:
    edge_head_and_tangent = get_edge_head_and_tangent(
        edge_index, edge_tangent_vectors, return_np=True
    )

    n_verts = len(edge_head_and_tangent)
    row_inds = []
    col_inds = []
    data_vals = []
    w_e = 1.0
    eps_reg = 1e-5
    for i_v, neigh_info in enumerate(edge_head_and_tangent):
        n_neigh = neigh_info.shape[0]
        lhs = w_e * neigh_info[:, 1:]
        lhs_t = lhs.T
        rhs = w_e * np.concatenate(
            [np.array([[-1]] * n_neigh), np.identity(n_neigh)], axis=1
        )
        lhs_inv = np.linalg.inv(lhs_t @ lhs + eps_reg * np.identity(2)) @ lhs_t

        sol_mat = lhs_inv @ rhs
        sol_coefs = sol_mat[0, :] + 1j * sol_mat[1, :]

        row_inds.extend([i_v] * (n_neigh + 1))
        col_inds.extend([i_v] + neigh_info[:, 0].tolist())
        data_vals.extend(sol_coefs.tolist())
    row_inds = np.array(row_inds)
    col_inds = np.array(col_inds)
    data_vals = np.array(data_vals)
    mat = scipy.sparse.coo_matrix(
        (data_vals, (row_inds, col_inds)), shape=(n_verts, n_verts)
    ).tocsc()
    return mat


def build_grad_batched(
    edge_index: torch.Tensor,
    edge_tangent_vectors: torch.Tensor,
    device: str = "cpu",
) -> scipy.sparse.coo_matrix:
    device = torch.device(device)

    edge_head_and_tangent = torch.stack(
        get_edge_head_and_tangent(
            edge_index, edge_tangent_vectors, return_np=False
        ),
    ).to(device)

    w_e = 1.0
    eps_reg = 1e-5

    n_verts = edge_head_and_tangent.shape[0]
    n_neigh = edge_head_and_tangent.shape[1]

    col_inds = torch.cat(
        [
            torch.arange(n_verts).view(-1, 1),
            edge_head_and_tangent[:, :, 0].cpu(),
        ],
        dim=1,
    )
    col_inds = col_inds.flatten().numpy()

    row_inds = torch.arange(n_verts).unsqueeze(1).expand(-1, n_neigh + 1)
    row_inds = to_np(row_inds.flatten())

    lhs = w_e * edge_head_and_tangent[:, :, 1:]
    rhs = w_e * torch.cat(
        [torch.tensor([[-1]] * n_neigh), torch.eye(n_neigh)], dim=1
    ).to(device)
    lhs_t = lhs.transpose(1, 2)
    eye = torch.eye(2, device=device)
    lhs_inv = torch.linalg.inv(lhs_t @ lhs + eps_reg * eye) @ lhs_t
    sol_mat = lhs_inv @ rhs
    sol_coefs = sol_mat[:, 0, :] + 1j * sol_mat[:, 1, :]
    data_vals = to_np(sol_coefs.flatten())

    mat = scipy.sparse.coo_matrix(
        (data_vals, (row_inds, col_inds)), shape=(n_verts, n_verts)
    ).tocsc()
    return mat


def get_edge_head_and_tangent(
    edge_index: torch.Tensor,
    edge_tangent_vectors: torch.Tensor,
    return_np: bool = False,
) -> List[torch.Tensor] | List[np.ndarray]:
    """
    For each vertex find the indices of all neighbouring vertices (heads) and
    the edges connecting them expressed in the coordinates of the tangent frame
    of the tail vertex.

    Args:
        edge_index (torch.Tensor): shape [2, E]
        edge_tangent_vectors (torch.Tensor): edges in coordinates of tangent
            frame. Shape [E, 2]

    Returns:
        List[torch.Tensor] | List[np.ndarray]: each tensor in the list
            corresponds to a vertex. It has many rows as neighbours and the
            first element of each row is the index of the vertex to which it
            is connected, while the remaining two elements are the tangent
            vectors of the edge.
    """
    csr = torch_geometric.utils.sparse.to_torch_csr_tensor(edge_index)
    csr_tang_x = torch_geometric.utils.sparse.to_torch_csr_tensor(
        edge_index, edge_tangent_vectors[:, 0]
    )
    csr_tang_y = torch_geometric.utils.sparse.to_torch_csr_tensor(
        edge_index, edge_tangent_vectors[:, 1]
    )
    triplets = torch.stack(
        [csr.col_indices(), csr_tang_x.values(), csr_tang_y.values()]
    ).T

    if return_np:
        triplets = triplets.numpy()
    # NB: vvvv not ideal when splitting tensors with a np functions.
    return np.split(triplets, csr.crow_indices()[1:-1])


def packed_to_padded(
    x: torch.Tensor,
    x_mask: torch.Tensor,
    max_number_vertices: int,
    batch_size: int,
) -> torch.Tensor:
    ch = x.shape[-1]
    x_padded = x.new_full((batch_size * max_number_vertices, ch), 0.0)
    x_padded[x_mask == True] = x
    return x_padded.view((batch_size, -1, ch))


def padded_to_packed(
    x: torch.Tensor,
    padded_to_packed_idx: torch.Tensor,
) -> torch.Tensor:
    x_packed = x.reshape(-1, x.shape[-1])
    return x_packed[padded_to_packed_idx]


def stack_list_padded_sparse(
    lx: (
        List[scipy.sparse.coo_matrix]
        | List[scipy.sparse.csc_matrix]
        | List[torch_sparse.SparseTensor]
    ),
    new_side_size: int | None = None,
) -> torch.sparse.FloatTensor:
    if isinstance(lx[0], torch_sparse.SparseTensor):
        stacked = stack_list_padded_torch_sparse(lx, new_side_size)
    elif isinstance(lx[0], (scipy.sparse.csc_matrix, scipy.sparse.coo_matrix)):
        stacked = stack_list_padded_scipy_sparse(lx, new_side_size)
    return stacked


def stack_list_padded_scipy_sparse(
    lx: List[scipy.sparse.coo_matrix] | List[scipy.sparse.csc_matrix],
    new_side_size: int | None = None,
) -> torch.sparse.FloatTensor:
    if new_side_size is None:
        stacked = torch.stack([sparse_np_to_torch(x) for x in lx])
    else:
        resized_tensors = []
        for x in lx:
            mat_coo = x.tocoo()
            values = mat_coo.data
            indices = np.vstack((mat_coo.row, mat_coo.col))
            shape = (new_side_size, new_side_size)
            t = torch.sparse.FloatTensor(
                torch.LongTensor(indices),
                torch.FloatTensor(values),
                torch.Size(shape),
            ).coalesce()
            resized_tensors.append(t)
        stacked = torch.stack(resized_tensors)
    return stacked


def stack_list_padded_torch_sparse(
    lx: List[torch_sparse.SparseTensor],
    new_side_size: int | None = None,
) -> torch.sparse.FloatTensor:
    if new_side_size is None:
        stacked = torch.stack([x.to_torch_sparse_coo_tensor() for x in lx])
    else:
        resized_tensors = []
        for x in lx:
            mat_coo = x.to_torch_sparse_coo_tensor().coalesce()
            shape = (new_side_size, new_side_size)
            t = torch.sparse.FloatTensor(
                mat_coo.indices(),
                mat_coo.values(),
                torch.Size(shape),
            ).coalesce()
            resized_tensors.append(t)
        stacked = torch.stack(resized_tensors)
    return stacked


def estimate_poisson_scaled_texture_size(
    faces: torch.Tensor,
    uv: torch.Tensor,
    pos: torch.Tensor,
    squared_poisson_radius: torch.Tensor,
    tex_w_h: Tuple[int],
) -> torch.Tensor:
    z_coordinate = torch.zeros([uv.shape[0], 1], device=uv.device)
    uv_areas = face_area(torch.cat([uv, z_coordinate], dim=-1), faces)

    k = min(250, faces.shape[0])
    biggest_uv_faces_areas, biggest_uv_faces_idx = torch.topk(uv_areas, k=k)
    area_3d_of_biggest_uv = face_area(pos, faces[biggest_uv_faces_idx, :])

    avg_area_3d_big_uv = torch.mean(area_3d_of_biggest_uv)
    approx_area_sampled = (
        squared_poisson_radius * torch.sin(torch.pi / torch.tensor(3)) / 2
    )
    approx_pts_in_highest_res_faces = avg_area_3d_big_uv / approx_area_sampled

    # area of triangle in uv is in a square 1x1 => num pixels in triangle
    # is equal to tot n pixels in texture * area of triangle / (1 *1)
    tex_pixels = tex_w_h[0] * tex_w_h[1]
    approx_pixels_biggest_uv = tex_pixels * torch.mean(biggest_uv_faces_areas)

    scale_area = approx_pts_in_highest_res_faces / approx_pixels_biggest_uv
    # bit bigger scale because don't know where I am sampling exactly
    scale = torch.sqrt(scale_area) * 3
    scale = scale.clip(0.05, 1.0).item()

    # make sure smallest side is at least 100 pixels
    w, h = round(scale * tex_w_h[0]), round(scale * tex_w_h[1])
    min_side = min(w, h)
    if min_side < 100:
        w, h = round(w / min_side * 100), round(h / min_side * 100)
    return w, h


def values_to_cmap(
    values: torch.Tensor,
    cmap: str | matplotlib.colors.Colormap | List[str] | None = None,
    min_value: float | None = None,
    max_value: float | None = None,
) -> torch.Tensor:
    device = values.device
    min_value = values.min() if min_value is None else min_value
    max_value = values.max() if max_value is None else max_value
    if min_value != max_value:
        values = (values - min_value) / (max_value - min_value)

    if cmap is None:
        cmap = "Reds"
    if isinstance(cmap, str):
        cmapper = matplotlib.cm.get_cmap(cmap)
    elif isinstance(cmap, list):
        cmapper = matplotlib.colors.ListedColormap(cmap)
    else:
        assert isinstance(cmap, matplotlib.colors.Colormap)
        cmapper = cmap
    values = cmapper(values.cpu().detach().numpy(), bytes=False)[:, :, :3]
    return torch.tensor(values, device=device, requires_grad=False)


def get_color_iterator():
    cmap10 = [matplotlib.cm.get_cmap("tab10")(i) for i in range(20)]
    cmap20b = [matplotlib.cm.get_cmap("tab20b")(i) for i in range(20)]
    cmap20c = [matplotlib.cm.get_cmap("tab20c")(i) for i in range(20)]
    return iter([*cmap10, *cmap20b, *cmap20c])


def get_rgb_color(color_name: str) -> torch.Tensor:
    return torch.tensor(matplotlib.colors.to_rgb(color_name))


def compute_bounding_box_sides(positions: torch.Tensor) -> torch.Tensor:
    min_pos = torch.min(positions, dim=0).values
    max_pos = torch.max(positions, dim=0).values
    sides = torch.abs(max_pos - min_pos)
    return sides.to(torch.float)


# From Diffusion Net repository ################################################


def get_grad_operators(
    verts: torch.Tensor,
    faces: torch.Tensor,
    edges: torch.Tensor,
    normals: torch.Tensor,
    as_cloud: bool = False,
) -> Tuple[torch_sparse.SparseTensor, torch_sparse.SparseTensor]:
    # For meshes, we use the same edges as were used to build the Laplacian.
    # For point clouds, use a whole local neighborhood
    frames = build_tangent_frames(verts, faces, normals=normals)
    if as_cloud or (faces is None and edges is None):
        grad_mat_np = build_grad_point_cloud(verts, frames)
    else:
        edge_vecs = edge_tangent_vectors(verts, frames, edges)
        grad_mat_np = build_grad(edges, edge_vecs)

    # Split complex gradient in to two real sparse mats (torch doesn't like
    # complex sparse matrices)
    gradX_np = np.real(grad_mat_np)
    gradY_np = np.imag(grad_mat_np)

    # === Convert back to torch
    grad_x = sparse_np_to_torch_sparse(gradX_np)
    grad_y = sparse_np_to_torch_sparse(gradY_np)

    return grad_x, grad_y


def sparse_np_to_torch_sparse(
    mat: scipy.sparse.coo_matrix | scipy.sparse.csc_matrix,
) -> torch_sparse.SparseTensor:
    mat_coo = mat.tocoo()
    return torch_sparse.SparseTensor.from_scipy(mat_coo)


def sparse_np_to_torch(
    mat: scipy.sparse.coo_matrix | scipy.sparse.csc_matrix,
) -> torch.sparse.FloatTensor:
    mat_coo = mat.tocoo()
    values = mat_coo.data
    indices = np.vstack((mat_coo.row, mat_coo.col))
    shape = mat_coo.shape
    return torch.sparse.FloatTensor(
        torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(shape)
    ).coalesce()


def sparse_torch_to_np(
    mat: torch.sparse.FloatTensor,
) -> scipy.sparse.csc_matrix:
    if len(mat.shape) != 2:
        raise RuntimeError(
            "should be a matrix-shaped type; dim is : " + str(mat.shape)
        )

    indices = to_np(mat.indices())
    values = to_np(mat.values())

    mat = scipy.sparse.coo_matrix((values, indices), shape=mat.shape).tocsc()

    return mat


def cross(vec_1: torch.Tensor, vec_2: torch.Tensor) -> torch.Tensor:
    return torch.cross(vec_1, vec_2, dim=-1)


def dot(vec_1: torch.Tensor, vec_2: torch.Tensor) -> torch.Tensor:
    return torch.sum(vec_1 * vec_2, dim=-1)


def norm(x: torch.Tensor) -> torch.Tensor:
    """
    Computes norm of an array of vectors. Given (shape,d), returns (shape)
    after norm along last dimension
    """
    return torch.norm(x, dim=len(x.shape) - 1)


def norm2(x: torch.Tensor) -> torch.Tensor:
    """
    Computes norm^2 of an array of vectors. Given (shape,d), returns (shape)
    after norm along last dimension
    """
    return dot(x, x)


def normalize(
    x: torch.Tensor, divide_eps: float = 1e-6, highdim: bool = False
) -> torch.Tensor:
    """
    Computes norm^2 of an array of vectors. Given (shape,d), returns (shape)
    after norm along last dimension
    """
    if len(x.shape) == 1:
        raise ValueError(
            "called normalize() on single vector of dim "
            + str(x.shape)
            + " are you sure?"
        )
    if not highdim and x.shape[-1] > 4:
        raise ValueError(
            "called normalize() with large last dimension "
            + str(x.shape)
            + " are you sure?"
        )
    return x / (norm(x) + divide_eps).unsqueeze(-1)


def project_to_tangent(
    vecs: torch.Tensor, unit_normals: torch.Tensor
) -> torch.Tensor:
    """
    Given (..., 3) vectors and normals, projects out any components of vecs
    which lies in the direction of normals. Normals are assumed to be unit.
    """
    dots = dot(vecs, unit_normals)
    return vecs - unit_normals * dots.unsqueeze(-1)


def face_coords(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    coords = verts[faces]
    return coords


def face_area(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    coords = face_coords(verts, faces)
    vec_1 = coords[:, 1, :] - coords[:, 0, :]
    vec_2 = coords[:, 2, :] - coords[:, 0, :]

    raw_normal = cross(vec_1, vec_2)
    return 0.5 * norm(raw_normal)


def face_normals(
    verts: torch.Tensor, faces: torch.Tensor, normalized: bool = True
) -> torch.Tensor:
    coords = face_coords(verts, faces)
    vec_1 = coords[:, 1, :] - coords[:, 0, :]
    vec_2 = coords[:, 2, :] - coords[:, 0, :]

    raw_normal = cross(vec_1, vec_2)

    if normalized:
        return normalize(raw_normal)

    return raw_normal


def neighborhood_normal(points: np.ndarray) -> np.ndarray:
    # points: (N, K, 3) array of neighborhood psoitions
    # points should be centered at origin
    # out: (N,3) array of normals
    (u, s, vh) = np.linalg.svd(points, full_matrices=False)
    normal = vh[:, 2, :]
    return normal / np.linalg.norm(normal, axis=-1, keepdims=True)


def mesh_vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    face_n = to_np(
        face_normals(torch.tensor(verts), torch.tensor(faces))
    )  # ugly torch <---> numpy

    vertex_normals = np.zeros(verts.shape)
    for i in range(3):
        np.add.at(vertex_normals, faces[:, i], face_n)

    vertex_normals = vertex_normals / np.linalg.norm(
        vertex_normals, axis=-1, keepdims=True
    )

    return vertex_normals


def vertex_normals(
    verts: torch.Tensor, faces: torch.Tensor, n_neighbors_cloud: int = 30
) -> torch.Tensor:
    verts_np = to_np(verts)

    if faces.numel() == 0:  # point cloud
        _, neigh_inds = find_knn(
            verts, verts, n_neighbors_cloud, omit_diagonal=True, method="cpu_kd"
        )
        neigh_points = verts_np[neigh_inds, :]
        neigh_points = neigh_points - verts_np[:, np.newaxis, :]
        normals = neighborhood_normal(neigh_points)

    else:  # mesh
        normals = mesh_vertex_normals(verts_np, to_np(faces))

        # if any are NaN, wiggle slightly and recompute
        bad_normals_mask = np.isnan(normals).any(axis=1, keepdims=True)
        if bad_normals_mask.any():
            bbox = np.amax(verts_np, axis=0) - np.amin(verts_np, axis=0)
            scale = np.linalg.norm(bbox) * 1e-4
            wiggle = (
                np.random.RandomState(seed=777).rand(*verts.shape) - 0.5
            ) * scale
            wiggle_verts = verts_np + bad_normals_mask * wiggle
            normals = mesh_vertex_normals(wiggle_verts, to_np(faces))

        # if still NaN assign random normals (probably means unreferenced
        # verts in mesh)
        bad_normals_mask = np.isnan(normals).any(axis=1)
        if bad_normals_mask.any():
            normals[bad_normals_mask, :] = (
                np.random.RandomState(seed=777).rand(*verts.shape) - 0.5
            )[bad_normals_mask, :]
            normals = normals / np.linalg.norm(normals, axis=-1)[:, np.newaxis]

    normals = torch.from_numpy(normals).to(
        device=verts.device, dtype=verts.dtype
    )

    if torch.any(torch.isnan(normals)):
        raise ValueError("NaN normals :(")

    return normals


def build_tangent_frames(
    verts: torch.Tensor,
    faces: torch.Tensor,
    normals: torch.Tensor | None = None,
) -> torch.Tensor:
    V = verts.shape[0]
    dtype = verts.dtype
    device = verts.device

    if normals == None:
        vert_normals = vertex_normals(verts, faces)  # (V,3)
    else:
        vert_normals = normals

    # = find an orthogonal basis

    basis_cand1 = (
        torch.tensor([1, 0, 0]).to(device=device, dtype=dtype).expand(V, -1)
    )
    basis_cand2 = (
        torch.tensor([0, 1, 0]).to(device=device, dtype=dtype).expand(V, -1)
    )

    basis_x = torch.where(
        (torch.abs(dot(vert_normals, basis_cand1)) < 0.9).unsqueeze(-1),
        basis_cand1,
        basis_cand2,
    )
    basis_x = project_to_tangent(basis_x, vert_normals)
    basis_x = normalize(basis_x)
    basis_y = cross(vert_normals, basis_x)
    frames = torch.stack((basis_x, basis_y, vert_normals), dim=-2)

    if torch.any(torch.isnan(frames)):
        raise ValueError("NaN coordinate frame! Must be very degenerate")

    return frames


def build_grad_point_cloud(
    verts: torch.Tensor, frames: torch.Tensor, n_neighbors_cloud: int = 30
) -> torch.Tensor:
    _, neigh_inds = find_knn(
        verts, verts, n_neighbors_cloud, omit_diagonal=True, method="cpu_kd"
    )

    edge_inds_from = np.repeat(np.arange(verts.shape[0]), n_neighbors_cloud)
    edges = np.stack((edge_inds_from, neigh_inds.flatten()))
    edge_tangent_vecs = edge_tangent_vectors(verts, frames, edges)

    return build_grad_batched(torch.tensor(edges), edge_tangent_vecs, "cpu")


def edge_tangent_vectors(
    verts: torch.Tensor, frames: torch.Tensor, edges: torch.Tensor
) -> torch.Tensor:
    edge_vecs = verts[edges[1, :], :] - verts[edges[0, :], :]
    basis_x = frames[edges[0, :], 0, :]
    basis_y = frames[edges[0, :], 1, :]

    compX = dot(edge_vecs, basis_x)
    compY = dot(edge_vecs, basis_y)
    edge_tangent = torch.stack((compX, compY), dim=-1)

    return edge_tangent


def build_grad_original_implementation(
    verts: np.ndarray, edges: torch.Tensor, edge_tangent_vectors: torch.Tensor
) -> scipy.sparse.coo_matrix:
    """
    Build a (V, V) complex sparse matrix grad operator. Given real inputs at
    vertices, produces a complex (vector value) at vertices giving the gradient.
    All values pointwise.
    - edges: (2, E)
    """

    edges_np = to_np(edges)

    # Build outgoing neighbor lists
    N = verts.shape[0]
    vert_edge_outgoing = [[] for i in range(N)]
    for i_e in range(edges_np.shape[1]):
        tail_ind = edges_np[0, i_e]
        tip_ind = edges_np[1, i_e]
        if tip_ind != tail_ind:
            vert_edge_outgoing[tail_ind].append(i_e)

    # Build local inversion matrix for each vertex
    row_inds = []
    col_inds = []
    data_vals = []
    eps_reg = 1e-5
    for i_v in range(N):
        n_neigh = len(vert_edge_outgoing[i_v])

        lhs_mat = np.zeros((n_neigh, 2))
        rhs_mat = np.zeros((n_neigh, n_neigh + 1))
        ind_lookup = [i_v]
        for i_neigh in range(n_neigh):
            i_e = vert_edge_outgoing[i_v][i_neigh]
            j_v = edges_np[1, i_e]
            ind_lookup.append(j_v)

            edge_vec = edge_tangent_vectors[i_e][:]
            w_e = 1.0

            lhs_mat[i_neigh][:] = w_e * edge_vec
            rhs_mat[i_neigh][0] = w_e * (-1)
            rhs_mat[i_neigh][i_neigh + 1] = w_e * 1

        lhs_T = lhs_mat.T
        lhs_inv = (
            np.linalg.inv(lhs_T @ lhs_mat + eps_reg * np.identity(2)) @ lhs_T
        )

        sol_mat = lhs_inv @ rhs_mat
        sol_coefs = (sol_mat[0, :] + 1j * sol_mat[1, :]).T

        for i_neigh in range(n_neigh + 1):
            i_glob = ind_lookup[i_neigh]

            row_inds.append(i_v)
            col_inds.append(i_glob)
            data_vals.append(sol_coefs[i_neigh])

    # build the sparse matrix
    row_inds = np.array(row_inds)
    col_inds = np.array(col_inds)
    data_vals = np.array(data_vals)
    mat = scipy.sparse.coo_matrix(
        (data_vals, (row_inds, col_inds)), shape=(N, N)
    ).tocsc()
    return mat


def to_basis(
    values: torch.Tensor, basis: torch.Tensor, massvec: torch.Tensor
) -> torch.Tensor:
    """
    Transform data in to an orthonormal basis (where orthonormal
    is wrt to massvec)
    Inputs:
      - values: (B,V,D)
      - basis: (B,V,K)
      - massvec: (B,V)
    Outputs:
      - (B,K,D) transformed values
    """
    basisT = basis.transpose(-2, -1)
    return torch.matmul(basisT, values * massvec.unsqueeze(-1))


def from_basis(values: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """
    Transform data out of an orthonormal basis
    Inputs:
      - values: (K,D)
      - basis: (V,K)
    Outputs:
      - (V,D) reconstructed values
    """
    if values.is_complex() or basis.is_complex():
        raise ValueError
    return torch.matmul(basis, values)


def compute_hks(
    evals: torch.Tensor, evecs: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    """
    Inputs:
      - evals: (K) eigenvalues
      - evecs: (V,K) values
      - scales: (S) times
    Outputs:
      - (V,S) hks values
    """

    # expand batch
    if len(evals.shape) == 1:
        expand_batch = True
        evals = evals.unsqueeze(0)
        evecs = evecs.unsqueeze(0)
        scales = scales.unsqueeze(0)
    else:
        expand_batch = False

    # TODO could be a matmul
    power_coefs = torch.exp(
        -evals.unsqueeze(1) * scales.unsqueeze(-1)
    ).unsqueeze(
        1
    )  # (B,1,S,K)
    terms = power_coefs * (evecs * evecs).unsqueeze(2)  # (B,V,S,K)

    out = torch.sum(terms, dim=-1)  # (B,V,S)

    if expand_batch:
        return out.squeeze(0)
    else:
        return out


def compute_hks_autoscale(
    evals: torch.Tensor, evecs: torch.Tensor, count: int
) -> torch.Tensor:
    # these scales roughly approximate those suggested in the hks paper
    scales = torch.logspace(
        -2, 0.0, steps=count, device=evals.device, dtype=evals.dtype
    )
    return compute_hks(evals, evecs, scales)


def find_knn(
    points_source: torch.Tensor,
    points_target: torch.Tensor,
    k: int,
    largest: bool = False,
    omit_diagonal: bool = False,
    method: str = "brute",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Finds the k nearest neighbors of source on target.
    Return is two tensors (distances, indices). Returned points will be sorted
    in increasing order of distance.
    """
    if omit_diagonal and points_source.shape[0] != points_target.shape[0]:
        raise ValueError(
            "omit_diagonal can only be used when source and target ",
            "are same shape",
        )

    if (
        method != "cpu_kd"
        and points_source.shape[0] * points_target.shape[0] > 1e8
    ):
        method = "cpu_kd"
        print("switching to cpu_kd knn")

    if method == "brute":
        # Expand so both are NxMx3 tensor
        points_source_expand = points_source.unsqueeze(1)
        points_source_expand = points_source_expand.expand(
            -1, points_target.shape[0], -1
        )
        points_target_expand = points_target.unsqueeze(0)
        points_target_expand = points_target_expand.expand(
            points_source.shape[0], -1, -1
        )

        diff_mat = points_source_expand - points_target_expand
        dist_mat = norm(diff_mat)

        if omit_diagonal:
            torch.diagonal(dist_mat)[:] = float("inf")

        result = torch.topk(dist_mat, k=k, largest=largest, sorted=True)
        return result

    elif method == "cpu_kd":
        if largest:
            raise ValueError("can't do largest with cpu_kd")

        points_source_np = to_np(points_source)
        points_target_np = to_np(points_target)

        # Build the tree
        kd_tree = sklearn.neighbors.KDTree(points_target_np)

        k_search = k + 1 if omit_diagonal else k
        _, neighbors = kd_tree.query(points_source_np, k=k_search)

        if omit_diagonal:
            # Mask out self element
            mask = neighbors != np.arange(neighbors.shape[0])[:, np.newaxis]

            # make sure we mask out exactly one element in each row, in rare
            # case of many duplicate points
            mask[np.sum(mask, axis=1) == mask.shape[1], -1] = False

            neighbors = neighbors[mask].reshape(
                (neighbors.shape[0], neighbors.shape[1] - 1)
            )

        inds = torch.tensor(
            neighbors, device=points_source.device, dtype=torch.int64
        )
        dists = norm(
            points_source.unsqueeze(1).expand(-1, k, -1) - points_target[inds]
        )

        return dists, inds

    else:
        raise ValueError("unrecognized method")


def farthest_point_sampling(
    points: torch.Tensor, n_sample: int
) -> torch.Tensor:
    # Torch in, torch out. Returns a |V| mask with n_sample elements set to true

    N = points.shape[0]
    if n_sample > N:
        raise ValueError("not enough points to sample")

    chosen_mask = torch.zeros(N, dtype=torch.bool, device=points.device)
    min_dists = torch.ones(N, dtype=points.dtype, device=points.device) * float(
        "inf"
    )

    # pick the centermost first point
    # points = normalize_positions(points)  # they should be already centered
    i = torch.min(norm2(points), dim=0).indices
    chosen_mask[i] = True

    for _ in range(n_sample - 1):
        # update distance
        dists = norm2(points[i, :].unsqueeze(0) - points)
        min_dists = torch.minimum(dists, min_dists)

        # take the farthest
        i = torch.max(min_dists, dim=0).indices.item()
        chosen_mask[i] = True

    return chosen_mask
