from typing import List, Callable

import scipy
import scipy.sparse.linalg as sla

# ^^^ we NEED to import scipy before torch, or it crashes :(
# (observed on Ubuntu 20.04 w/ torch 1.6.0 and scipy 1.5.2 installed via conda)

import torch

from dataclasses import dataclass
from diffusers.models.embeddings import Timesteps
from diffusers.utils.outputs import BaseOutput
from torch_geometric.typing import SparseTensor

import utils


class LearnedTimeDiffusion(torch.nn.Module):
    """
    Applies diffusion with learned per-channel t.

    In the spectral domain this becomes
        f_out = e ^ (lambda_i t) f_in
    """

    def __init__(self, in_out_channels: int):
        super(LearnedTimeDiffusion, self).__init__()
        self.ch_inout = in_out_channels
        self.diffusion_time = torch.nn.Parameter(torch.Tensor(in_out_channels))

        torch.nn.init.constant_(self.diffusion_time, 0.01)

    def forward(
        self,
        x: torch.Tensor,
        mass: torch.Tensor,
        evals: torch.Tensor,
        evecs: torch.Tensor,
        batch_conversion_info: dict | None = None,
    ) -> torch.Tensor:
        # project times to the positive halfspace
        # (and away from 0 in the incredibly rare chance that they get stuck)
        with torch.no_grad():
            self.diffusion_time.data = torch.clamp(
                self.diffusion_time, min=1e-8
            )

        if x.shape[-1] != self.ch_inout:
            raise ValueError(
                f"Tensor has wrong shape = {x.shape}. Last dim shape should "
                f"have number of channels = {self.ch_inout}"
            )

        if batch_conversion_info is not None:  # batched packing
            # Convert x packed to zero padded
            x = utils.packed_to_padded(
                x,
                batch_conversion_info["to_padded_mask"],
                batch_conversion_info["max_verts"],
                batch_conversion_info["num_graphs"],
            )

        # Transform to spectral
        x_spec = utils.to_basis(x, evecs, mass)

        # Diffuse
        time = self.diffusion_time
        diffusion_coefs = torch.exp(-evals.unsqueeze(-1) * time.unsqueeze(0))
        x_diffuse_spec = diffusion_coefs * x_spec

        # Transform back to per-vertex
        x_diffuse = utils.from_basis(x_diffuse_spec, evecs)

        if batch_conversion_info is not None:  # batched packing
            # Convert x_diffuse from zero padded to packed
            x_diffuse = utils.padded_to_packed(
                x_diffuse, batch_conversion_info["to_packed_idx"]
            ).unsqueeze(0)

        return x_diffuse


class SpatialGradientFeatures(torch.nn.Module):
    """
    Compute dot-products between input vectors. Uses a learned complex-linear
    layer to keep dimension down.
    """

    def __init__(self, in_out_channels: int, with_gradient_rotations=True):
        super(SpatialGradientFeatures, self).__init__()

        self.c_inout = in_out_channels
        self.with_gradient_rotations = with_gradient_rotations

        self.A_re = torch.nn.Linear(self.c_inout, self.c_inout, bias=False)
        if with_gradient_rotations:
            self.A_im = torch.nn.Linear(self.c_inout, self.c_inout, bias=False)

    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vectors (torch.Tensor): (V,C,2)

        Returns:
            torch.Tensor: (V,C) dots
        """
        vA = vectors  # (V,C)

        if self.with_gradient_rotations:
            vBreal = self.A_re(vectors[..., 0]) - self.A_im(vectors[..., 1])
            vBimag = self.A_re(vectors[..., 1]) + self.A_im(vectors[..., 0])
        else:
            vBreal = self.A_re(vectors[..., 0])
            vBimag = self.A_re(vectors[..., 1])

        dots = vA[..., 0] * vBreal + vA[..., 1] * vBimag

        return torch.tanh(dots)


class MiniMLP(torch.nn.Sequential):
    """
    A simple MLP with configurable hidden layer sizes.
    """

    def __init__(
        self,
        layer_sizes: List[int],
        dropout: bool = False,
        activation: Callable = torch.nn.ReLU,
        name: str = "miniMLP",
    ):
        super(MiniMLP, self).__init__()

        for i in range(len(layer_sizes) - 1):
            is_last = i + 2 == len(layer_sizes)

            if dropout and i > 0:
                self.add_module(
                    name + "_mlp_layer_dropout_{:03d}".format(i),
                    torch.nn.Dropout(p=0.5),
                )

            # Affine map
            self.add_module(
                name + "_mlp_layer_{:03d}".format(i),
                torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1]),
            )

            # Nonlinearity (but not on the last layer)
            if not is_last:
                self.add_module(
                    name + "_mlp_act_{:03d}".format(i), activation()
                )


class DiffusedFarthestAttention(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        attention_inner_channels: int | None = None,
        n_heads: int = 8,
    ):
        super(DiffusedFarthestAttention, self).__init__()
        self._out_ch = out_channels
        self._n_heads = n_heads
        assert in_channels % n_heads == 0, "in_ch must be divisible by n_heads"

        if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            raise ImportError(
                "AttnProcessor2_0 requires PyTorch 2.0 or newer",
                "to use it, please upgrade PyTorch",
            )
        # self.ch_inout = in_out_channels
        if attention_inner_channels is None:
            self.att_inner_ch = in_channels
        else:
            self.att_inner_ch = attention_inner_channels

        # Diffusions
        self.diffusion_in_time = torch.nn.Parameter(torch.Tensor(in_channels))
        self.diffusion_out_time = torch.nn.Parameter(torch.Tensor(out_channels))

        torch.nn.init.constant_(self.diffusion_in_time, 0.01)
        torch.nn.init.constant_(self.diffusion_out_time, 0.01)

        # Attention
        self.group_norm = torch.nn.GroupNorm(
            num_channels=in_channels,
            num_groups=32,
            eps=1e-6,
            affine=True,
        )

        self.to_query = torch.nn.Linear(in_channels, self.att_inner_ch)
        self.to_key = torch.nn.Linear(in_channels, self.att_inner_ch)
        self.to_value = torch.nn.Linear(in_channels, self.att_inner_ch)

        self.to_out = torch.nn.Linear(self.att_inner_ch, out_channels)

        # Learn how to weight each out channel
        self.out_weight = torch.nn.Parameter(torch.Tensor(out_channels))
        torch.nn.init.constant_(self.out_weight, 1.0)


    def _heat_diffusion(
        self,
        x: torch.Tensor,
        mass: torch.Tensor,
        evals: torch.Tensor,
        evecs: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        # Transform to spectral
        x_spec = utils.to_basis(x, evecs, mass)

        # Diffuse
        diffusion_coefs = torch.exp(-evals.unsqueeze(-1) * time.unsqueeze(0))
        x_diffuse_spec = diffusion_coefs * x_spec

        # Transform back to per-vertex
        x_diffuse = utils.from_basis(x_diffuse_spec, evecs)
        return x_diffuse

    def forward(
        self,
        x: torch.Tensor,
        mass: torch.Tensor,
        evals: torch.Tensor,
        evecs: torch.Tensor,
        farthest_sampling_mask: torch.Tensor,
        batch_conversion_info: dict | None = None,
    ) -> torch.Tensor:
        # project times and out weight to the positive halfspace
        # (and away from 0 in the incredibly rare chance that they get stuck)
        with torch.no_grad():
            self.diffusion_in_time.data = torch.clamp(
                self.diffusion_in_time, min=1e-8
            )
            self.diffusion_out_time.data = torch.clamp(
                self.diffusion_out_time, min=1e-8
            )
            self.out_weight.data = torch.clamp(self.out_weight, min=1e-8)

        if batch_conversion_info is not None:  # batched packing
            x = utils.packed_to_padded(  # Convert x packed to zero padded
                x,
                batch_conversion_info["to_padded_mask"],
                batch_conversion_info["max_verts"],
                batch_conversion_info["num_graphs"],
            )

        # Diffuse to spread information towards farthest samples
        x = self._heat_diffusion(x, mass, evals, evecs, self.diffusion_in_time)

        # Select farthest samples
        x_far = x[farthest_sampling_mask, :].view(x.shape[0], -1, x.shape[-1])

        # Attention on farthest samples
        x_far = self.group_norm(x_far.transpose(1, 2)).transpose(1, 2)

        b = x_far.shape[0]
        head_dim = self.att_inner_ch // self._n_heads
        head_n = self._n_heads
        q = self.to_query(x_far).view(b, -1, head_n, head_dim).transpose(1, 2)
        k = self.to_key(x_far).view(b, -1, head_n, head_dim).transpose(1, 2)
        v = self.to_value(x_far).view(b, -1, head_n, head_dim).transpose(1, 2)

        x_far = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x_far = x_far.transpose(1, 2).reshape(b, -1, self.att_inner_ch)

        x_far = self.to_out(x_far)

        # Transfer on full-res points
        z = torch.zeros(
            [x.shape[0], x.shape[1], self._out_ch],
            device=x.device,
            dtype=x.dtype,
        ).view(-1, self._out_ch)
        z[farthest_sampling_mask.view(-1)] = x_far.view(-1, self._out_ch)
        x = z.view(x.shape[0], -1, self._out_ch)

        # Diffuse to spread attention output from farthest samples to neighbours
        x = self._heat_diffusion(x, mass, evals, evecs, self.diffusion_out_time)
        x = x * self.out_weight.unsqueeze(0)

        if batch_conversion_info is not None:  # batched packing
            # Convert x_diffuse from zero padded to packed
            x = utils.padded_to_packed(
                x, batch_conversion_info["to_packed_idx"]
            ).unsqueeze(0)

        return x


class DiffusionNetBlock(torch.nn.Module):
    """
    Inputs and outputs are defined at vertices
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None,
        emb_dims: int,
        geom_embed_dim: int,
        mlp_hidden_dims: List[int],
        attention_inner_channels: int | None,
        dropout: bool = True,
        norm_eps: float = 1e-6,
        norm_groups: int = 32,
    ):
        super(DiffusionNetBlock, self).__init__()

        # Specified dimensions
        self.in_ch = in_channels
        self.out_ch = in_channels if out_channels is None else out_channels
        # stack input, diffused, gradients, and geometry embeddings
        self.mlp_ch = 3 * self.in_ch
        self.mlp_hidden_dims = mlp_hidden_dims

        self.dropout = dropout

        # Spatial Diffusion block
        self.diffusion = LearnedTimeDiffusion(self.in_ch)

        # Spatial Gradients block
        self.gradient_features = SpatialGradientFeatures(self.in_ch)

        # Used defualt time embedding normalisation
        self.emb_proj = torch.nn.Linear(emb_dims, self.mlp_ch)
        self.norm = torch.nn.GroupNorm(
            num_groups=norm_groups,
            num_channels=self.mlp_ch,
            eps=norm_eps,
            affine=True,
        )

        # Attention block
        if attention_inner_channels is not None:
            self.attention = DiffusedFarthestAttention(
                in_channels + geom_embed_dim,
                out_channels,
                attention_inner_channels,
            )
        else:
            self.attention = None

        # MLP
        self.mlp = MiniMLP(
            [self.mlp_ch] + self.mlp_hidden_dims + [self.out_ch],
            dropout=self.dropout,
        )

        # Linear in skip connection to match channels
        if self.in_ch != self.out_ch:
            self.linear_skip = torch.nn.Linear(self.in_ch, self.out_ch)
        else:
            self.linear_skip = None

    def forward(
        self,
        x_in: torch.Tensor,
        emb: torch.Tensor,
        geom_emb: torch.Tensor,
        mass: torch.Tensor,
        evals: torch.Tensor,
        evecs: torch.Tensor,
        grad_x: torch.Tensor,
        grad_y: torch.Tensor,
        batch_conversion_info: dict | None = None,
        farthest_sampling_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b_conv_i = batch_conversion_info
        # Manage dimensions
        if x_in.shape[-1] != self.in_ch:
            raise ValueError(
                f"Tensor has wrong shape = {x_in.shape}. Last dim shape should "
                f"have number of channels = {self.in_ch}"
            )

        # Diffusion block
        x_diffuse = self.diffusion(x_in, mass, evals, evecs, b_conv_i)

        # Compute gradients
        x_grad = torch.stack(
            [torch.bmm(grad_x, x_diffuse), torch.bmm(grad_y, x_diffuse)], dim=-1
        )

        # Evaluate gradient features
        x_grad_features = self.gradient_features(x_grad)

        # Stack inputs to mlp
        feature_combined = torch.cat((x_in, x_diffuse, x_grad_features), dim=-1)

        emb = torch.nn.functional.relu(emb)  # in diffusers it uses swish
        emb = self.emb_proj(emb)
        if b_conv_i is not None:
            emb = emb.unsqueeze(1).expand(-1, b_conv_i["max_verts"], -1)
            emb = utils.padded_to_packed(emb, b_conv_i["to_packed_idx"])
            emb = emb.unsqueeze(0)

        feature_combined = feature_combined + emb
        feature_combined = self.norm(torch.permute(feature_combined, (0, 2, 1)))

        # Apply the mlp
        x_out = self.mlp(torch.permute(feature_combined, (0, 2, 1)))

        if self.attention is not None:
            x_cond = torch.cat((x_in, geom_emb), dim=-1)
            x_out = x_out + self.attention(
                x_cond, mass, evals, evecs, farthest_sampling_mask, b_conv_i
            )

        if self.linear_skip is not None:
            x_in = self.linear_skip(x_in)
        # Skip connection
        x_out = x_out + x_in

        return x_out


class MultiConvDiffusionNetBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None,
        emb_dims: int,
        geom_embed_dim: int,
        mlp_hidden_dims: List[int],
        attention_inner_channels: int | None,
        dropout: bool = True,
        norm_eps: float = 1e-6,
        norm_groups: int = 32,
        num_convs: int = 3,
    ):
        super(MultiConvDiffusionNetBlock, self).__init__()

        if attention_inner_channels is not None:
            self.attention = DiffusedFarthestAttention(
                in_channels + geom_embed_dim,
                out_channels,
                attention_inner_channels,
            )
        else:
            self.attention = None

        self.convs = []
        for i_conv in range(num_convs):  # also mid-layer here
            in_channels = in_channels if i_conv == 0 else out_channels
            conv = DiffusionNetBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                emb_dims=emb_dims,
                geom_embed_dim=geom_embed_dim,
                mlp_hidden_dims=mlp_hidden_dims,
                attention_inner_channels=None,
                dropout=dropout,
                norm_eps=norm_eps,
                norm_groups=norm_groups,
            )
            self.convs.append(conv)
            self.add_module("conv_" + str(i_conv), self.convs[-1])

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor,
        geom_emb: torch.Tensor,
        mass: torch.Tensor,
        evals: torch.Tensor,
        evecs: torch.Tensor,
        grad_x: torch.Tensor,
        grad_y: torch.Tensor,
        batch_conversion_info: dict | None = None,
        farthest_sampling_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.attention is not None:
            y = torch.cat((x, geom_emb), dim=-1)
            x_att = self.attention(
                y,
                mass,
                evals,
                evecs,
                farthest_sampling_mask,
                batch_conversion_info,
            )

        for conv in self.convs:
            x = conv(
                x,
                emb,
                geom_emb,
                mass,
                evals,
                evecs,
                grad_x,
                grad_y,
                batch_conversion_info,
                farthest_sampling_mask,
            )

        if self.attention is not None:
            x = x + x_att
        return x


@dataclass
class NetOutput(BaseOutput):
    """
    The output of DiffusionNet. Created for compatibility with diffusers library
    """

    sample: torch.FloatTensor


class DiffusionNet(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        io_mlp_channels: int = 128,
        attention_inner_channels: int | None = 64,
        blocks_depth: int = 4,
        last_activation=None,
        mlp_hidden_dims=None,
        dropout: bool = True,
        time_freq_shift: int = 0,
        time_flip_sin_to_cos: bool = True,
        k_eig: int = 128,
        n_hks: int = 0,
    ):
        """
        Construct a DiffusionNet.

        Parameters:
            with_gradient_features (bool):
                (default: True)
            with_gradient_rotations (bool): if True, use gradient also learn a
                rotation of each gradient. Set to True if your surface has
                consistently oriented normals, and False otherwise
                (default: True)
        Args:
            in_channels (int): input dimension
            out_channels (int): output dimension
            io_mlp_channels (int, optional): dimension of internal DiffusionNet
                blocks. Defaults to 128.
            attention_inner_channels (int, optional): number of channels to use
                for query, key, and value in self-attention.
                If None, no self-attention is used.
            blocks_depth (int, optional): number of DiffusionNet blocks in
                descending or ascending branch of 'U-Net'-like net.
                Defaults to 4.
            last_activation (_type_, optional): a function to apply to the final
                outputs of the network, such as torch.nn.functional.log_softmax.
                Defaults to None.
            mlp_hidden_dims (_type_, optional): a list of hidden layer sizes for
                MLPs. Defaults to None. If None, set to [C_width, C_width].
            dropout (bool, optional): dropout in internal MLPs. Defaults to True.
            time_freq_shift (int, optional): frequency shift in timesteps
                embedding creation for DDPM conditioning. Defaults to 0.
            time_flip_sin_to_cos (bool, optional): sine to cosine flipping in
                timesteps embedding creation for DDPM conditioning.
                Defaults to True.
            k_eig (int, optional): number of eigenvectors used for geometry
                conditioning of DDPM. Defaults to 128.
            n_hks (int, optional): number of heat kernel signatures used for
                geometry conditioninf of DDPM. Defaults to 0. If greater than 0
                hks are used instead of eigenvectors for geometry conditioning.
        """

        super(DiffusionNet, self).__init__()

        # Basic parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.io_mlp_channels = io_mlp_channels
        self.blocks_depth = blocks_depth
        self.has_attention = attention_inner_channels is not None

        # Outputs
        self.last_activation = last_activation

        # MLP options
        if mlp_hidden_dims == None:
            mlp_hidden_dims = [io_mlp_channels, io_mlp_channels]
        self.mlp_hidden_dims = mlp_hidden_dims
        self.dropout = dropout

        ## DDPM embeddings
        embed_dim = 4 * io_mlp_channels
        # time embedding
        timestep_input_dim = io_mlp_channels
        self.time_proj = Timesteps(
            timestep_input_dim, time_flip_sin_to_cos, time_freq_shift
        )
        self.time_embedding = MiniMLP(
            [timestep_input_dim, embed_dim, embed_dim],
            dropout=False,
            activation=torch.nn.SiLU,
            name="TimestepEmbedding",
        )

        # geometry embedding
        geom_input_dim = k_eig if n_hks <= 0 else n_hks
        geom_embed_dim = max(32, geom_input_dim // 4)

        self.geometry_embedding = MiniMLP(
            [geom_input_dim, 2 * geom_embed_dim, geom_embed_dim // 2],
            dropout=False,
            activation=torch.nn.SiLU,
            name="GeometryEmbedding",
        )
        shape_d_input_dim = k_eig
        shape_d_embed_dim = max(32, k_eig // 4)
        self.shape_embedding = MiniMLP(
            [shape_d_input_dim, 2 * shape_d_embed_dim, shape_d_embed_dim // 2],
            dropout=False,
            activation=torch.nn.SiLU,
            name="ShapeEmbedding",
        )
        geom_embed_dim = geom_embed_dim // 2 + shape_d_embed_dim // 2
        ## Set up the network

        # First and last affine layers
        self.first_lin = torch.nn.Linear(in_channels, io_mlp_channels)
        self.last_lin = torch.nn.Linear(io_mlp_channels, out_channels)

        self.blocks_down = []
        for i_block in range(self.blocks_depth + 1):  # also mid-layer here
            block = MultiConvDiffusionNetBlock(
                in_channels=io_mlp_channels,
                out_channels=io_mlp_channels,
                emb_dims=embed_dim,
                geom_embed_dim=geom_embed_dim,
                mlp_hidden_dims=mlp_hidden_dims,
                attention_inner_channels=attention_inner_channels,
                dropout=dropout,
            )
            self.blocks_down.append(block)
            self.add_module("block_down_" + str(i_block), self.blocks_down[-1])

        self.blocks_up = []
        for i_block in range(self.blocks_depth):
            block = MultiConvDiffusionNetBlock(
                in_channels=2 * io_mlp_channels,
                out_channels=io_mlp_channels,
                emb_dims=embed_dim,
                geom_embed_dim=geom_embed_dim,
                mlp_hidden_dims=mlp_hidden_dims,
                attention_inner_channels=attention_inner_channels,
                dropout=dropout,
            )
            self.blocks_up.append(block)
            self.add_module("block_up_" + str(i_block), self.blocks_up[-1])

    def forward(self, data, timesteps: torch.Tensor) -> NetOutput:
        """
        A forward pass on the DiffusionNet as DDPM.

        In the notation below, dimension are:
            - C is the input channel dimension (C_in on construction)
            - C_OUT is the output channel dimension (C_out on construction)
            - N is the number of vertices/points, which CAN be different for
                each forward pass
            - B is an OPTIONAL batch dimension
            - K_EIG is the number of eigenvalues used for spectral acceleration
        The supported data layouts are [N,C] or [B,N,C].

        Args:
            data (torch_geometric.data.Data): data containing the tensors:
                data.x: input features, dimension [N,C] or [B,N,C]
                data.massvec: mass vector, dimension [N] or [B,N]
                data.lapl: Laplace matrix, sparse tensor with dimension
                    [N,N] or [B,N,N]
                data.evals: Eigenvalues of Laplace matrix, dimension
                    [K_EIG] or [B,K_EIG]
                data.evecs: Eigenvectors of Laplace matrix, dimension
                    [N,K_EIG] or [B,N,K_EIG]
                data.grad_x: Half of gradient matrix, sparse real tensor with
                    dimension [N,N] or [B,N,N]
                data.grad_y: Half of gradient matrix, sparse real tensor with
                    dimension [N,N] or [B,N,N]
            timesteps (torch.Tensor): tensor containing the DDPM timestaps to
                use when denoising, dimension [1] or [B].

        Returns:
            NetOutput: class introduced for compatibility with diffusers.
                NetOutput.sample is a tensor with dimension [N,C_out]
                or [B,N,C_out]
        """
        x_in = data.x
        mass = data.massvec
        evals, evecs = data.evals, data.evecs
        grad_x, grad_y = data.grad_x, data.grad_y
        hks = data.hks if "hks" in data.keys() else None
        surf_area = data.surface_area if "surface_area" in data.keys() else None

        ## Check dimensions, and append batch dimension if not given
        if x_in.shape[-1] != self.in_channels:
            raise ValueError(
                f"DiffusionNet was constructed with C_in={self.in_channels}, "
                f"but x_in has last dim={x_in.shape[-1]}"
            )

        if len(x_in.shape) == 2:
            appended_batch_dim = True

            # add a batch dim to all inputs
            x_in = x_in.unsqueeze(0)
            if (
                "to_padded_mask" in data.keys()
                and mass.shape[0] == evals.shape[0]  # sampled point colours
            ):
                mass = mass.unsqueeze(-1)
                rebatch_mass = False
            else:
                mass = mass.unsqueeze(0)
                rebatch_mass = True
            if evals != None and "to_padded_mask" not in data.keys():
                evals = evals.unsqueeze(0)
            if evecs != None:
                evecs = evecs.unsqueeze(0)
            if hks != None:
                hks = hks.unsqueeze(0)
            if grad_x != None:  # Suppose that grad_y is also not None
                # Only unsqueeze when pytorch will support loading sparse tens
                if isinstance(grad_x, SparseTensor):
                    # This will be done in MeshCollater.custom_collate()
                    grad_x = grad_x.to_torch_sparse_coo_tensor()
                    grad_y = grad_y.to_torch_sparse_coo_tensor()
                else:
                    grad_x = utils.sparse_np_to_torch(grad_x)
                    grad_y = utils.sparse_np_to_torch(grad_y)
                grad_x = grad_x.unsqueeze(0)
                grad_y = grad_y.unsqueeze(0)

        elif len(x_in.shape) == 3:
            appended_batch_dim = False
            grad_x = utils.stack_list_padded_sparse(
                grad_x, new_side_size=x_in.shape[1]
            )
            grad_y = utils.stack_list_padded_sparse(
                grad_y, new_side_size=x_in.shape[1]
            )

        else:
            raise ValueError(
                "x_in should be tensor with shape [N,C], [BN, C], or [B,N,C]"
            )

        grad_x, grad_y = grad_x.to(x_in.device), grad_y.to(x_in.device)

        # DDPM time and class conditioning
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor(
                [timesteps], dtype=torch.long, device=x_in.device
            )
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(x_in.device)

        # broadcast to batch dimension in a way that's compatible with
        # ONNX/Core ML
        timesteps = timesteps * torch.ones(
            x_in.shape[0], dtype=timesteps.dtype, device=timesteps.device
        )

        geom_features = hks if hks is not None else evecs

        # Evals change with scale, follow first steps of cShapeDNA to get them
        # on a straight line and remove scale dependency.
        arange = torch.arange(0, evals.shape[1], device=evals.device).float()
        shape_desc = evals * surf_area[:, None] - 4 * torch.pi * arange

        time_pr = self.time_proj(timesteps)
        temb = self.time_embedding(time_pr)
        gemb = self.geometry_embedding(geom_features)
        semb = self.shape_embedding(shape_desc)

        # Apply the first linear layer
        x = self.first_lin(x_in)

        # Convert evecs from packed to padded
        if "to_padded_mask" in data.keys():
            evecs = utils.packed_to_padded(
                evecs, data.to_padded_mask, data.max_verts, data.num_graphs
            )
            if rebatch_mass:
                mass = utils.packed_to_padded(
                    mass.unsqueeze(-1),
                    data.to_padded_mask,
                    data.max_verts,
                    data.num_graphs,
                ).squeeze(-1)
            semb = semb.unsqueeze(1).expand(-1, data.max_verts, -1)
            semb = utils.padded_to_packed(semb, data.to_packed_idx).unsqueeze(0)
            # collect batch conversion info in a dict
            bconv = {
                "max_verts": data.max_verts,
                "num_graphs": data.num_graphs,
                "to_padded_mask": data.to_padded_mask,
                "to_packed_idx": data.to_packed_idx,
            }
        else:
            semb = semb.unsqueeze(1).expand(-1, x_in.shape[1], -1)
            bconv = None

        gemb = torch.cat([gemb, semb], dim=-1)
        fmask = data.farthest_sampling_mask if self.has_attention else None

        down_xs = [x]
        for i, b in enumerate(self.blocks_down):
            x = b(
                x, temb, gemb, mass, evals, evecs, grad_x, grad_y, bconv, fmask
            )
            if i < len(self.blocks_down) - 1:
                down_xs.append(x)

        for i, b in enumerate(self.blocks_up):
            x = torch.cat([x, down_xs[-i - 1]], dim=-1)
            x = b(
                x, temb, gemb, mass, evals, evecs, grad_x, grad_y, bconv, fmask
            )

        # Apply the last linear layer
        x_out = self.last_lin(x)

        # Apply last nonlinearity if specified
        if self.last_activation != None:
            x_out = self.last_activation(x_out)

        # Remove batch dim if we added it
        if appended_batch_dim:
            x_out = x_out.squeeze(0)

        return NetOutput(sample=x_out)


def get_backward_hook(module_name: str):

    class BackwardHook:
        name: str

        def __init__(self, name):
            self.name = name

        def __call__(self, module, grad_input, grad_output):
            for i, g_in in enumerate(grad_input):
                if g_in is not None and torch.any(torch.isnan(g_in)):
                    raise RuntimeError(f"{module_name}'s {i}th in grad is nan")
            for i, g_out in enumerate(grad_output):
                if g_out is not None and torch.any(torch.isnan(g_out)):
                    raise RuntimeError(f"{module_name}'s {i}th out grad is nan")

    return BackwardHook(module_name)


if __name__ == "__main__":
    import transforms
    import torch_geometric

    pre_transform = torch_geometric.transforms.Compose(
        [
            torch_geometric.transforms.NormalizeScale(),
            transforms.LaplacianEigendecomposition(
                mix_lapl_w=0.05,
                k_eig=128,
                timeout_seconds=300,
                store_lapl_and_massvec=True,
            ),
            transforms.TangentGradients(
                as_cloud=False, save_edges=True, save_normals=True
            ),
            transforms.DropTrimesh(),
        ]
    )

    transform = transforms.VertexColoursFromBaseTexture()

    train_set = transforms.AmazonBerkeleyObjectsDataset(
        root="/data/AmazonBerkeleyObjects/original",
        dataset_type="train",
        pre_transform=pre_transform,
        transform=transform,
        filter_only_files_with="base_color_tex",
    )

    train_loader = transforms.MeshLoader(train_set, 1, shuffle=True)
    batch = next(iter(train_loader))

    diffnet = DiffusionNet(
        in_channels=3,
        out_channels=3,
        io_mlp_channels=256,
        blocks_depth=5,
        last_activation=None,
        mlp_hidden_dims=None,
        dropout=True,
        with_gradient_features=True,
        with_gradient_rotations=True,
        space_diffusion_method="spectral",
        time_freq_shift=0,
        time_flip_sin_to_cos=True,
        k_eig=128,
    )

    x = diffnet(batch, timesteps=torch.LongTensor([50])).sample
    print("done")
