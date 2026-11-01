import math
from typing import Optional, Union
import torch
import torch.nn.functional as F
from torch import nn



def conv2d_block(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        activation: nn.Module = None,
        norm_type: str = None,
        num_groups_for_gn: int = 1,
        bias: bool = True
) -> nn.Sequential:
    """
    Overview:
        Create a 2-dimensional convolution layer with activation and normalization.
    Arguments:
        - in_channels (:obj:`int`): Number of channels in the input tensor.
        - out_channels (:obj:`int`): Number of channels in the output tensor.
        - kernel_size (:obj:`int`): Size of the convolving kernel.
        - stride (:obj:`int`, optional): Stride of the convolution. Default is 1.
        - padding (:obj:`int`, optional): Zero-padding added to both sides of the input. Default is 0.
        - dilation (:obj:`int`): Spacing between kernel elements.
        - groups (:obj:`int`, optional): Number of blocked connections from input channels to output channels. \
            Default is 1.
        - activation (:obj:`nn.Module`): the optional activation function.
        - norm_type (:obj:`str`): If provided, GroupNorm is applied.
        - num_groups_for_gn (:obj:`int`): Number of groups for GroupNorm.
        - bias (:obj:`bool`): whether to add a learnable bias to the nn.Conv2d. Default is True.
    Returns:
        - block (:obj:`nn.Sequential`): A sequential list containing the torch layers of the 2-dimensional \
            convolution layer.

    .. note::
        Conv2d (https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d)
    """
    block = []
    block.append(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
    )
    if norm_type is not None:
        block.append(nn.GroupNorm(num_groups_for_gn, out_channels))

    if activation is not None:
        block.append(activation)
    return nn.Sequential(*block)

class ResBlock(nn.Module):
    """
    Overview:
        Residual Block with 2D convolution layers, including 3 types:
            basic block:
                input channel: C
                x -> 3*3*C -> norm -> act -> 3*3*C -> norm -> act -> out
                \__________________________________________/+
            bottleneck block:
                x -> 1*1*(1/4*C) -> norm -> act -> 3*3*(1/4*C) -> norm -> act -> 1*1*C -> norm -> act -> out
                \_____________________________________________________________________________/+
            downsample block: used in EfficientZero
                input channel: C
                x -> 3*3*C -> norm -> act -> 3*3*C -> norm -> act -> out
                \__________________ 3*3*C ____________________/+

    .. note::
        You can refer to `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`_ for more \
        details.

    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(
        self,
        in_channels: int,
        activation: nn.Module = nn.ReLU(),
        norm_type: str = 'GN',
        res_type: str = 'basic',
        bias: bool = True,
        out_channels: Union[int, None] = None,
    ) -> None:
        """
        Overview:
            Init the 2D convolution residual block.
        Arguments:
            - in_channels (:obj:`int`): Number of channels in the input tensor.
            - activation (:obj:`nn.Module`): The optional activation function.
            - norm_type (:obj:`str`): If provided, GroupNorm is used.
            - res_type (:obj:`str`): Type of residual block, supports ['basic', 'bottleneck', 'downsample']
            - bias (:obj:`bool`): Whether to add a learnable bias to the conv2d_block. default set to True.
            - out_channels (:obj:`int`): Number of channels in the output tensor, default set to None, \
                which means out_channels = in_channels.
        """
        super(ResBlock, self).__init__()
        self.act = activation
        assert res_type in ['basic', 'bottleneck',
                            'downsample'], 'residual type only support basic and bottleneck, not:{}'.format(res_type)
        self.res_type = res_type
        if out_channels is None:
            out_channels = in_channels
        if self.res_type == 'basic':
            self.conv1 = conv2d_block(
                in_channels, out_channels, 3, 1, 1, activation=self.act, norm_type=norm_type, bias=bias
            )
            self.conv2 = conv2d_block(
                out_channels, out_channels, 3, 1, 1, activation=None, norm_type=norm_type, bias=bias
            )
        elif self.res_type == 'bottleneck':
            self.conv1 = conv2d_block(
                in_channels, out_channels, 1, 1, 0, activation=self.act, norm_type=norm_type, bias=bias
            )
            self.conv2 = conv2d_block(
                out_channels, out_channels, 3, 1, 1, activation=self.act, norm_type=norm_type, bias=bias
            )
            self.conv3 = conv2d_block(
                out_channels, out_channels, 1, 1, 0, activation=None, norm_type=norm_type, bias=bias
            )
        elif self.res_type == 'downsample':
            self.conv1 = conv2d_block(
                in_channels, out_channels, 3, 2, 1, activation=self.act, norm_type=norm_type, bias=bias
            )
            self.conv2 = conv2d_block(
                out_channels, out_channels, 3, 1, 1, activation=None, norm_type=norm_type, bias=bias
            )
            self.conv3 = conv2d_block(in_channels, out_channels, 3, 2, 1, activation=None, norm_type=None, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Return the redisual block output.
        Arguments:
            - x (:obj:`torch.Tensor`): The input tensor.
        Returns:
            - x (:obj:`torch.Tensor`): The resblock output tensor.
        """
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.res_type == 'bottleneck':
            x = self.conv3(x)
        elif self.res_type == 'downsample':
            identity = self.conv3(identity)
        x = self.act(x + identity)
        return x

class Conv2dSame(nn.Conv2d):
    """
    Overview:
         Conv2dSame Network for dreamerv3.
    Interfaces:
        ``__init__``, ``forward``
    """

    def calc_same_pad(self, i, k, s, d):
        """
        Overview:
            Calculate the same padding size.
        Arguments:
            - i (:obj:`int`): Input size.
            - k (:obj:`int`): Kernel size.
            - s (:obj:`int`): Stride size.
            - d (:obj:`int`): Dilation size.
        """
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x):
        """
        Overview:
            compute the forward of Conv2dSame.
        Arguments:
            - x (:obj:`torch.Tensor`): Input tensor.
        """
        ih, iw = x.size()[-2:]
        pad_h = self.calc_same_pad(i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])

        ret = F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return ret


class DreamerLayerNorm(nn.Module):
    """
    Overview:
         DreamerLayerNorm Network for dreamerv3.
    Interfaces:
        ``__init__``, ``forward``
    """

    def __init__(self, ch, eps=1e-03):
        """
        Overview:
            Init the DreamerLayerNorm class.
        Arguments:
            - ch (:obj:`int`): Input channel.
            - eps (:obj:`float`): Epsilon.
        """

        super(DreamerLayerNorm, self).__init__()
        self.norm = nn.LayerNorm(ch, eps=eps)

    def forward(self, x):
        """
        Overview:
            compute the forward of DreamerLayerNorm.
        Arguments:
            - x (:obj:`torch.Tensor`): Input tensor.
        """

        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


class ConvEncoder(nn.Module):
    """
    Overview:
        The Convolution Encoder is used to encode 2-dim image observations.
    Interfaces:
        ``__init__``, ``forward``.
    """

    def __init__(
            self,
            obs_shape,
            hidden_size_list= [32, 64, 64, 128],
            activation: Optional[nn.Module] = nn.ReLU(),
            kernel_size= [8, 4, 3],
            stride= [4, 2, 1],
            padding: Optional[list] = None,
            layer_norm: Optional[bool] = False,
            norm_type: Optional[str] = None
    ) -> None:
        """
        Overview:
            Initialize the ``Convolution Encoder`` according to the provided arguments.
        Arguments:
            - obs_shape (:obj:`SequenceType`): Sequence of ``in_channel``, plus one or more ``input size``.
            - hidden_size_list (:obj:`SequenceType`): Sequence of ``hidden_size`` of subsequent conv layers \
                and the final dense layer.
            - activation (:obj:`nn.Module`): Type of activation to use in the conv ``layers`` and ``ResBlock``. \
                Default is ``nn.ReLU()``.
            - kernel_size (:obj:`SequenceType`): Sequence of ``kernel_size`` of subsequent conv layers.
            - stride (:obj:`SequenceType`): Sequence of ``stride`` of subsequent conv layers.
            - padding (:obj:`SequenceType`): Padding added to all four sides of the input for each conv layer. \
                See ``nn.Conv2d`` for more details. Default is ``None``.
            - layer_norm (:obj:`bool`): Whether to use ``DreamerLayerNorm``, which is kind of special trick \
                proposed in DreamerV3.
            - norm_type (:obj:`str`): Type of normalization to use. See ``ding.torch_utils.network.ResBlock`` \
                for more details. Default is ``None``.
        """
        super(ConvEncoder, self).__init__()
        self.obs_shape = obs_shape
        self.act = activation
        self.hidden_size_list = hidden_size_list
        if padding is None:
            padding = [0 for _ in range(len(kernel_size))]

        layers = []
        input_size = obs_shape[0]  # in_channel
        for i in range(len(kernel_size)):
            if layer_norm:
                layers.append(
                    Conv2dSame(
                        in_channels=input_size,
                        out_channels=hidden_size_list[i],
                        kernel_size=(kernel_size[i], kernel_size[i]),
                        stride=(2, 2),
                        bias=False,
                    )
                )
                layers.append(DreamerLayerNorm(hidden_size_list[i]))
                layers.append(self.act)
            else:
                layers.append(nn.Conv2d(input_size, hidden_size_list[i], kernel_size[i], stride[i], padding[i]))
                layers.append(self.act)
            input_size = hidden_size_list[i]
        if len(self.hidden_size_list) >= len(kernel_size) + 2:
            assert self.hidden_size_list[len(kernel_size) - 1] == self.hidden_size_list[
                len(kernel_size)], "Please indicate the same hidden size between conv and res block"
        assert len(
            set(hidden_size_list[len(kernel_size):-1])
        ) <= 1, "Please indicate the same hidden size for res block parts"
        for i in range(len(kernel_size), len(self.hidden_size_list) - 1):
            layers.append(ResBlock(self.hidden_size_list[i - 1], activation=self.act, norm_type=norm_type))
        layers.append(nn.Flatten())
        self.main = nn.Sequential(*layers)

        flatten_size = self._get_flatten_size()
        self.output_size = hidden_size_list[-1]  # outside to use
        self.mid = nn.Linear(flatten_size, hidden_size_list[-1])

    def _get_flatten_size(self) -> int:
        """
        Overview:
            Get the encoding size after ``self.main`` to get the number of ``in-features`` to feed to ``nn.Linear``.
        Returns:
            - outputs (:obj:`torch.Tensor`): Size ``int`` Tensor representing the number of ``in-features``.
        Shapes:
            - outputs: :math:`(1,)`.
        Examples:
            >>> conv = ConvEncoder(
            >>>    obs_shape=(4, 84, 84),
            >>>    hidden_size_list=[32, 64, 64, 128],
            >>>    activation=nn.ReLU(),
            >>>    kernel_size=[8, 4, 3],
            >>>    stride=[4, 2, 1],
            >>>    padding=None,
            >>>    layer_norm=False,
            >>>    norm_type=None
            >>> )
            >>> flatten_size = conv._get_flatten_size()
        """
        test_data = torch.randn(1, *self.obs_shape)
        with torch.no_grad():
            output = self.main(test_data)
        return output.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Return output 1D embedding tensor of the env's 2D image observation.
        Arguments:
            - x (:obj:`torch.Tensor`): Raw 2D observation of the environment.
        Returns:
            - outputs (:obj:`torch.Tensor`): Output embedding tensor.
        Shapes:
            - x : :math:`(B, C, H, W)`, where ``B`` is batch size, ``C`` is channel, ``H`` is height, ``W`` is width.
            - outputs: :math:`(B, N)`, where ``N = hidden_size_list[-1]`` .
        Examples:
            >>> conv = ConvEncoder(
            >>>    obs_shape=(4, 84, 84),
            >>>    hidden_size_list=[32, 64, 64, 128],
            >>>    activation=nn.ReLU(),
            >>>    kernel_size=[8, 4, 3],
            >>>    stride=[4, 2, 1],
            >>>    padding=None,
            >>>    layer_norm=False,
            >>>    norm_type=None
            >>> )
            >>> x = torch.randn(1, 4, 84, 84)
            >>> output = conv(x)
        """
        x = self.main(x)
        x = self.mid(x)
        return x