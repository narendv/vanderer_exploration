import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from typing import List, Dict, Optional, Tuple, Callable
from efficientnet_pytorch import EfficientNet
from vint_train.models.vint.self_attention import PositionalEncoding, get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid
from expnav.models.query_selection import GoalGuidedHardQuerySelector
from diffusers.models import AutoencoderKL
from einops import rearrange, repeat


class NoMaD_ViNT(nn.Module):
    def __init__(
        self,
        context_size: int = 5,
        obs_encoder: Optional[str] = "efficientnet-b0",
        obs_encoding_size: Optional[int] = 256,
        mha_num_attention_heads: Optional[int] = 2,
        mha_num_attention_layers: Optional[int] = 2,
        mha_ff_dim_factor: Optional[int] = 4,
        use_cache_encodings: Optional[bool] = True,
    ) -> None:
        """
        NoMaD ViNT Encoder class
        """
        super().__init__()
        self.obs_encoding_size = obs_encoding_size
        self.goal_encoding_size = obs_encoding_size
        self.context_size = context_size
        self.use_cache_encodings = use_cache_encodings

        # Initialize the observation encoder
        if obs_encoder.split("-")[0] == "efficientnet":
            self.obs_encoder = EfficientNet.from_name(obs_encoder, in_channels=3) # context
            self.obs_encoder = replace_bn_with_gn(self.obs_encoder)
            self.num_obs_features = self.obs_encoder._fc.in_features
            self.obs_encoder_type = "efficientnet"

            # Initialize the goal encoder
            self.goal_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=6) # obs+goal
            self.goal_encoder = replace_bn_with_gn(self.goal_encoder)
            self.num_goal_features = self.goal_encoder._fc.in_features
        elif obs_encoder == "vae":
            self.obs_encoder = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema")
            self.obs_encoder.requires_grad_(False) # freeze VAE weights
            self.obs_encoder.eval() # set to eval mode
            self.obs_encoder_type = "vae"
            self.num_obs_features = self.obs_encoder.config.latent_channels

            # Initialize the goal encoder
            self.goal_encoder = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema")
            self.goal_encoder.requires_grad_(False) # freeze VAE weights
            self.goal_encoder.eval() # set to eval mode
            self.num_goal_features = self.goal_encoder.config.latent_channels

        # Initialize compression layers if necessary
        if self.num_obs_features != self.obs_encoding_size:
            if self.obs_encoder_type == "vae":
                self.compress_obs_enc = nn.Conv2d(self.num_obs_features, self.obs_encoding_size, kernel_size=1, stride=1)
            else:
                self.compress_obs_enc = nn.Linear(self.num_obs_features, self.obs_encoding_size)
        else:
            self.compress_obs_enc = nn.Identity()
        
        if self.num_goal_features != self.goal_encoding_size:
            self.compress_goal_enc = nn.Linear(self.num_goal_features, self.goal_encoding_size)
        else:
            self.compress_goal_enc = nn.Identity()

        # Initialize positional encoding and self-attention layers
        if self.obs_encoder_type == "vae":
            # self.spatial_encoding = nn.Parameter(torch.randn(1, self.obs_encoding_size, 28, 28))
            # self.temporal_encoding = nn.Parameter(torch.randn(1, self.obs_encoding_size, self.context_size + 2, 1, 1))
            spatial_encoding = get_2d_sincos_pos_embed(self.obs_encoding_size, 14, cls_token=False, extra_tokens=0).unsqueeze(0).unsqueeze(0).float()
            temporal_encoding = get_1d_sincos_pos_embed_from_grid(self.obs_encoding_size, torch.arange(self.context_size + 1).float()).unsqueeze(0).unsqueeze(-2).float()
            self.register_buffer('spatial_encoding', spatial_encoding, persistent=False)
            self.register_buffer('temporal_encoding', temporal_encoding, persistent=False)
            self.goal_embedding = nn.Parameter(torch.randn(1, 1, self.obs_encoding_size)).unsqueeze(0)
        self.positional_encoding = PositionalEncoding(self.obs_encoding_size, max_seq_len=self.context_size + 2)
        self.sa_layer = nn.TransformerEncoderLayer(
            d_model=self.obs_encoding_size, 
            nhead=mha_num_attention_heads, 
            dim_feedforward=mha_ff_dim_factor*self.obs_encoding_size, 
            activation="gelu", 
            batch_first=True, 
            norm_first=True
        )
        self.sa_encoder = nn.TransformerEncoder(self.sa_layer, num_layers=mha_num_attention_layers)
        if self.obs_encoder_type == "vae":
            # self.output_proj = nn.Sequential(
            #     nn.Conv2d(self.obs_encoding_size, self.obs_encoding_size, kernel_size=3, stride=1, padding=1),
            #     nn.BatchNorm2d(self.obs_encoding_size),
            #     nn.SiLU(),
            #     nn.MaxPool2d(kernel_size=2, stride=2),
            #     nn.Conv2d(self.obs_encoding_size, self.obs_encoding_size, kernel_size=3, stride=1, padding=1),
            #     nn.BatchNorm2d(self.obs_encoding_size),
            #     nn.SiLU(),
            #     nn.MaxPool2d(kernel_size=2, stride=2),
            #     nn.AdaptiveAvgPool2d(1)
            # )
            red_factor = 2
            self.output_proj = nn.AvgPool2d(kernel_size=red_factor, stride=red_factor)
            self.output_size = (28 // red_factor, 28 // red_factor)
            self.query_guided_pool = GoalGuidedHardQuerySelector(
                                        model_dim=self.obs_encoding_size,
                                        num_query=196,
                                        out_dim=self.obs_encoding_size,
                                    )

        # Definition of the goal mask (convention: 0 = no mask, 1 = mask)
        goal_mask = torch.zeros((1, self.context_size + 2), dtype=torch.bool)
        goal_mask[:, -1] = True # Mask out the goal 
        no_mask = torch.zeros((1, self.context_size + 2), dtype=torch.bool) 
        all_masks = torch.cat([no_mask, goal_mask], dim=0)
        avg_pool_mask = torch.cat([1 - no_mask.float(), (1 - goal_mask.float()) * ((self.context_size + 2)/(self.context_size + 1))], dim=0)
        self.register_buffer('goal_mask', goal_mask, persistent=False)
        self.register_buffer('no_mask', no_mask, persistent=False)
        self.register_buffer('all_masks', all_masks, persistent=False)
        self.register_buffer('avg_pool_mask', avg_pool_mask, persistent=False)

    def forward(self, obs_img: torch.tensor, goal_img: torch.tensor, input_goal_mask: torch.tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.obs_encoder_type == "efficientnet":
            return self._forward_effnet(obs_img, goal_img, input_goal_mask)
        else:
            return self._forward_vae(obs_img, goal_img, input_goal_mask)

    def _forward_effnet(self, obs_img: torch.tensor, goal_img: torch.tensor, input_goal_mask: torch.tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:

        device = obs_img.device

        # Initialize the goal encoding
        goal_encoding = torch.zeros((obs_img.size()[0], 1, self.goal_encoding_size)).to(device)

        # Get the goal encoding 
        obsgoal_img = torch.cat([obs_img[:, 3*self.context_size:, :, :], goal_img], dim=1) # concatenate the obs image/context and goal image --> non image goal?
        obsgoal_encoding = self.goal_encoder.extract_features(obsgoal_img) # get encoding of this img 
        obsgoal_encoding = self.goal_encoder._avg_pooling(obsgoal_encoding) # avg pooling
        
        if self.goal_encoder._global_params.include_top:
            obsgoal_encoding = obsgoal_encoding.flatten(start_dim=1)
            obsgoal_encoding = self.goal_encoder._dropout(obsgoal_encoding)
        obsgoal_encoding = self.compress_goal_enc(obsgoal_encoding)

        if len(obsgoal_encoding.shape) == 2:
            obsgoal_encoding = obsgoal_encoding.unsqueeze(1)
        assert obsgoal_encoding.shape[2] == self.goal_encoding_size
        goal_encoding = obsgoal_encoding
        
        # Get the observation encoding
        obs_img = torch.split(obs_img, 3, dim=1)
        obs_img = torch.concat(obs_img, dim=0)


        obs_encoding = self.obs_encoder.extract_features(obs_img)
        obs_encoding = self.obs_encoder._avg_pooling(obs_encoding)
        if self.obs_encoder._global_params.include_top:
            obs_encoding = obs_encoding.flatten(start_dim=1)
            obs_encoding = self.obs_encoder._dropout(obs_encoding)
        obs_encoding = self.compress_obs_enc(obs_encoding)
        obs_encoding = obs_encoding.unsqueeze(1)
        obs_encoding = obs_encoding.reshape((self.context_size+1, -1, self.obs_encoding_size))
        obs_encoding = torch.transpose(obs_encoding, 0, 1)
            
        obs_encoding = torch.cat((obs_encoding, goal_encoding), dim=1)
        
        # If a goal mask is provided, mask some of the goal tokens
        if input_goal_mask is not None:
            goal_mask = input_goal_mask.to(device)
            no_goal_mask = goal_mask.long()
            src_key_padding_mask = torch.index_select(self.all_masks, 0, no_goal_mask)
        else:
            src_key_padding_mask = None
        
        # Apply positional encoding 
        if self.positional_encoding:
            obs_encoding = self.positional_encoding(obs_encoding)
        
        obs_encoding_tokens = self.sa_encoder(obs_encoding, src_key_padding_mask=src_key_padding_mask)
        if src_key_padding_mask is not None:
            avg_mask = torch.index_select(self.avg_pool_mask, 0, no_goal_mask).unsqueeze(-1)
            obs_encoding_tokens = obs_encoding_tokens * avg_mask
        pooled_obs_encoding_tokens = torch.mean(obs_encoding_tokens, dim=1)

        return pooled_obs_encoding_tokens   #, obs_encoding_tokens
    
    # def forward_vae(self, obs_img: torch.tensor, goal_img: torch.tensor, input_goal_mask: torch.tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
    #     device = obs_img.device
    #     obs_img = obs_img.to(device)
    #     goal_img = goal_img.to(device)
    #     if not self.use_cache_encodings:
    #         # Unbind to get context images and goal image
    #         obs_img = torch.split(obs_img, 3, dim=1)
    #         obs_img = torch.concat(obs_img, dim=0)
    #         obs_n_goal = torch.cat([obs_img, goal_img], dim=0)
            
    #         obs_n_goal = self.obs_encoder.encode(obs_n_goal).latent_dist.sample()
    #         # obs_encoding = obs_encoding.flatten(start_dim=1)
    #         obs_n_goal = self.compress_obs_enc(obs_n_goal)
    #         obs_n_goal = self.output_proj(obs_n_goal)
    #         obs_n_goal = rearrange(obs_n_goal, '(b l) d h w -> b l d h w', l=self.context_size+2)
    #         obs_encoding = obs_n_goal[:,:-1,...]
    #         goal_encoding = obs_n_goal[:,-1,...]
    #     else:
    #         obs_encoding = rearrange(obs_img, 'b l d h w -> (b l) d h w')
    #         obs_encoding = self.compress_obs_enc(obs_encoding)
    #         obs_encoding = self.output_proj(obs_encoding)
    #         obs_encoding = rearrange(obs_encoding, '(b l) d h w -> b l d h w', l=self.context_size+1)

    #         goal_encoding = self.compress_obs_enc(goal_img)
    #         goal_encoding = self.output_proj(goal_encoding)
            
    #     obs_encoding = rearrange(obs_encoding, 'b l d h w -> b l (h w) d')
    #     goal_encoding = rearrange(goal_encoding, 'b d h w -> b (h w) d').unsqueeze(1)
    #     # If a goal mask is provided, mask some of the goal tokens
    #     if input_goal_mask is not None:
    #         goal_mask = input_goal_mask.to(device)
    #         no_goal_mask = goal_mask.long()
    #         src_key_padding_mask = torch.index_select(self.all_masks.to(device), 0, no_goal_mask)
    #         src_key_padding_mask = src_key_padding_mask.repeat_interleave(self.output_size[0]*self.output_size[1], dim=1)
    #     else:
    #         src_key_padding_mask = None
        
    #     # Apply positional encoding 
    #     if self.positional_encoding:
    #         obs_encoding += self.spatial_encoding.to(device)
    #         obs_encoding += self.temporal_encoding.to(device)
    #         goal_encoding += self.goal_embedding.to(device)
    #     obs_encoding = torch.cat((obs_encoding, goal_encoding), dim=1)
        
    #     obs_encoding = rearrange(obs_encoding, 'b l (h w) d -> b (l h w) d', h=self.output_size[0], w=self.output_size[1])
        
    #     obs_encoding_tokens = self.sa_encoder(obs_encoding, src_key_padding_mask=src_key_padding_mask)
    #     obs_encoding_tokens = rearrange(obs_encoding_tokens, 'b (l h w) d -> b l d h w', l=self.context_size + 2, h=self.output_size[0], w=self.output_size[1])

    #     if src_key_padding_mask is not None:
    #         avg_mask = torch.index_select(self.avg_pool_mask.to(device), 0, no_goal_mask)
    #         avg_mask = avg_mask[:, :, None, None, None].repeat(1, 1, 256, 1, 1)
    #         obs_encoding_tokens = obs_encoding_tokens * avg_mask
        
    #     # pooled_obs_encoding_tokens = torch.mean(obs_encoding_tokens, dim=[1,3,4])
    #     pooled_obs_encoding_tokens = self.query_guided_pool(obs_encoding_tokens[:, :-1, ...], obs_encoding_tokens[:, -1, ...])

    #     return pooled_obs_encoding_tokens   #, obs_encoding_tokens
    
    def _forward_vae(self, obs_img: torch.tensor, goal_img: torch.tensor, input_goal_mask: torch.tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        device = obs_img.device
        obs_img = obs_img.to(device)
        goal_img = goal_img.to(device)
        if not self.use_cache_encodings:
            # Unbind to get context images and goal image
            obs_img = torch.split(obs_img, 3, dim=1)
            obs_img = torch.concat(obs_img, dim=0)
            obs_n_goal = torch.cat([obs_img, goal_img], dim=0)
            
            obs_n_goal = self.obs_encoder.encode(obs_n_goal).latent_dist.sample()
            # obs_encoding = obs_encoding.flatten(start_dim=1)
            obs_n_goal = self.compress_obs_enc(obs_n_goal)
            obs_n_goal = self.output_proj(obs_n_goal)
            obs_n_goal = rearrange(obs_n_goal, '(b l) d h w -> b l d h w', l=self.context_size+2)
            obs_encoding = obs_n_goal[:,:-1,...]
            goal_encoding = obs_n_goal[:,-1,...]
        else:
            obs_encoding = rearrange(obs_img, 'b l d h w -> (b l) d h w')
            obs_encoding = self.compress_obs_enc(obs_encoding)
            obs_encoding = self.output_proj(obs_encoding)
            obs_encoding = rearrange(obs_encoding, '(b l) d h w -> b l d h w', l=self.context_size+1)
            
        obs_encoding = rearrange(obs_encoding, 'b l d h w -> b l (h w) d')
        
        # Apply positional encoding 
        if self.positional_encoding:
            obs_encoding = obs_encoding + self.spatial_encoding
            obs_encoding = obs_encoding + self.temporal_encoding
        
        obs_encoding = rearrange(obs_encoding, 'b l (h w) d -> b (l h w) d', h=self.output_size[0], w=self.output_size[1])
        
        obs_encoding_tokens = self.sa_encoder(obs_encoding)
        obs_encoding_tokens = rearrange(obs_encoding_tokens, 'b (l h w) d -> b l d h w', l=self.context_size + 1, h=self.output_size[0], w=self.output_size[1])
        
        pooled_obs_encoding_tokens = torch.mean(obs_encoding_tokens, dim=[1,3,4])

        return pooled_obs_encoding_tokens   #, obs_encoding_tokens

        



# Utils for Group Norm
def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module


def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module


# if __name__ == "__main__":
#     model = NoMaD_ViNT(context_size=3, obs_encoder="vae").to("cuda")
#     from torchinfo import summary
#     input_obs = torch.randn((2, 3*4, 128, 128)).to("cuda")
#     input_goal = torch.randn((2, 3, 128, 128)).to("cuda")
#     out, out_tokens = model(input_obs, input_goal)
#     print(out.shape, out_tokens.shape)
#     summary(model, input_data=(input_obs, input_goal))




    