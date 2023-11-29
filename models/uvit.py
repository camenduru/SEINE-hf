import torch
import torch.nn as nn
import math
import timm
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import PatchEmbed, Mlp
# assert timm.__version__ == "0.3.2"  # version checks
import einops
import torch.utils.checkpoint

# the xformers lib allows less memory, faster training and inference
try:
    import xformers
    import xformers.ops
except:
    XFORMERS_IS_AVAILBLE = False
    # print('xformers disabled')


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def patchify(imgs, patch_size):
    x = einops.rearrange(imgs, 'B C (h p1) (w p2) -> B (h w) (p1 p2 C)', p1=patch_size, p2=patch_size)
    return x


def unpatchify(x, channels=3):
    patch_size = int((x.shape[2] // channels) ** 0.5)
    h = w = int(x.shape[1] ** .5)
    assert h * w == x.shape[1] and patch_size ** 2 * channels == x.shape[2]
    x = einops.rearrange(x, 'B (h w) (p1 p2 C) -> B C (h p1) (w p2)', h=h, p1=patch_size, p2=patch_size)
    return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, L, C = x.shape

        qkv = self.qkv(x)
        if XFORMERS_IS_AVAILBLE:  # the xformers lib allows less memory, faster training and inference
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B L H D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B L H D
            x = xformers.ops.memory_efficient_attention(q, k, v)
            x = einops.rearrange(x, 'B L H D -> B L (H D)', H=self.num_heads)
        else:
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, L, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip=False, use_checkpoint=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.use_checkpoint = use_checkpoint

    def forward(self, x, skip=None):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, skip)
        else:
            return self._forward(x, skip)

    def _forward(self, x, skip=None):
        if self.skip_linear is not None:
            # print('x shape', x.shape)
            # print('skip shape', skip.shape)
            # exit()
            x = self.skip_linear(torch.cat([x, skip], dim=-1))
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class UViT(nn.Module):
    def __init__(self, input_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, mlp_time_embed=False, num_classes=-1,
                 use_checkpoint=False, conv=True, skip=True, num_frames=16, class_guided=False, use_lora=False):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_classes = num_classes
        self.in_chans = in_chans

        self.patch_embed = PatchEmbed(
            img_size=input_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        ) if mlp_time_embed else nn.Identity()

        if self.num_classes > 0:
            self.label_emb = nn.Embedding(self.num_classes, embed_dim)
            self.extras = 2
        else:
            self.extras = 1

        self.pos_embed = nn.Parameter(torch.zeros(1, self.extras + num_patches, embed_dim))
        self.frame_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))

        self.in_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        self.mid_block = Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint)

        self.out_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, skip=skip, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        self.norm = norm_layer(embed_dim)
        self.patch_dim = patch_size ** 2 * in_chans
        self.decoder_pred = nn.Linear(embed_dim, self.patch_dim, bias=True)
        self.final_layer = nn.Conv2d(self.in_chans, self.in_chans * 2, 3, padding=1) if conv else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.frame_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def forward_(self, x, timesteps, y=None):
        x = self.patch_embed(x) # 48, 256, 1152
        # print(x.shape)
        B, L, D = x.shape 

        time_token = self.time_embed(timestep_embedding(timesteps, self.embed_dim)) # 3, 1152
        # print(time_token.shape)
        time_token = time_token.unsqueeze(dim=1) # 3, 1, 1152
        x = torch.cat((time_token, x), dim=1)

        if y is not None:
            label_emb = self.label_emb(y)
            label_emb = label_emb.unsqueeze(dim=1)
            x = torch.cat((label_emb, x), dim=1)
        x = x + self.pos_embed

        skips = []
        for blk in self.in_blocks:
            x = blk(x)
            skips.append(x)

        x = self.mid_block(x)

        for blk in self.out_blocks:
            x = blk(x, skips.pop())

        x = self.norm(x)
        x = self.decoder_pred(x)
        assert x.size(1) == self.extras + L
        x = x[:, self.extras:, :]
        x = unpatchify(x, self.in_chans)
        x = self.final_layer(x)
        return x
    
    def forward(self, x, timesteps, y=None):
        # print(x.shape)
        batch, frame, _, _, _ = x.shape
        # 这里rearrange后每隔f是同一个视频
        x = einops.rearrange(x, 'b f c h w -> (b f) c h w')  # 3 16 4 256 256
        x = self.patch_embed(x) # 48, 256, 1152
        B, L, D = x.shape 

        time_token = self.time_embed(timestep_embedding(timesteps, self.embed_dim)) # 3, 1152
        # timestep_spatial的repeat需要保证每f帧为同一个timesteps
        time_token_spatial = einops.repeat(time_token, 'n d -> (n c) d', c=frame) # 48, 1152
        time_token_spatial = time_token_spatial.unsqueeze(dim=1) # 48, 1, 1152
        x = torch.cat((time_token_spatial, x), dim=1)  # 48, 257, 1152

        if y is not None:
            label_emb = self.label_emb(y)
            label_emb = label_emb.unsqueeze(dim=1)
            x = torch.cat((label_emb, x), dim=1)
        x = x + self.pos_embed

        skips = []
        for i in range(0, len(self.in_blocks), 2):
            # print('The {}-th run'.format(i))
            spatial_block, time_block = self.in_blocks[i:i+2]
            x = spatial_block(x)
           
            # add time embeddings and conduct attention as frame.
            x = einops.rearrange(x, '(b f) t d -> (b t) f d', b=batch) # t 代表单帧token数; 771, 16, 1152; 771: 3 * 257
            skips.append(x)
            # print(x.shape)

            if i == 0:
                x = x + self.frame_embed # 771, 16, 1152

            x = time_block(x)

            x = einops.rearrange(x, '(b t) f d -> (b f) t d', b=batch) # 48, 257, 1152
            skips.append(x)

        x = self.mid_block(x)

        for i in range(0, len(self.out_blocks), 2):
            # print('The {}-th run'.format(i))
            spatial_block, time_block = self.out_blocks[i:i+2]
            x = spatial_block(x, skips.pop())

            # add time embeddings and conduct attention as frame.
            x = einops.rearrange(x, '(b f) t d -> (b t) f d', b=batch) # t 代表单帧token数; 771, 16, 1152; 771: 3 * 257

            x = time_block(x, skips.pop())

            x = einops.rearrange(x, '(b t) f d -> (b f) t d', b=batch) # 48, 256, 1152
        

        x = self.norm(x)
        x = self.decoder_pred(x)
        assert x.size(1) == self.extras + L
        x = x[:, self.extras:, :]
        x = unpatchify(x, self.in_chans)
        x = self.final_layer(x)
        x = einops.rearrange(x, '(b f) c h w -> b f c h w', b=batch)
        # print(x.shape)
        return x
    
def UViT_XL_2(**kwargs):
    return UViT(patch_size=2, in_chans=4, embed_dim=1152, depth=28,
                num_heads=16, mlp_ratio=4, qkv_bias=False, mlp_time_embed=4,
                use_checkpoint=True, conv=False, **kwargs)

def UViT_L_2(**kwargs):
    return UViT(patch_size=2, in_chans=4, embed_dim=1024, depth=20,
                num_heads=16, mlp_ratio=4, qkv_bias=False, mlp_time_embed=False,
                use_checkpoint=True, **kwargs)

# 没有L以下的，UViT中L以下的img_size为64

UViT_models = {
    'UViT-XL/2': UViT_XL_2, 'UViT-L/2':  UViT_L_2
}


if __name__ == '__main__':

    
    nnet = UViT_XL_2().cuda()
    
    imgs = torch.randn(3, 16, 4, 32, 32).cuda()
    timestpes = torch.tensor([1, 2, 3]).cuda()

    outputs = nnet(imgs, timestpes)
    print(outputs.shape)
    
