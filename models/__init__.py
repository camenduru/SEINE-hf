import os
import sys
sys.path.append(os.path.split(sys.path[0])[0])

from .dit import DiT_models
from .uvit import UViT_models
from .unet import UNet3DConditionModel
from torch.optim.lr_scheduler import LambdaLR

def customized_lr_scheduler(optimizer, warmup_steps=5000): # 5000 from u-vit
    from torch.optim.lr_scheduler import LambdaLR
    def fn(step):
        if warmup_steps > 0:
            return min(step / warmup_steps, 1)
        else:
            return 1
    return LambdaLR(optimizer, fn)


def get_lr_scheduler(optimizer, name, **kwargs):
    if name == 'warmup':
        return customized_lr_scheduler(optimizer, **kwargs)
    elif name == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, **kwargs)
    else:
        raise NotImplementedError(name)
    
def get_models(args):
    
    if 'DiT' in args.model:
        return DiT_models[args.model](
                input_size=args.latent_size,
                num_classes=args.num_classes,
                class_guided=args.class_guided,
                num_frames=args.num_frames,
                use_lora=args.use_lora,
                attention_mode=args.attention_mode
            )
    elif 'UViT' in args.model:
        return UViT_models[args.model](
                input_size=args.latent_size,
                num_classes=args.num_classes,
                class_guided=args.class_guided,
                num_frames=args.num_frames,
                use_lora=args.use_lora,
                attention_mode=args.attention_mode
            )
    elif 'TAV' in args.model:
        pretrained_model_path = args.pretrained_model_path
        return UNet3DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder="unet", use_concat=args.use_mask)
    else:
        raise '{} Model Not Supported!'.format(args.model)
    