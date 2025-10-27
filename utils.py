import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as TF
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer
from PIL import Image
from pathlib import Path
from typing import Any, Iterable, Union



# === Extract the attention of the model ===

def extract_image_token_attention(
    attention,
    image_token_positions,
    text_token_positions,
    steps,
    layers,
    heads,
    reduction,      
):

    """
    Extracts and optionally averages attention from generated tokens to image tokens.
    
    [B, H, T, S] = Batch, Heads, Target tokens, Source tokens 
    
    Args:
        attention: Output of model.generate with output_attentions=True
        image_token_positions: List of token indices corresponding to image tokens
        text_token_positions: List of token indices corresponding to text tokens
        steps: Int or list of generation steps      
        layers: Int or list of layer indices to extract from
        heads: 'all', int, or list of head indices
        reduction: 'mean' or 'none'

    Returns:
        A tensor of shape:
        - [576] if reduction= 1
        - [num_steps, 576] if reduction= 2
        - [num_steps, num_layers, 576] if reduction= 3
        - [num_steps, num_layers, num_heads, 576] if reduction= 'none'
    """
    
    collected_steps = []
    collected_layers = []
    
    max_val_image_position = image_token_positions.max()
    max_val_text_position = text_token_positions.max()

    for step_idx in steps:
        step_attns = attention.attentions[step_idx]  # list of tensors per layer

        for layer_idx in layers:
            attn = step_attns[layer_idx]  # [B, H, T, S] 

            # Step 1: remove the first and third dimension for tensor shape (1, 32, [newly generated token: shape = 1], [input_shape + all new generated token])
            attn = attn.squeeze(0, 2)

            # Step 2: slice to relate the output text token to the input image token 
            attn = attn[:, 1:max_val_image_position+1] 

            if heads != 31:
                attn = attn[heads, :]
                    
            collected_layers.append(attn) 
                
        collected_steps.append(collected_layers.copy())                              
        collected_layers.clear()
 
    inner_stacks = [torch.stack(inner) for inner in collected_steps]
    stacked = torch.stack(inner_stacks)

    if reduction == 1:
        return stacked.mean(dim=(0, 1, 2))  # mean over steps, layers and heads

    elif reduction == 2:
        return stacked.mean(dim=(1, 2))  # mean over layers and heads

    elif reduction == 3:
        return stacked.mean(dim=(2))  # mean over heads

    elif reduction == "none":
        return stacked  

    else:
        raise ValueError("Unsupported reduction mode.")



# === Reformat input image to match the preprocessed image of the model ===

def resize_and_center_crop(img, size=336):
    w, h = img.size
    scale = size / min(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = img.resize((new_w, new_h), resample=Image.BICUBIC)
    left = (new_w - size) // 2
    top = (new_h - size) // 2
    img_cropped = img_resized.crop((left, top, left + size, top + size))
    
    return img_cropped
    
    
    
# === Generate and save overlay ===

def save_overlay_heatmap(heatmap_1d, output_path, number_of_image_tokens, original_image_path, generated_token):
    
    H = W = int(round(number_of_image_tokens ** 0.5))
    assert H * W == number_of_image_tokens, f"Can't reshape {number_of_image_tokens} into square."

    original_image = Image.open(original_image_path).convert("RGB")
    aligned_image = resize_and_center_crop(original_image, size=336)
    image_np = np.array(aligned_image)

    heatmap_2d = heatmap_1d.view(H, W).cpu().numpy()
    heatmap_tensor = torch.tensor(heatmap_2d).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    heatmap_resized = TF.resize(heatmap_tensor, size=(336, 336), interpolation=TF.InterpolationMode.BICUBIC)
    heatmap_resized = heatmap_resized.squeeze().numpy()

    plt.figure(figsize=(6, 6))
    plt.imshow(image_np)
    plt.imshow(heatmap_resized, cmap='jet', alpha=0.4)
    plt.axis("off")

    if generated_token is not False:
        plt.figtext(
            0.5, -0.05, str(generated_token),
            wrap=True, ha='center', fontsize=32
        )

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Saved overlay heatmap image to: {output_path}")
    plt.close()



