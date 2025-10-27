import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer
from pathlib import Path
from datetime import datetime
from utils import extract_image_token_attention, resize_and_center_crop, save_overlay_heatmap



# === Experiment Settings ===

'''
num_tokens_to_generate:  options: int                  // number of newly generated token
steps_config             options: list, 'all'          // range( 1, number of token to be analyzed) !DO NOT INCLUDE 0!
layers_config            options: list                 // range(0, 32)
heads_config             options: list, 'all'          // range(0, 32)
reduction_config         options: 1, 2, 3, 'none'
                         option 1: mean over steps, layers and heads
                         option 2: mean over layers and heads
                         option 3: mean over heads
inspect                  options: True, False   
   
'''   

num_tokens_to_generate = 15 

steps_config = 'all'   

layers_config = [14] 
         
heads_config = [13, 24] 
                   
reduction_config = 2 
                              
inspect = True           



# === Prompts, Pathways, Filenames  And Documentation ===

prompts = ["<image>\nIs there food in the image?"]

image_paths = ["/pfs/work9/workspace/scratch/ul_suh74-Pixtral/Attention_Pipeline/FINAL_REPO/dataset/image_1.jpg"]

model_path = "/pfs/work9/workspace/scratch/ul_suh74-Pixtral/llava-1.5-7b-hf/llava-1.5-7b-hf/"

base_out_dir = Path(f"/pfs/work9/workspace/scratch/ul_suh74-Pixtral/Attention_Pipeline/FINAL_REPO/results/")

base_out_dir.mkdir(parents=True, exist_ok=True) 



# === Load model and input ===

processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForVision2Seq.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    local_files_only=True
)


# === Starts Experiment ===

for idx, (img_path, prompt) in enumerate(zip(image_paths, prompts), start=1):
    print(f"\n=== Processing Image {idx}: {img_path} ===")
    
    
    start_time = datetime.now()
    timestamp = start_time.strftime("%Y_%m_%d_%H_%M") # Year_Month_Day_Hour_Minute
    
    out_dir = base_out_dir / f"Experiment_{idx}"
    out_dir.mkdir(parents=True, exist_ok=True)


    overlay_filename = f"Experiment_{idx}_{timestamp}.png"
    overlay_path = out_dir / overlay_filename
    overlay_path_str = str(overlay_path) 
    

    if inspect == True:
        documentation_filename = f"Experiment_{idx}_{timestamp}.txt"
        documentation_path = out_dir / documentation_filename
        with open(documentation_path, "w") as f:
            pass

    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

    
    
# === Find image token indices ===

    input_ids = inputs["input_ids"]
    mask_image = (input_ids == 32000)
    mask_text = (input_ids != 32000) & (input_ids != 1)
    image_prompt_positions = torch.nonzero(mask_image, as_tuple=True)[1]
    text_prompt_positions = torch.nonzero(mask_text, as_tuple=True)[1]
    num_image_tokens = len(image_prompt_positions) # 576
    


# === Generate outputs ===

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens= num_tokens_to_generate,
            output_attentions=True,
            return_dict_in_generate=True,
        )



# === Safety check ===

    if not hasattr(outputs, "attentions"):
        raise RuntimeError("No outputs.attentions found. Make sure output_attentions=True at generation.")

    num_steps = len(outputs.attentions)   
    output_token_position = outputs.sequences[0][- (num_steps-1):]
    generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    
    output_token = []
    
    for i in range(num_steps-1):
        token_id = output_token_position[i].item() 
        token_text = tokenizer.decode(token_id, skip_special_tokens=True)
        output_token.append(token_text)
        
    if steps_config == 'all':
        steps_config = list(range(1,len(outputs.attentions)))
        
    elif len(steps_config) > len(outputs.attentions):
        raise ValueError("Scope of steps is out of bounds. Please reduce the steps to be analyzed.")

    if heads_config == 'all':
        heads_config = list(range(32))
        
    elif len(heads_config) > 31:
        raise ValueError("Scope of heads is out of bounds. Please reduce the heads to be analyzed.")


# === Get the attention ===

    attn_tensor = extract_image_token_attention(
        attention = outputs,
        image_token_positions = image_prompt_positions,
        text_token_positions = text_prompt_positions,
        
        steps= steps_config,
        layers= layers_config,
        heads= heads_config,
        reduction= reduction_config)  



# === Handle different attention outpt ===

    if attn_tensor.ndim == 1:
        # Shape: [576]
        token = False
        save_overlay_heatmap(attn_tensor, overlay_path, num_image_tokens, img_path, token)
    
    elif attn_tensor.ndim == 2:
        # Shape: [N, 576]
        for i, step in enumerate(attn_tensor):
            path_i = overlay_path_str.replace(".png", f"_Step{steps_config[i]}.png")
            token = output_token[i]
            save_overlay_heatmap(step, path_i, num_image_tokens, img_path, token)

    elif attn_tensor.ndim == 3:
        # Shape: [layer, heads, 576]
        steps, layers, _ = attn_tensor.shape
        for step in range(steps):
            for layer in range(layers):
                heatmap = attn_tensor[step, layer]  # shape [576]
                path_ij = overlay_path_str.replace(".png", f"_Step{steps_config[step]}_Layer{layers_config[layer]}.png")
                token = output_token[step]
                save_overlay_heatmap(heatmap, path_ij, num_image_tokens, img_path, token)

    elif attn_tensor.ndim == 4:
        # Shape: [steps, layers, heads, 576]
        steps, layers, heads, _ = attn_tensor.shape
        for step in range(steps):
            for layer in range(layers):  
                for head in range(heads):
                    heatmap = attn_tensor[step, layer, head]  # shape [576]
                    path_ij = overlay_path_str.replace(".png", f"_Step{steps_config[step]}_Layer{layers_config[layer]}_Head{heads_config[head]}.png")
                    token = output_token[step]
                    save_overlay_heatmap(heatmap, path_ij, num_image_tokens, img_path, token)

    else:
        raise ValueError(f"Unsupported attn_tensor shape: {attn_tensor.shape}")



# === Inspect the input and output ===

    if inspect == True:

        # Experiment duration
        end_time = datetime.now()
        duration = end_time - start_time
    
        days = duration.days
        hours, remainder = divmod(duration.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
    
        formatted_start_time = start_time.strftime("%Y/%m/%d %H:%M") # Year/Month/Day Hour:Minute
        formatted_end_time = end_time.strftime("%Y/%m/%d %H:%M") # Year/Month/Day Hour:Minute
    
        if reduction_config == 1:
            formatted_reduction_config = 'mean over steps, layers and heads'

        elif reduction_config == 2:
            formatted_reduction_config = 'mean over layers and heads'
        
        elif reduction_config == 3:
            formatted_reduction_config = 'mean over heads'   

        elif reduction_config == 'none':
            formatted_reduction_config = 'none'      
        
        output_token_position = outputs.sequences[0][- (num_steps-1):] # disregard the first step
        
        with open(documentation_path, "w") as f:
          # Header
          f.write("=== XAI Experiment ======\n\n")
          f.write(f"Start time: {formatted_start_time}\n")
          f.write(f"Endt time: {formatted_end_time}\n")
          f.write(f"Duration: {days} days, {hours} hours, {minutes} minutes, {seconds} seconds\n\n")    
          f.write("="*25 + "\n\n")
    
          f.write(f"Prompt: \n{prompt}\n\n")    
          f.write(f"Examined steps: {steps_config}\n")
          f.write(f"Examined layers: {layers_config}\n")
          f.write(f"Examined attention heads: {heads_config}\n")
          f.write(f"Aggregation: {formatted_reduction_config}\n\n")
          f.write("="*25 + "\n\n")  
    
          # Result
          generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
          f.write(f"Generated text:\n{generated_text}\n\n")
          f.write("="*25 + "\n\n")
          
          # Map output token to step
          for step, token in enumerate(output_token):
              f.write(f"step: {step + 1} token: {token}\n\n")
          f.write("="*25 + "\n\n")
          
          # Additional information
          f.write(f"Inputs keys: {list(inputs.keys())}\n")
          f.write(f"Input IDs: {inputs.input_ids}\n")
          f.write(f"Image processor: {processor.image_processor}\n")
          f.write(f"List of output keys: {list(outputs.keys())}\n\n")
          f.write(f"Type of output sequences: {type(outputs['sequences'])}\n")
          f.write(f"Shape of output sequences: {outputs['sequences'].shape}\n")
          f.write(f"Output sequences:\n{outputs['sequences']}\n\n")







