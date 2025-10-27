# Explainable AI Toolkit for LLaVa

## Overview
This project presents an **Explainable AI (XAI) toolkit** designed to address the **black-box problem** in multimodal language models.  
It enables users to **extract, aggregate, and visualize attention** from the [`llava-1.5-7b-hf`](https://huggingface.co/llava-hf/llava-1.5-7b-hf) model.

The toolkit provides an interpretable view of the model’s internal reasoning by generating **attention heatmaps** that illustrate the relationship between generated text tokens and visual input regions.
This toolkit contributes to enhancing transparency and interpretability in multimodal generative models, allowing researchers to analyze how multimodal language models integrate visual and textual information during generation.

Our pipeline supports:
- Extraction of attention weights across **specific steps, layers and attention heads**
- **Aggregation** of the attention output for clearer interpretation
- **Visualization** of attention as overlayed heatmaps on input images


## Example: Attention Visualization Output

### Input Image & Prompt
<table style="border: none; border-collapse: collapse;">
<tr>
<td style="border: none; padding: 5px;">
<strong>Prompt:</strong> <em>Is there food in the image?</em><br>
<strong>Answer:</strong> <em>Yes, there is a plate of food in the image.</em>
</td>
<td style="border: none; padding: 5px;">
<img src="dataset/image_1.jpg" width="300">
</td>
</tr>
</table>



### Attention Visualization
The attention heatmaps illustrate how the model focuses on specific regions of the input image during token generation.

<p align="center">
 <img src="results/Experiment_1/Experiment_1_2025_10_27_16_31_Step1.png" width="200">
 <img src="results/Experiment_1/Experiment_1_2025_10_27_16_31_Step2.png" width="200">
 <img src="results/Experiment_1/Experiment_1_2025_10_27_16_31_Step3.png" width="200">
 <img src="results/Experiment_1/Experiment_1_2025_10_27_16_31_Step4.png" width="200">
 <img src="results/Experiment_1/Experiment_1_2025_10_27_16_31_Step5.png" width="200">
 <img src="results/Experiment_1/Experiment_1_2025_10_27_16_31_Step6.png" width="200">
 <img src="results/Experiment_1/Experiment_1_2025_10_27_16_31_Step7.png" width="200">
 <img src="results/Experiment_1/Experiment_1_2025_10_27_16_31_Step8.png" width="200">
 <img src="results/Experiment_1/Experiment_1_2025_10_27_16_31_Step9.png" width="200">
 <img src="results/Experiment_1/Experiment_1_2025_10_27_16_31_Step10.png" width="200">
 <img src="results/Experiment_1/Experiment_1_2025_10_27_16_31_Step11.png" width="200">
 <img src="results/Experiment_1/Experiment_1_2025_10_27_16_31_Step12.png" width="200">
</p>


## Quickstart & Usage

### 1. Environment Setup
For usage on the **bwHPC**, we recommend following the setup instructions provided in the [Medical_Imaging repository](https://github.com/DeveloperNomis/Medical_Imaging).  
That repository outlines the correct environment setup for working on the bwHPC system. Afterwards ensure that all dependencies are installed and compatible by running:
```bash
pip install -r requirements.txt
```

### 2. Model Download
Download the llava-1.5-7b-hf model by executing:
```bash
python download_model.py
```
This script will automatically retrieve the model weights and store them in the appropriate directory for later use.


### 3. Configuration
Open the attention.py file and navigate to the configuration section at the top of the script. Here, you can modify:
* File paths (also to the input images)
* Output directories
* Prompts
* Experiment parameters (Recommendations are based on research done by Kang et al. 2025):
  * reduction_config: Controls the dimensionality reduction used during attention aggregation by taking the mean.
    *  Recommended setting: 2
  * steps_config: Determines which step(s) to extract attention from.
    *  Recommended setting: 'all'
  * layers_config: Determines which layer(s) to extract attention from.
    *  Recommended setting: 14
  * heads_config: Determines which attention heads to extract attention from.
    *  Recommended setting: [13, 24] 
For more information see the documentation in the attention.py file.

### 4. Running the Attention Extraction
Execute the main attention extraction script:
```bash
python attention.py
```

Depending on your configuration, the pipeline will:
* Extract and optionally aggregate attention heatmaps across specified steps, layers and heads
* Save resulting visualizations as overlayed heatmaps in the output directory

If inspect == True a text log file will be created, documenting:
* Experiment runtime 
* Experiment parameters
* Input and output mappings
* Generated output text


### 5. Options
You can choose between:
* Generating multiple heatmaps for specified parameters or
* Aggregating results via mean aggregation into a single composite heatmap
* This flexibility allows for both fine-grained and global analysis of visual attention.


## References 
Kang, S., Kim, J., Kim, J., & Hwang, S.J. (2025, March 8). Your large vision-language model only needs a few attention heads for visual grounding. arXiv.org. https://arxiv.org/abs/2503.06287

<!-- 
<p align="center">
  <img src="assets/LLaVa_image.png" alt="Project Banner" width="70%">
</p>
--->

<p align="center">
  <i>Thanks for visiting — contributions and stars are always welcome ⭐</i>
</p>

