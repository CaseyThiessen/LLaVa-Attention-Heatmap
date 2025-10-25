What is this? Overview


Explainable AI Tool/ Toolkit to extract the attention of the llava-1.5-7b-hf model

https://huggingface.co/llava-hf/llava-1.5-7b-hf

Address this black box problem

Our Pipeline
Supports extraction, aggregation and visualization of the attention for specific generation steps, layers and attention heads 



Example with labeled output images



Settings

Currently reduction method for the first step / only   mean over layers and heads is supported
Extract attention values for which step(s), layer(s), attention head(s)
Extract multiple heatmaps for specificed parameters or aggregate with mean into one single heatmap

Either chose to investigate step == 0 
Or any other range from 1 to max steps
Step == 0 is a different processing mode of the model then all the susequent steps 
    

Choose whether to inspect the model's input and output.
If inspect == True:
- The inputs and outputs of the model will be printed.
- Additionally, a text file will be created to document the experiment,
including details such as the experiment timestamp, inputs, outputs,
and generated text.



Quick Start Hugging Face/ LLava Analysis / Usage Example

Reference git to setup the bw cluster



