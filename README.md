# Fact Verifier with Knowledge Transfer Approach

This repo includes official code for [_How to Train Your Fact Verifier: Knowledge Transfer with Multimodal Open Models"](https://aclanthology.org/2024.findings-emnlp.764/)

## Setting up the environment
```
conda install pytorch torchvision torchaudio -c pytorch -c nvidia
pip install -r requirements.txt
```
We use `wandb` for logging. `WANDB_KEY` should be also set as an environmental variable. For more details, set up the `wandb` by following the instruction [here](https://docs.wandb.ai/quickstart).

## Training
```
torchrun --standalone
         --nproc_per_node={num_gpus} 
         verification/train.py -cn {config_name} 
         'model_name={model_name}'
```

- `config_name` is the config name.  
   - For the single dataset, we use the abbreviation from the first three letters (e.g., `moc` for `mocheg`).
   - For the multi-dataset, we combine the abbreviation of each datasets (e.g., `moc_fak` for `mocheg` and `fakeddit` ).
   - For explanation augmented datasets, we append the settings of each explanation configuration (e.g., `moc_eg_gpt_guided` for `mocheg` with GPT guided explanation).
- `model_name` is the model name. It can be one of the following four.
    - `openai/clip-vit-base-patch32`
    - `openai/clip-vit-large-patch14`
    - `openai/clip-vit-large-patch14-336` 
    - `llava-hf/llava-v1.6-mistral-7b-hf`

## Evaluation
```
 torchrun --standalone
          --nproc_per_node={num_gpus} 
          verification/infer.py
          -cn infer 'target_run_id={target_run_id}'
          'model_name={model_name}'
```
- `target_run_id` is the run id of the trained model.
- `model_name` is the model name. It should be the same as the one used for training.
