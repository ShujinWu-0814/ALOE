# Aligning LLMs with Individual Preferences via Interaction


## ✨Overview
In this work, we train LLMs that can “interact to align”, essentially cultivating the meta-skill of LLMs to implicitly infer the unspoken personalized preferences of the current user through multi-turn conversations, and then dynamically align their following behaviors and responses to these inferred preferences. Our approach
involves establishing a diverse pool of 3,310 distinct user personas using iterative self-generation. Then guided by these personas, we leverage multi-LLMs collaboration to develop a multi-turn preference dataset containing 3K+ multi-turn conversations in tree structures. We then finetune various LLMs using SFT and reinforcement learning.

![ALOE](./ALOE.pdf)
## 🔧Requirements
The required Python packages for running this repo are listed in [requirements.txt](./requirements.txt). To install these pacakages at one time, plaese run
```shell
pip install -r requirements.txt
```

You also need to set up variable `openai.api_key` in [persona_generation.py](./data_generation/persona_generation.py), [data_generation.py](./data_generation/data_generation.py), and [eval_own.py](./evaluation/eval_own.py).

## 📊Datasets
Our curated user profile and personality pool used for guiding both training and evaluation, as well as the multi-turn conversations based preference dataset are stored in folder [datasets](./datasets).


## 🔥Training
The training scripts for SFT and reinforcement learning(DPO) are in folder [training](./training). You need to specify the base model you are going to finetune in your command line argument '--model' when running both of the scripts. If you are doing SFT, you should add one more argument '--data' to specify whether you want to use the preferred response or rejected response in the preference dataset for finetuning. For example, if you want to finetune llama3 using SFT on preferred response, you may run
```shell
python training/SFT.py --model llama3 --data preferred
```
If you want to do SFT on rejected response, change '--data preferred' to '--data rejected'.

If you want to finetune mistral using dpo, you should run
```shell
python training/dpo.py --model mistral
```
Models you can choose from include 'llama3', 'mistral', 'qwen2', 'olmo'. If you want to finetune other models, you can add its huggingface path into line 19-26 in [SFT.py](./training/SFT.py) or line 17-24 in [dpo.py](./training/dpo.py). Specifically, you shuold add two lines like
```shell
elif args.model == 'Model Name':
    model_id = 'Its huggingface path'
```
and specify the model name in command line argument when running scripts.


## 🧐Evaluation
To evaluate how models can align with customized human preferences during interactive conversations, you'll need to run file [eval_own.py](./evaluation/eval_own.py). You can evaluate both the base models and models after finetuning(the finetuned model will be stored in a new folder 'model_output' if you use our training scripts.)
Similar to training, you still need to specify the model you want to evaluate in the argument of command-line. If you want to evaluate the base model including llama3, qwen2, mistral, or olmo, you can directly specify the model name. For example, 
```shell
python evaluation/eval_own.py --model llama3
```
If you finished finetuning with our training scripts and wish to evaluate the finetuned model, you need to format your argument as 'model_method'. For instance, to evaluate llama3 after dpo, you should run 
```shell
python evaluation/eval_own.py --model llama3_dpo
```
To evaluate mistral after SFT on preferred response only, you should run
```shell
python evaluation/eval_own.py --model mistral_sft_preferred
```
If you want to evaluate mistral after SFT on rejected response, simply change --model mistral_sft_rejected' to '--model mistral_sft_rejected'.
