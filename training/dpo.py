from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
from trl import DPOConfig, DPOTrainer
from tqdm import tqdm
from datasets import Dataset
import wandb
import datasets
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
args = parser.parse_args()


if args.model == 'llama3':
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
elif args.model == 'mistral':
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
elif args.model == 'qwen2':
    model_id = "Qwen/Qwen2-7B-Instruct"
elif args.model == 'olmo':
    model_id = "allenai/OLMo-7B-0724-Instruct-hf"


wandb.login()


HF_token = 'xxxxxxx'

wandb.init(
    project='Alignment on the fly',
    config={'method':'dpo_llama_claude_gpt'}
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map = 'auto',
    torch_dtype=torch.bfloat16,
    token = HF_token
)

model_ref = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map = 'auto',
    torch_dtype=torch.bfloat16,
    token = HF_token
)

with open('../datasets/conversations.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
tokenizer.bos_token_id = tokenizer.eos_token_id



prompt = []
chosen = []
rejected = []
for item in tqdm(data):
    conversation = [
    {'role': 'user', 'content': item['conversations'][0]['user']}
    ]
    for i in range(len(item['conversations'])):
        templated = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        prompt.append(templated)

        chosen.append(item['conversations'][i]['assistant']['preferred'] + tokenizer.eos_token)
        rejected_responses = item['conversations'][i]['assistant']['rejected']
        random.choice(rejected_responses)
        rejected.append(random.choice(rejected_responses) + tokenizer.eos_token)
        # rejected.append(item['conversations'][i]['assistant']['rejected'] + tokenizer.eos_token)
        
        selected = item['conversations'][i]['chosen']
        conversation.append(
            {'role': 'assistant', 'content': item['conversations'][i]['assistant'][selected]}
        )
        if i < len(item['conversations']) - 1:
            conversation.append(
                {'role': 'user', 'content': item['conversations'][i+1]['user']}
            )

dpo_data = {
    "prompt": prompt,
    "chosen": chosen,
    "rejected": rejected
}

# print(dpo_data['prompt'][0])
# print(dpo_data['chosen'][0])
# print(dpo_data['rejected'][0])

dpo_data = Dataset.from_dict(dpo_data)
# # print(dpo_data)

# def return_prompt_and_responses(samples):
#     return {
#         "prompt": [
#             "Question: " + question + "\n\nAnswer: "
#             for question in samples["question"]
#         ],
#         "chosen": samples["response_j"],   # rated better than k
#         "rejected": samples["response_k"], # rated worse than j
#     }

# dataset = datasets.load_dataset(
#     "lvwerra/stack-exchange-paired",
#     split="train[:1000]",
#     data_dir="data/rl"
# )

# original_columns = dataset.column_names

# new_dataset = dataset.map(
#     batched=True,
#     remove_columns=original_columns
# )



# tokenizer.add_special_tokens({"pad_token": "<|reserved_special_token_0|>"})
# model.config.pad_token_id = tokenizer.pad_token_id


training_args = DPOConfig(
    output_dir="../model_output/{}_dpo".format(args.model),
    beta=0.9,
    report_to='wandb',
    remove_unused_columns=False,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=12,
    logging_steps=1,
    learning_rate=1e-5,
    num_train_epochs=1
)

dpo_trainer = DPOTrainer(
    model, 
    model_ref,
    train_dataset=dpo_data, 
    tokenizer=tokenizer, 
    args=training_args
)


dpo_trainer.train()
dpo_trainer.save_model(training_args.output_dir)
