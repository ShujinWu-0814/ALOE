from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
from trl import SFTConfig, SFTTrainer
from tqdm import tqdm
from datasets import Dataset, load_dataset
import wandb
import datasets
import argparse

wandb.login()

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
parser.add_argument('--data', required=True)
args = parser.parse_args()


if args.model == 'llama3':
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
elif args.model == 'mistral':
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
elif args.model == 'qwen2':
    model_id = "Qwen/Qwen2-7B-Instruct"
elif args.model == 'olmo':
    model_id = "allenai/OLMo-7B-0724-Instruct-hf"

HF_token = 'xxxxxx'

wandb.init(
    project='Alignment on the fly',
    config={'method':'qwen_SFT_llama+claude+gpt_data_6r'}
)

tokenizer = AutoTokenizer.from_pretrained(model_id,trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    token = HF_token,
    trust_remote_code=True
)



with open('../datasets/conversations.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

codeact = load_dataset("xingyaoww/code-act")
ultrachat = load_dataset("HuggingFaceH4/ultrachat_200k", split='train_sft')
ultrachat = ultrachat.add_column('rounds', [len(i) for i in ultrachat['messages']])

ultrachat = ultrachat.sort('rounds',reverse = True)[:4000]
dpo_data = {'text': []}


for item in tqdm(data):
    conversation = []
    for i in range(10):
        conversation.append({'role': 'user', 'content': item['conversations'][0]['user']})   
        conversation.append(
            {'role': 'assistant', 'content': item['conversations'][i]['assistant'][args.data]}
        )
        templated = tokenizer.apply_chat_template(conversation, tokenize=False, padding=True, max_length=8192, truncation=True)
        dpo_data['text'].append(templated)


codeact = codeact.map(
    lambda example: {'text': tokenizer.apply_chat_template(
        example['conversations'], 
        tokenize=False, 
        padding=True, 
        max_length=8192, 
        truncation=True
    )}, 
    remove_columns=['id', 'conversations']
)['codeact']

ultrachat = Dataset.from_dict(ultrachat).map(
    lambda example: {'text': tokenizer.apply_chat_template(
        example['messages'], 
        tokenize=False, 
        padding=True, 
        max_length=8192, 
        truncation=True
    )}, 
    remove_columns=['prompt_id', 'prompt', 'messages', 'rounds']
)

data = Dataset.from_dict(dpo_data)

mixed_data = datasets.concatenate_datasets([data, codeact])
# mixed_data = codeact
# mixed_data = data
# print(mixed_data)
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
#     return_prompt_and_responses,
#     batched=True,
#     remove_columns=original_columns
# )



# tokenizer.add_special_tokens({"pad_token": "<unk>"})
# model.config.pad_token_id = tokenizer.pad_token_id


training_args = SFTConfig(
    output_dir="./model_output/{}_sft_{}".format(args.model, args.data),
    dataset_text_field='text',
    report_to='wandb',
    remove_unused_columns=False,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=48,
    logging_steps=1,
    learning_rate=1e-5,
    packing=True,
    num_train_epochs=1,
    save_strategy='epoch'
)

sft_trainer = SFTTrainer(
    model, 
    train_dataset=mixed_data, 
    args=training_args,
    max_seq_length=8192
)


sft_trainer.train()
sft_trainer.save_model(training_args.output_dir)
