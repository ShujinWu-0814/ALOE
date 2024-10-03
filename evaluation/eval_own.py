import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
from tqdm import tqdm
import random
import tenacity
import boto3
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Dict, Any
import logging
import re
from openai import OpenAI
import openai


def remove_asterisk_text(text):
    # Use regular expression to remove text between asterisks
    cleaned_text = re.sub(r'\*.*?\*', '', text)
    # Remove extra whitespace that might result from the removal
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

def call_gpt(messages):
    client = openai.Client(api_key='xxxxxx')
    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages = messages,
        max_tokens=300,
    )
    return response.choices[0].message.content

parser = argparse.ArgumentParser()
parser.add_argument('--model')
# parser.add_argument('--evaluator')
args = parser.parse_args()

if args.model == 'llama3':
    eval_model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
elif args.model == 'mistral':
    eval_model_path = "mistralai/Mistral-7B-Instruct-v0.3"
elif args.model == 'qwen2':
    eval_model_path = "Qwen/Qwen2-7B-Instruct"
elif args.model == 'olmo_orig':
    eval_model_path = "allenai/OLMo-7B-0724-Instruct-hf"
else:
    # eval_model_path = '/home/yangyic3/yangyic3/PIG/personality/models/' + 'model_output_' + args.model
    eval_model_path = '../model_output/' + args.model
eval_model = AutoModelForCausalLM.from_pretrained(eval_model_path, torch_dtype=torch.bfloat16).to('cuda')
eval_tokenizer = AutoTokenizer.from_pretrained(eval_model_path, model_max_length = eval_model.config.max_position_embeddings, truncation=True, truncation_side='left')


def chat_with_model(model, tokenizer, messages):
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt', return_dict=True, truncation = True, max_length=eval_model.config.max_position_embeddings).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens = 2048)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).split('assistant\n\n')[-1]



with open('../datasets/profile_eval.jsonl', 'r') as f:
    profiles = f.readlines()

with open('../datasets/personality_eval.jsonl', 'r') as f:
    personalities = f.readlines()

evaluate_prompt = '''
You will be given a user's profile, personalities, and a message that the user sent to a chatbot. You will also be given a response from a model. Your task is to carefully evaluate how much the response is tailored to the user's potential preferences based on the user's profile and personalities.
Here is the user's profile: {}
Here is the user's personalities: {}
Here is the user's message: {}
Here is the model's response: {}

You should follow the criteria for evaluation:
1. Is the conversational style of the message tailored to the user's personalities?
2. Is the content or topic relevant to the user's profile?
3. Is response human-like, engaging, and concise?
You should give a score to the response ranging from 1-5, where 1 represents the least tailored to the user and 5 represents the most user-aligned. Please do not include any analysis about how you evaluate the responses. Only output the score from 1-5. 
'''

num_rounds = 10

results = [
    [] for _ in range(num_rounds)
]

# for profile in tqdm(random.sample(profiles, 10)):
#     personality = random.choice(personalities)
for profile in tqdm(profiles):
    for personality in personalities[:5]:
        role_play_conv = []
        eval_model_conv = []
        for i in tqdm(range(num_rounds)):
            if i == 0:
                role_play_prompt = '''Your task is to play the role of a person with the following profile and personalities traits and chat with a chatbot:
                            Profile: {}
                            Personalities: {}
                            Please ignore the gender pronouns in the personalities and use the correct pronouns based on the given profile.
                            Please follow the requirements:
                            1. You should determine the topic of conversation based on the given profile. You should determine the conversational styles based on the given personalities.
                            2. IMPORTANTLY!!! You should only reveal partial information about your profile in each round of conversation instead of disclosing all the provided information at once.
                            3. Keep in mind that you are chatting with a friend instead of a robot or assistant. So do not always seek for advice or recommendations.
                            4. Do not include any analysis about how you role-play this user. Only output your messages content.
                            Now, initiate the conversation with the chatbot in whatever way you like. Please always be concise in your questions and responses and remember that you are pretending to be a human now, so you should generate human-like language.'''.format(profile, personality)
                #gpt speaking
                role_play_conv.append({'role': 'user', 'content': role_play_prompt})
                role_play_message = call_gpt(role_play_conv)
                role_play_message = remove_asterisk_text(role_play_message)
                eval_model_conv.append({'role': 'user', 'content': role_play_message})
                role_play_conv.append({'role': 'assistant', 'content': role_play_message})
                
                #eval model speaking
                eval_model_message = chat_with_model(eval_model, eval_tokenizer, eval_model_conv)
                
                #evaluate response
                eval_result = call_gpt([{'role':'user', 'content':evaluate_prompt.format(profile, personality, role_play_message, eval_model_message)}])
            
                
                score = int(eval_result)
                results[i].append(score)
                
                #random pick
                # model_message = random.choice([eval_model_message, baseline_model_message]) 
                eval_model_conv.append({'role': 'assistant', 'content': eval_model_message})
                role_play_conv.append({'role': 'user', 'content': eval_model_message})
                                

            else:
                #gpt speaking
                try:
                    role_play_message = call_gpt(role_play_conv)
                except Exception as e:
                    print(e)
                    break
                role_play_message = remove_asterisk_text(role_play_message)
                eval_model_conv.append({'role': 'user', 'content': role_play_message})
                role_play_conv.append({'role': 'assistant', 'content': role_play_message})
                
                #eval model and baseline model speaking
                eval_model_message = chat_with_model(eval_model, eval_tokenizer, eval_model_conv)
                
                
                #evaluate two models'responses
                eval_result = call_gpt([{'role':'user', 'content':evaluate_prompt.format(profile, personality, role_play_message, eval_model_message)}])
            
                score = int(eval_result)
                results[i].append(score)
                
                print(evaluate_prompt.format(profile, personality, role_play_message, eval_model_message))
                print(score)
                print("*" * 89)
                
                
                # model_message = random.choice([eval_model_message, baseline_model_message])
                eval_model_conv.append({'role': 'assistant', 'content': eval_model_message})
                role_play_conv.append({'role': 'user', 'content': eval_model_message})

        # print(results)                         
final_results = [sum(r) / len(r) for r in results]
print('{} final vertical evaluation results:'.format(args.model))
print(final_results)
            
        
    
