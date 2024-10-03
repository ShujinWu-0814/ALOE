import openai
from tqdm import tqdm
import transformers
import torch
import random
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import json
import tenacity
import boto3
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Dict, Any
import logging
from openai import OpenAI



def remove_asterisk_text(text):
    # Use regular expression to remove text between asterisks
    cleaned_text = re.sub(r'\*.*?\*', '', text)
    # Remove extra whitespace that might result from the removal
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text


HF_token = 'xxxxxx'


model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
llama = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, token = HF_token).to('cuda')
llama_tokenizer = AutoTokenizer.from_pretrained(model_path, token = HF_token)


client = OpenAI(api_key='xxxxxx')

def get_response(model, messages):
    # if model == 'claude':
    #     claude_messages = [{'messages': messages}]
    #     response = call_claude(claude_messages)[0]['claude_response']
    if model == 'llama':
        roles = ['user', 'assistant']
        llama_messages = []
        for i, text in enumerate(messages):
            role = roles[i % 2]  # Alternates between 'user' and 'assistant'
            llama_messages.append({'role': role, 'content': text})

        # model.generation_config.pad_token_id = model.generation_config.eos_token_id
        input_ids = llama_tokenizer.apply_chat_template(llama_messages, add_generation_prompt=True, return_tensors='pt',return_dict=True).to('cuda')
        outputs = llama.generate(**input_ids, pad_token_id=llama_tokenizer.eos_token_id)
        response = llama_tokenizer.decode(outputs[0], skip_special_tokens=True).split('assistant\n')[-1].strip()
    elif model == 'gpt':
        roles = ['user', 'assistant']
        llama_messages = []
        for i, text in enumerate(messages):
            role = roles[i % 2]  # Alternates between 'user' and 'assistant'
            llama_messages.append({'role': role, 'content': text})
        response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages = llama_messages,
        max_tokens=300,
        )
        response = response.choices[0].message.content   
    return response

# messages = ['hello!', 'hi! How are you today?', 'I am doing well, thank you. How about you?']
# print(get_response('llama', messages))

num_rounds = 10

with open('../datasets/profile.jsonl', 'r') as f:
    profiles = f.readlines()

with open('../datasets/personality.jsonl', 'r') as f:
    personalities = f.readlines()


profile_1 = "He is a 19-year-old college student studying anthropology. He has a strong interest in cultures. He plays the guitar, loves indie films, and is a member of a local theater group. He is always seen with a coffee in hand."
personality_1 = "He is curious and open-minded and always eager to learn new things. He has a strong sense of humor and is known for his wit. He is sociable and enjoys interacting with others. He is also a good listener and is always willing to help others."
conversation_1 = '''A: Hey! How are you doing today? I just finished my anthropology exam and I'm so relieved that it's over. I'v completed my freshman year of college!'''


# In this dialogue, A represents the user and B (if there exists one) represents the person A is interacting with. Focus specifically on the information mentioned by "A" to identify the elements of the profile and personalities that have been revealed. If you can not infer the gender of the user, you should use He/She in your response.
            # Based solely on the user's statements, return both the inferred partial user profile and personalities. Do not include any analysis of how you arrived at this conclusion and other new user information that is not in the provided profile and personalities in your response. And you should simply extract partial information in the original sentence structure or language instead of rephrasing it.\n
answer_1 = '''Profile: He/She is a college student studying anthropology.\nPersonalities: None'''

profile_2 = "He is a 19-year-old college student studying anthropology. He has a strong interest in cultures. He plays the guitar, loves indie films, and is a member of a local theater group. He is always seen with a coffee in hand."
personality_2 = "He is curious and open-minded and always eager to learn new things. He has a strong sense of humor and is known for his wit. He is sociable and enjoys interacting with others. He is also a good listener and is always willing to help others."
conversation_2 = ''''A: Hey! How are you doing today? I just finished my anthropology exam and I'm so relieved that it's over. I'v completed my freshman year of college!
            B: That's great! I'm glad to hear that you're done with your exam. What are your plans for the upcoming summer break?
            A: I'm embarking on a cinematic adventure to our local theaters to soak up some indie film goodness—yeah, I'm pretty much a fanatic about those artsy gems. Plus, I'm' cranking up my guitar skills to 'rock star'—or at least 'decent strummer'—and might even pen a couple of tunes. Super stoked for this break! It's going to be epic... or at least entertaining!'''

answer_2 = '''Profile: He/She is college student studying anthropology. He/She plays the guitar, loves indie films.\nPersonalities: He/She has a strong sense of humor and is known for his/her wit.'''



profile_extract_instruction = '''Analyze a conversation (presented below with 'A' as the user and 'B' as the interaction partner) to identify aspects of the user's profile and personality traits that have been revealed in the conversation:
{}

Review the user's profile and personality descriptions. 
Profile: {}
Personalities: {}
Focus specifically on the information mentioned by "A" to identify the elements of the profile and personalities that have been revealed. Use direct evidence from the user's statements to deduce disclosed details about their profile and personality. If personality traits are not evident, output 'None' for personalities. If the user's gender is unclear, use 'He/She'. 

Provide your findings in the following format without additional analysis:
Profile: [inferred user profile details]
Personalities: [inferred user personality traits]

Important!!! Please make conservative judgments, and only infer information that is obvious from the conversation. You should simply extract partial information in the original sentence structure or language instead of rephrasing it'''

determine_better_response = '''You will be given a user's profile, personality descriptions and two messages sent to the user. Your task is to determine which message would be preferred by the user based on the his profile and personalities. 
Below are the user's profile and personality descriptions. 
Profile: {}
Personalities: {}

And here are the two messages you need to compare:
Message 1: {}
Message 2: {}
You should not include any analysis about how you determine the better response. If you think Message 1 is preferred, output 'first'. If you think Message 2 is preferred, output 'second'.'''



total_count = 0

for i in range(10):
    for profile in tqdm(profiles):
        personality = random.choice(personalities)
        data = {}
        data['profile'] = profile
        data['personality'] = personality
        data['conversations'] = []
        final_conversations = ''''''
        for i in tqdm(range(num_rounds)):
            infer_persona = [profile_extract_instruction.format(conversation_1, profile_1, personality_1), answer_1, profile_extract_instruction.format(conversation_2, profile_2, personality_2),answer_2]
            
            hint = '''(Hint: Below is the known user profile and personalities based on the conversation history: 
            {}
            You should implicitely infer the user's preferences about the topic to discuss, the conversation style, the way others respond to themselves, etc based on tbese given profile and personalities.
            You should directly generate a response that is tailored to the potential user preferences.)'''
    # You should not mention the user's profile or personalities in your responses and you also should not state that you have inferred their preferences. You should just directly generate a tailored responses naturally.)

            count = 0
            if i == 0:
                flag = True
                while count < 3:
                    try:
                        A_messages = ['''Your task is to play the role of a person with the following profile and personalities traits and chat with a chatbot:
                            Profile: {}
                            Personalities: {}
                            Please ignore the gender pronouns in the personalities and use the correct pronouns based on the given profile.
                            Please follow the requirements:
                            1. You should determine the topic of conversation based on the given profile. You should determine the conversational styles based on the given personalities.
                            2. IMPORTANTLY!!! You should only reveal partial information about your profile in each round of conversation instead of disclosing all the provided information at once.
                            3. Keep in mind that you are chatting with a friend instead of a robot or assistant. So do not always seek for advice or recommendations.
                            4. Do not include any analysis about how you role-play this user. Only output your messages content.
                            Now, initiate the conversation with the chatbot in whatever way you like. Please always be concise in your questions and responses and remember that you are pretending to be a human now, so you should generate human-like language.'''.format(profile, personality)]
                        # 4. Do not include any descriptions of your facial expressions or gestures in your responses. Only output your messages content.
                        chatbot_A = get_response('gpt', A_messages)
                        chatbot_A = remove_asterisk_text(chatbot_A)
                        # print(chatbot_A)
                        # print("*" * 89)
                        # # exit the whole process
                        # exit()


                        final_conversations += 'A: ' + chatbot_A + '\n'
                        A_messages.append(chatbot_A)


                        infer_persona.append(profile_extract_instruction.format(final_conversations,profile, personality))
                        inferred_persona = get_response('gpt', infer_persona)
                        # print(inferred_persona)
                        inferred_profile = inferred_persona.split('Profile: ')[1].split('Personalities: ')[0].strip()
                        inferred_personality = inferred_persona.split('Profile: ')[1].split('Personalities: ')[1].strip()

                        # print(chatbot_A)
                        # print(profile)
                        # print(personality)
                        # print(inferred_persona)
                        # print("*" * 89)
                        # exit()

                        B_messages_instruct = [chatbot_A + '\n' + hint.format(inferred_persona) + "\n" + "(We are doing role-play so please reply me using very concise and **human-like** language (within 3 sentences)" ]
                        B_messages = [chatbot_A + "\n" + "(We are doing role-play so please reply me using very concise and **human-like** language (within 3 sentences)"]


                        # print(chatbot_A)
                        first = get_response('gpt', B_messages_instruct)
                        second = get_response('gpt', B_messages)
                        
                        better = get_response('gpt', [determine_better_response.format(profile, personality, first, second)])
                        if better == 'first':
                            preferred = first
                            rejected = second
                        else:
                            preferred = second
                            rejected = first
                        # print(preferred)
                        # print(rejected)
                        # print("*" * 89)

                        chatbot_B = random.choice([preferred, rejected])

                        if chatbot_B == preferred:
                            chosen = 'preferred'
                        else:
                            chosen = 'rejected'
                        
                        data['conversations'].append({
                            'user': chatbot_A,
                            'assistant': {'preferred': preferred, 'rejected': rejected},
                            'chosen': chosen,
                            'inferred_profile': inferred_profile,
                            'inferred_personality': inferred_personality
                        })

                        # preferred = re.split(r'\d+\.\s*', chatbot_B)[1:][0].strip()
                        # rejected = re.split(r'\d+\.\s*', chatbot_B)[1:][1].strip()
                        final_conversations += 'B: ' + chatbot_B + '\n'

                        A_messages.append(chatbot_B + '\n' + "(Please reply me using very concise and **human-like** language (within 3 sentences)")

                        # B_messages.append({'role': 'assistant', 'content':chatbot_B})
                        # B_messages_instruct.append({'role': 'assistant', 'content':chatbot_B})

                        B_messages.append(chatbot_B)
                        B_messages_instruct.append(chatbot_B)

                        break

                    except:
                        print("Exception")
                        count += 1

                if count == 3:
                    flag = False
                    print("Failed")
                    break
        
            else:
                flag = True
                while count < 3:
                    try:
                        chatbot_A = get_response('gpt', A_messages)
                        chatbot_A = remove_asterisk_text(chatbot_A)
                        final_conversations += 'A: ' + chatbot_A + '\n'

                        A_messages.append(chatbot_A)
                        
                                    
                        infer_persona.append(profile_extract_instruction.format(final_conversations,profile, personality))
                        inferred_persona = get_response('gpt', infer_persona)

                        inferred_profile = inferred_persona.split('Profile: ')[1].split('Personalities: ')[0].strip()
                        inferred_personality = inferred_persona.split('Profile: ')[1].split('Personalities: ')[1].strip()

                        # print(final_conversations)
                        # print("*" * 89)
                        # print(profile)
                        # print("*" * 89)
                        # print(personality)
                        # print("*" * 89)
                        # print(inferred_persona)

                        # B_messages_instruct.append({'role':'user', 'content': chatbot_A + '\n' + hint.format(inferred_persona)})
                        # B_messages.append({'role':'user', 'content': chatbot_A})

                        B_messages_instruct.append(chatbot_A + '\n' + hint.format(inferred_persona)  + "\n" + "(We are doing role-play so please reply me using very concise and **human-like** language (within 3 sentences)")
                        B_messages.append(chatbot_A + "\n" + "(We are doing role-play so please reply me using very concise and **human-like** language (within 3 sentences)")

        
                        # preferred = chat_with_mistral(B_messages_instruct)
                        # rejected = chat_with_mistral(B_messages)
                        first = get_response('gpt', B_messages_instruct)
                        second = get_response('gpt', B_messages)
                        
                        better = get_response('gpt', [determine_better_response.format(profile, personality, first, second)])
                        if better == 'first':
                            preferred = first
                            rejected = second
                        else:
                            preferred = second
                            rejected = first
                            
                        chatbot_B = random.choice([preferred, rejected])

                        if chatbot_B == preferred:
                            chosen = 'preferred'
                        else:
                            chosen = 'rejected'

                        data['conversations'].append({
                            'user': chatbot_A,
                            'assistant': {'preferred': preferred, 'rejected': rejected},
                            'chosen': chosen,
                            'inferred_profile': inferred_profile,
                            'inferred_personality': inferred_personality
                        })

                        # preferred = re.split(r'\d+\.\s*', chatbot_B)[1:][0].strip()
                        # rejected = re.split(r'\d+\.\s*', chatbot_B)[1:][1].strip()
                        final_conversations += 'B: ' + chatbot_B + '\n'

                        A_messages.append(chatbot_B + '\n' + "(Please reply me using very concise and **human-like** language (within 3 sentences)")

                        # B_messages.append({'role':'assistant', 'content':chatbot_B})
                        # B_messages_instruct.append({'role':'assistant', 'content': chatbot_B})

                        B_messages.append(chatbot_B)
                        B_messages_instruct.append(chatbot_B)
                        break
                    except:
                        count += 1

                if count == 3:
                    flag = False
                    print("Failed")
                    break

        if not flag:
            continue
        print(final_conversations)   
        print("*" * 89)
        print(data)
        
        with open('../datasets/conversations.jsonl', 'a') as f:
            f.write(json.dumps(data) + '\n')
        total_count += 1
        if total_count == 3000:
            exit()