import openai
import itertools
import json
from openai import OpenAI
import random
import evaluate
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import argparse

client = OpenAI(api_key='xxxxxx')

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
model.to('cuda')

argparser = argparse.ArgumentParser()
argparser.add_argument('--output', type=str)
args = argparser.parse_args()


with open('./datasets/{}.jsonl'.format(args.output), 'r') as f:
    pool = f.readlines()


if args.output == 'profile':
    messages = '''Generate 20 different user profiles. Something you can consider includes but not limited to age range, occupation, hobbies, family structure, education background, or any other fun facts. Note that you don't need to include all of these details for each persona. You can use any kinds of combinations and please think about other aspects other than these. 
# #              You should include something that can be elicited from a daily and natural conversations. You should not include too much information about this person's work content and you should not give any description about the user's personality traits. Focus more on some daily, objective facts about the person him/herself. Each profile should contain around 8-10 distinct facts about the person. Here are some examples:
# #              {}
# #              You should only output the personas in plain text format. Separate each user profile with a new line and do not include a number for each profile. IMPORTANT: Try to avoid generating similar profiles and avoid always describing the same type of topic for every persona. You should be creative, diverse and comprehensive!!'''
else:
    messages = '''Generate 20 different descriptions of a user's personality traits. You should include something that can be elicited from a daily and natural conversations. Each description should contain around 8-10 personality traits about the person. Here are some examples:
#              {}
#              You should only output the personality descriptions in plain text format. Separate each description with a new line and do not include a number for each. IMPORTANT: You should not include any other content that is beyond personality traits, such as occupation. Try to avoid generating similar personality descriptions. You should be creative and diverse!!'''

for i in tqdm(range(3000)):
    seed_examples = ''
    for i in random.sample(pool, 5):
        seed_examples += i + '\n'
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature= 0.8,
        messages = [
            {'role':'user', 'content': messages.format(seed_examples)}
            ]
    )
    messages = response.choices[0].message.content
    outputs = messages.split('\n\n')
    
    for output in outputs:
        scores = []
        for item in pool:
            embeddings_output = model.encode([output])
            embeddings_personality = model.encode([item])
            similarity = cosine_similarity(embeddings_output, embeddings_personality)
            scores.append(similarity[0][0])
            
        if max(scores) <= 0.6:
            pool.append(output) 
            with open('./datasets/{}.jsonl'.format(args.output), 'a') as f:
                f.write(json.dumps(output) + '\n')