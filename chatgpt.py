import os
import openai
import pandas as pd
import asyncio
from typing import Any
import time
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

import openai  # for OpenAI API calls
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff


openai.api_key='######'


# p="Durer's father died in 1502, and his mother died in 1513."
# q="How long was his mother ill?"
# ans="4 months"



# prompt="""Passage: Durer's father died in 1502, and his mother died in 1513. Question: How long was his mother ill? Answer: six centuries Response: No*****Passage: Durer's father died in 1502, and his mother died in 1513. Question: How long was his mother ill? Answer: 6 months Response: Yes*****Passage: Durer's father died in 1502, and his mother died in 1513. Question: How long was his mother ill? Answer: 3 minutes Response: No*****Passage: """+p+""" Question: """+q+""" Answer: """+ans+""" Response: """


# response1 = openai.ChatCompletion.create(

# 	  model="gpt-3.5-turbo",
# messages=[
# {"role": "user", "content":prompt}
# ]

# )
# print('PP1: {}'.format(response1['choices'][0]['message']['content']))

async def dispatch_openai_requests(
    messages_list: list[list[dict[str,Any]]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
) -> list[str]:
    """Dispatches requests to OpenAI API asynchronously.
    
    Args:
        messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        model: OpenAI model to use.
        temperature: Temperature to use for the model.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use for the model.
    Returns:
        List of responses from OpenAI API.
    """
    async_responses = [
        openai.ChatCompletion.acreate(
            model=model,
            messages=x,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        for x in messages_list
    ]
    return await asyncio.gather(*async_responses)



@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)





df = pd.read_csv('/DATA/sriparna/Prince/minillm/updated_1000_samples_withART.csv')
   

#for loop

img_id = []
meme_text = []
gen_response = []

responselist=[]

for i in range(len(df)):
    #if i==5:
    #    break
    print('curr sample: {}'.format(i))
    
    img_path = df["imagePath"][i]
    meme_description = df["VLMeme Descripttion"][i]
    meme_caption = df["VLMeme Caption"][i]
    OCR_text = df["Text"][i]
    bias = df["VLMeme Bias"][i]
    stereotype = df["VLMeme stereotype"][i]
    claim = df["VLMeme Claims"][i]
    toxic = df["VLMeme Hate"][i]

    prompt = """A toxic meme has the description: {}. The caption of this toxic meme is this: {}. The following text is written inside the meme: {}. Rationale: Bias: {}, Toxicity: {}, claims: {}, and stereotypes: {}. Write an intervention for the this toxic meme to discourage user posting such memes based on provided knwoledge?""".format(meme_description,meme_caption,OCR_text,bias,toxic,claim,stereotype)
    #prompt = """ The following text is written inside the meme: {}. Write an intervention for the this meme based on all this knowledge?""".format(OCR_text)
    
    

    response1 = completion_with_backoff(model="gpt-3.5-turbo",	messages=[
    {"role": "user", "content":prompt}
    ])
    
    
    response=format(response1['choices'][0]['message']['content'])
    print(str(response))
    print('*************************************^^^^^^^^^^^^^^^^^^^^^^')
    #print(response)
    print('*************************************^^^^^^^^^^^^^^^^^^^^^^')
    
    img_id.append(img_path)
    meme_text.append(OCR_text)
    gen_response.append(response)
    
   


print(gen_response)

data = {}
 
data["img_id"] = img_id
data["meme_text"] = meme_text
data["response"] = gen_response
 
df_generated = pd.DataFrame(data)
    
df_generated.to_csv("response_generated_chatgpt.csv")    
  




























