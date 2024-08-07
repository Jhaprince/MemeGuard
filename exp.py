import minillm.executor as minillm
import pandas as pd

import torch
import os

device = torch.device("cuda:0")  # Set the device



llm, llm_config = minillm.load_llm('llama-7b-4bit', 'llama-7b-4bit.pt')



df = pd.read_csv('/DATA/sriparna/Prince/minillm/filtered_data.csv')



responselist=[]
end_sequence="####"

labels=[]


img_id = []
meme_text = []
gen_response = []

responselist=[]

for i in range(len(df)):
    print('curr sample: {}'.format(i))
    img_path = df["imagePath"][i]
    meme_description = df["VLMeme Descripttion"][i]
    meme_caption = df["VLMeme Caption"][i]
    OCR_text = df["Text"][i]
    bias = df["VLMeme Bias"][i]
    stereotype = df["VLMeme stereotype"][i]
    claim = df["VLMeme Claims"][i]
    toxic = df["VLMeme Hate"][i]
    
    prompt = """This is a toxic meme with the description: {}. The caption of this toxic meme is this: {}. The following text is written inside the meme: {}. Rationale: Bias: {}, Toxicity: {}, claims: {}, and stereotypes: {}. Write an intervention for the this meme based on all this knowledge?""".format(meme_description,meme_caption,OCR_text,bias,toxic,claim,stereotype)
    
    
    output = minillm.generate(
	    llm, 
	    llm_config, 
	    prompt=prompt, 
	    min_length=10, 
	    max_length=100, 
	    top_p=0.95, 
	    top_k=50,
	    temperature=0.3,
	    # eos_token_id=[int(tokenizer.convert_tokens_to_ids(end_sequence))]
	)
    
    
    #print(output)
    last_response_index = output.rfind("Response:")
    response = output[last_response_index+len("Response:"):].strip()
    
    print('ANS: {}'.format(response))
    
       
    img_id.append(img_path)
    meme_text.append(OCR_text)
    gen_response.append(response)
    
    
    
   
data = {}
 
data["img_id"] = img_id
data["meme_text"] = meme_text
data["response"] = gen_response
 
df_generated = pd.DataFrame(data)
    
df_generated.to_csv("filtered_data_llama.csv")    
  



	
