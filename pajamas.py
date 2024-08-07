#!pip install transformers
import pandas as pd
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

MIN_TRANSFORMERS_VERSION = '4.25.1'

# check transformers version
assert transformers.__version__ >= MIN_TRANSFORMERS_VERSION, f'Please upgrade transformers to version {MIN_TRANSFORMERS_VERSION} or higher.'



df = pd.read_csv('/DATA/sriparna/Prince/minillm/filtered_data.csv')
   

# init
tokenizer = AutoTokenizer.from_pretrained("togethercomputer/RedPajama-INCITE-Instruct-3B-v1")
model = AutoModelForCausalLM.from_pretrained("togethercomputer/RedPajama-INCITE-Instruct-3B-v1", torch_dtype=torch.float16)
model = model.to('cuda:0')



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

    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    input_length = inputs.input_ids.shape[1]
    outputs = model.generate(
        **inputs, max_new_tokens=128, do_sample=True, temperature=0.7, top_p=0.7, top_k=50, return_dict_in_generate=True
    )
    token = outputs.sequences[0, input_length:]
    response = tokenizer.decode(token)
    
    print(response)
    
  
    img_id.append(img_path)
    meme_text.append(OCR_text)
    gen_response.append(response)
    
   

data = {}
 
data["img_id"] = img_id
data["meme_text"] = meme_text
data["response"] = gen_response
 
df_generated = pd.DataFrame(data)
    
df_generated.to_csv("filtered_data_pajamas.csv")    
  
