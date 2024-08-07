import torch
from transformers import pipeline
import pandas as pd
from transformers import pipeline
import torch
# device = torch.device("cpu")
from transformers import GPTJForCausalLM, AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer,GPTJForCausalLM



from transformers import GPTNeoForCausalLM, GPT2Tokenizer


device = torch.device("cuda:0")


#tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-12b", padding_side="left")
tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v2-3b", padding_side="left")
#model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-12b", torch_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v2-3b", torch_dtype=torch.bfloat16)


df = pd.read_csv('/DATA/sriparna/Prince/minillm/updated_1000_samples.csv')





gen = pipeline("text-generation",model=model,tokenizer=tokenizer,torch_dtype=torch.bfloat16, trust_remote_code=True,device=device,temperature=0.8,max_length=700,return_full_text=False,eos_token_id=[int(tokenizer.convert_tokens_to_ids('###'))])



    


img_id = []
meme_text = []
gen_response = []

responselist=[]

for i in range(len(df)):
    #if i==5:
    #    break
    #print('curr sample: {}'.format(i))
    
    img_path = df["imagePath"][i]
    meme_description = df["VLMeme Descripttion"][i]
    meme_caption = df["VLMeme Caption"][i]
    OCR_text = df["Text"][i]
    bias = df["VLMeme Bias"][i]
    stereotype = df["VLMeme stereotype"][i]
    claim = df["VLMeme Claims"][i]
    toxic = df["VLMeme Hate"][i]
    
    #prompt = """A toxic meme has the description: {}. The caption of this toxic meme is this: {}. The following text is written inside the meme: {}. Rationale: Bias: {}, Toxicity: {}, claims: {}, and stereotypes: {}. Write an intervention for the this toxic meme to discourage user posting such memes based on provided knwoledge?""".format(meme_description,meme_caption,OCR_text,bias,toxic,claim,stereotype)
    prompt = """ The following text is written inside the meme: {}. Write an intervention for the this meme based on all this knowledge?""".format(OCR_text)

    print('**************$$$$$$$$$$####################')
    response=gen(prompt)
    print(response[0]["generated_text"])
    
    response = response[0]["generated_text"]
    #response=response[0]['generated_text'].split(' ')[0]
    print('**************$$$$$$$$$$####################')
    
    #print('ANS: {}'.format(response))
    
    img_id.append(img_path)
    meme_text.append(OCR_text)
    gen_response.append(response)
    
    
    

data = {}
 
data["img_id"] = img_id
data["meme_text"] = meme_text
data["response"] = gen_response
 
df_generated = pd.DataFrame(data)
    
df_generated.to_csv("response_generated.csv")    
  

    
    
