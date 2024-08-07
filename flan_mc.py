from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from transformers import pipeline
import torch
# device = torch.device("cpu")
from transformers import GPTJForCausalLM, AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer,GPTJForCausalLM
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration




from transformers import GPTNeoForCausalLM, GPT2Tokenizer

#torch.cuda.set_device(1)
device = torch.device("cuda:0")  # Set the device


model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
model = model.to(device) 
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

df = pd.read_csv('/DATA/sriparna/Prince/minillm/filtered_data.csv')



responselist=[]
end_sequence="####"

labels=[]


img_id = []
meme_text = []
gen_response = []

for i in range(len(df)):


    print("curr: ",i)
   
    img_path = df["imagePath"][i]
    meme_description = df["VLMeme Descripttion"][i]
    meme_caption = df["VLMeme Caption"][i]
    OCR_text = df["Text"][i]
    bias = df["VLMeme Bias"][i]
    stereotype = df["VLMeme stereotype"][i]
    claim = df["VLMeme Claims"][i]
    toxic = df["VLMeme Hate"][i]
    prompt = """This is a toxic meme with the description: {}. The caption of this toxic meme is this: {}. The following text is written inside the meme: {}. Rationale: Bias: {}, Toxicity: {}, claims: {}, and stereotypes: {}. Write an intervention for the this meme based on all this knowledge?""".format(meme_description,meme_caption,OCR_text,bias,toxic,claim,stereotype)

    print('***********PROMPT**************')
    print(prompt)
    # try:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model.generate(**inputs,max_length=len(prompt)+100,temperature=0.5)
    response=tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    print('$$$$$$$$$$$$$$$$$$$^^^^^^^^^^********************')
    print(response)
    print('$$$$$$$$$$$$$$$$$$$^^^^^^^^^^********************')
    
    img_id.append(img_path)
    meme_text.append(OCR_text)
    gen_response.append(response)
    
    
    
    
data = {}
 
data["img_id"] = img_id
data["meme_text"] = meme_text
data["response"] = gen_response
 
df_generated = pd.DataFrame(data)
    
df_generated.to_csv("filtered_data_flan.csv")    
  

