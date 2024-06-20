# import modules for detector + tokeniser
import transformers
import torch
import torch.nn.functional as F
import random

# setup RADAR detector and tokeniser
device = "cpu" # example: cuda:0
detector = transformers.AutoModelForSequenceClassification.from_pretrained("TrustSafeAI/RADAR-Vicuna-7B")
tokenizer = transformers.AutoTokenizer.from_pretrained("TrustSafeAI/RADAR-Vicuna-7B")
detector.eval()
detector.to(device)
print("detector and tokeniser successfully loaded")

# Helper function to assist with testing
def getRADARoutput(input:str) -> float:
    with torch.no_grad():
        inputs = tokenizer(input, padding=True, truncation=True, max_length=512, return_tensors="pt")
        inputs = {k:v.to(device) for k,v in inputs.items()}
        output_probs = F.log_softmax(detector(**inputs).logits,-1)[:,0].exp().tolist()
        return output_probs
