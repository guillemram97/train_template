from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers import BitsAndBytesConfig
import pdb
import torch




model_name = "google/gemma-2-27b-it" #"meta-llama/Llama-3.1-70B" #"mistralai/Mixtral-8x7B-v0.1" #"google/gemma-2-27b-it" #"mistralai/Mistral-7B-v0.3"#"mistralai/Mixtral-8x7B-v0.1" # "tiiuae/falcon-40b-instruct"

save_dict = {
    "questions":[],
    "completions":[],
    "targets":[],
}
# 

use_4bit_quantization = True
use_8bit_quantization = False
bnb_config = None
quant_storage_dtype = None
if use_4bit_quantization:
    bnb_4bit_quant_type="nf4"
    # bnb_4bit_use_double_quant=True
    bnb_4bit_compute_dtype="bfloat16"
    bnb_4bit_quant_storage_dtype="bfloat16"
    use_nested_quant = False

    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    quant_storage_dtype = getattr(torch, bnb_4bit_quant_storage_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit_quantization,
        load_in_8bit=use_8bit_quantization,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
        bnb_4bit_quant_storage=quant_storage_dtype,
    )

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=bnb_config, torch_dtype=torch.bfloat16)
pdb.set_trace()

quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
input_ids = tokenizer("Hello, how are you?", return_tensors="pt").input_ids

pdb.set_trace()