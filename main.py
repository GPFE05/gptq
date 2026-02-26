import os
windows_host="http://child-prc.intel.com"
os.environ["http_proxy"] = f"{windows_host}:913"
os.environ["https_proxy"] = f"{windows_host}:913"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import AutoTokenizer, AutoModelForCausalLM
from quant_utils import test_ppl

wbits = 4
nsamples = 128

# cal_dataset = "wikitext2"
cal_dataset = "gsm8k"

save_path = f"/home/wsg/llm/bitnet-sliding/GPTQ/log_model/Qwen3-gptq-w{wbits}-{cal_dataset}"
model_path = "/home/wsg/llm/bitnet-sliding/model_zoo/Qwen3-8B"
# model_path = save_path

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu")
model.eval()


ppl = test_ppl(model=model,tokenizer=tokenizer,datasets="wikitext2")
print(f"fp16 model wikitext2 ppl: {ppl}")


from gptq import gptq_for_model
model_quant = gptq_for_model(model,model_path=model_path,w_bits=wbits,nsamples=nsamples,cal_dataset=cal_dataset,use_rtn=False)

# import ipdb;ipdb.set_trace()

ppl_quant = test_ppl(model=model_quant,tokenizer=tokenizer,datasets="wikitext2")
print(f"quant model wikitext2 ppl: {ppl_quant}")


print(f"save file is {save_path}")
if save_path is not None:
    ori_lm = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu")
    ori_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    new_model_weights = model_quant.state_dict()
    ori_lm.load_state_dict(new_model_weights,strict=True)
    ori_lm.save_pretrained(save_path)
    ori_tokenizer.save_pretrained(save_path)

    save_lm = AutoModelForCausalLM.from_pretrained(save_path, device_map="cpu")
    ppl_quant = test_ppl(model=save_lm,tokenizer=tokenizer,datasets="wikitext2")
    print(f"quant model from disk wikitext2 ppl: {ppl_quant}")
    



