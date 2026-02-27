import os
from pathlib import Path
import torch
windows_host="http://child-prc.intel.com"
os.environ["http_proxy"] = f"{windows_host}:913"
os.environ["https_proxy"] = f"{windows_host}:913"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"          # two GPUs for MoE eval
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import AutoTokenizer, AutoModelForCausalLM
from quant_utils import test_ppl

# ---------------------------------------------------------------------------
# Quantization configuration
# ---------------------------------------------------------------------------
wbits       = 4          # default bit-width
attn_bits   = 4          # attention layers (None → use wbits)
expert_bits = 4          # MoE expert layers (None → use wbits)
nsamples    = 128
batch_size  = 4          # calibration batch size (> 1 speeds up Hessian collection)

cal_dataset = "wikitext2"
# cal_dataset = "gsm8k"

# ---------------------------------------------------------------------------
# Model selection — switch between dense and MoE by (un)commenting
# ---------------------------------------------------------------------------
# Dense model (single-GPU eval is fine):
# model_path  = "/home/wsg/llm/bitnet-sliding/model_zoo/Qwen3-8B"
# eval_device = "cuda"

# MoE model (requires multi-GPU dispatch for evaluation):
model_path  = "Qwen/Qwen1.5-MoE-A2.7B"
eval_device = "auto"                    # accelerate multi-GPU dispatch
local_files_only = True                  # 不触发下载，仅从本地缓存/本地目录读取

model_name = model_path.split("/")[-1]
save_root = Path(__file__).resolve().parent / "models"
save_path = str(
    save_root /
    f"{model_name}-gptq-w{wbits}-attn{attn_bits}-exp{expert_bits}-{cal_dataset}"
)
save_root.mkdir(parents=True, exist_ok=True)
# model_path = save_path

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
    local_files_only=local_files_only,
)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="cpu",
    local_files_only=local_files_only,
)
model.eval()

# --- Evaluate original (FP16) model ---
ppl = test_ppl(model=model, tokenizer=tokenizer, datasets="wikitext2", device=eval_device)
print(f"fp16 model wikitext2 ppl: {ppl}")

# --- Reload model after FP16 eval, then quantize ---
del model
torch.cuda.empty_cache()
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="cpu",
    local_files_only=local_files_only,
)
model.eval()

# --- GPTQ quantization ---
from gptq import gptq_for_model
model_quant = gptq_for_model(
    model, model_path=model_path,
    w_bits=wbits, nsamples=nsamples,
    cal_dataset=cal_dataset, use_rtn=False,
    attn_bits=attn_bits,
    expert_bits=expert_bits,
    batch_size=batch_size,
)

# --- Save quantized model (while weights are still on CPU) ---
print(f"save file is {save_path}")
if save_path is not None:
    ori_lm = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="cpu",
        local_files_only=local_files_only,
    )
    ori_tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    new_model_weights = model_quant.state_dict()
    ori_lm.load_state_dict(new_model_weights, strict=True)
    ori_lm.save_pretrained(save_path)
    ori_tokenizer.save_pretrained(save_path)

# --- Evaluate quantized model ---
ppl_quant = test_ppl(model=model_quant, tokenizer=tokenizer, datasets="wikitext2", device=eval_device)
print(f"quant model wikitext2 ppl: {ppl_quant}")

# --- Evaluate from disk ---
if save_path is not None:
    save_lm = AutoModelForCausalLM.from_pretrained(
        save_path,
        device_map="cpu",
        local_files_only=local_files_only,
    )
    ppl_quant = test_ppl(model=save_lm, tokenizer=tokenizer, datasets="wikitext2", device=eval_device)
    print(f"quant model from disk wikitext2 ppl: {ppl_quant}")
    



