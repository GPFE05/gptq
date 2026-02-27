import os
import gc
from pathlib import Path
import torch

windows_host = "http://child-prc.intel.com"
os.environ["http_proxy"] = f"{windows_host}:913"
os.environ["https_proxy"] = f"{windows_host}:913"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # two GPUs for MoE eval
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import AutoTokenizer, AutoModelForCausalLM
from quant_utils import test_ppl

# ---------------------------------------------------------------------------
# Quantization configuration
# ---------------------------------------------------------------------------
wbits = 4  # default bit-width
attn_bits = 4  # attention layers (None → use wbits)
expert_bits = 2  # MoE expert layers (None → use wbits)
nsamples = 128
batch_size = 80  # calibration batch size (> 1 speeds up Hessian collection)
group_size = 128  # enabled by default; set -1 to disable
act_order = False

cal_dataset = "wikitext2"
# cal_dataset = "gsm8k"

# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------
model_path = "/home/kyyx/zyc/myomni/models/Qwen1.5-MoE-A2.7B"
eval_device = "auto"  # accelerate multi-GPU dispatch
local_files_only = True  # 不触发下载，仅从本地缓存/本地目录读取

model_name = model_path.split("/")[-1]
save_root = Path(__file__).resolve().parent / "models"
# save_path = str(
#     save_root /
#     f"{model_name}-gptq-w{wbits}-attn{attn_bits}-exp{expert_bits}-{cal_dataset}"
# )
save_path = None
save_root.mkdir(parents=True, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
    local_files_only=local_files_only,
)

# 1. 首次加载模型到 CPU 内存 (默认占用约 57GB FP32 空间)
print("Loading model to CPU...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="cpu",
    local_files_only=local_files_only,
)
model.eval()

# --- GPTQ quantization ---
print("Starting quantization...")
from gptq import gptq_for_model

model_quant = gptq_for_model(
    model, model_path=model_path,
    w_bits=wbits, nsamples=nsamples,
    cal_dataset=cal_dataset, use_rtn=False,
    attn_bits=attn_bits,
    expert_bits=expert_bits,
    batch_size=batch_size,
    group_size=group_size,
    act_order=act_order,
)

# --- Save quantized model ---
print(f"save file is {save_path}")
if save_path is not None:
    # 2. 直接利用量化后的对象保存模型，绝不重新拉起 ori_lm
    model_quant.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print("Quantized model and tokenizer saved successfully.")

# --- Evaluate quantized model (In-memory) ---
print("Evaluating in-memory quantized model...")
ppl_quant = test_ppl(model=model_quant, tokenizer=tokenizer, datasets="wikitext2", device=eval_device)
print(f"quant model wikitext2 ppl: {ppl_quant}")

# ---------------------------------------------------------------------------
# 3. 核心内存清理区：彻底摧毁庞大的 FP32 模型和量化中间态
# ---------------------------------------------------------------------------
print("Executing Garbage Collection...")
del model
del model_quant
gc.collect()  # 强制 Python 回收内存
torch.cuda.empty_cache()  # 清空 PyTorch 在 GPU 端预留的显存池

# --- Evaluate from disk ---
if save_path is not None:
    print("Loading quantized model from disk for evaluation...")
    # 此时内存已经被腾空，再次加载量化后的模型毫无压力
    save_lm = AutoModelForCausalLM.from_pretrained(
        save_path,
        device_map="cpu",
        local_files_only=local_files_only,
    )
    ppl_quant_disk = test_ppl(model=save_lm, tokenizer=tokenizer, datasets="wikitext2", device=eval_device)
    print(f"quant model from disk wikitext2 ppl: {ppl_quant_disk}")

    # 良好的代码习惯：脚本结尾也清理干净
    del save_lm
    gc.collect()