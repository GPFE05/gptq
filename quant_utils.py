import logging
import gc
import os
from datetime import datetime
from pathlib import Path
import torch
from tqdm import tqdm
from datasets import load_dataset


def _first_device_from_map(model):
    if not hasattr(model, 'hf_device_map'):
        return None

    for _, value in model.hf_device_map.items():
        if isinstance(value, int):
            return torch.device(f'cuda:{value}')
        if isinstance(value, str) and value.startswith('cuda'):
            return torch.device(value)
    return None


def _device_map_has_cuda(model):
    if not hasattr(model, 'hf_device_map'):
        return False

    for _, value in model.hf_device_map.items():
        if isinstance(value, int):
            return True
        if isinstance(value, str) and value.startswith('cuda'):
            return True
    return False


def get_no_split_module_classes(net):
    """Return decoder-layer classes that must not be split across devices."""
    net_name = (net or "").lower()
    if "qwen3" in net_name and "moe" in net_name:
        return ["Qwen3MoeDecoderLayer"]
    return []


def get_max_memory_map(usage_ratio=0.95):
    """Build a per-GPU max-memory map from current free VRAM."""
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return {}

    max_memory = {}
    for gpu_id in range(torch.cuda.device_count()):
        free_bytes, _ = torch.cuda.mem_get_info(gpu_id)
        usable_bytes = int(free_bytes * usage_ratio)
        usable_gib = max(1, usable_bytes // (1024 ** 3))
        max_memory[gpu_id] = f"{usable_gib}GiB"
    return max_memory


def dispatch_model_for_eval(model, gpu_ids=None, no_split_module_classes=None, dtype=None):
    """
    Dispatch *model* across selected GPUs in-memory using ``accelerate``.
    """
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return model

    from accelerate import infer_auto_device_map, dispatch_model
    from accelerate.utils import get_balanced_memory

    if gpu_ids is None:
        gpu_ids = tuple(range(torch.cuda.device_count()))
    gpu_ids = tuple(gpu_ids)
    if len(gpu_ids) == 0:
        return model

    # Clear stale allocator state before planning placement.
    gc.collect()
    torch.cuda.empty_cache()

    if no_split_module_classes is None:
        model_name = getattr(getattr(model, "config", None), "_name_or_path", "")
        no_split_module_classes = get_no_split_module_classes(model_name)

    # Auto-add Qwen3 MoE decoder layer if present to avoid accidental split.
    if "Qwen3MoeDecoderLayer" not in no_split_module_classes:
        for module in model.modules():
            if module.__class__.__name__ == "Qwen3MoeDecoderLayer":
                no_split_module_classes = [*no_split_module_classes, "Qwen3MoeDecoderLayer"]
                break

    inferred_dtype = dtype if dtype is not None else next(model.parameters()).dtype
    selected_memory = get_max_memory_map(0.95)
    if gpu_ids:
        selected_memory = {k: v for k, v in selected_memory.items() if k in gpu_ids}

    balanced_mem = get_balanced_memory(
        model,
        max_memory=selected_memory,
        no_split_module_classes=no_split_module_classes,
    )
    logging.getLogger("GPTQ").info(f"Auto-balancing memory: {balanced_mem}")

    device_map = infer_auto_device_map(
        model,
        max_memory=balanced_mem,
        no_split_module_classes=no_split_module_classes,
        dtype=inferred_dtype,
    )
    model = dispatch_model(model, device_map=device_map)
    return model


def _load_eval_text(dataset_name):
    dataset_name = dataset_name.lower()
    if dataset_name == "wikitext2":
        dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    elif dataset_name == "c4":
        from gptq_utils import get_c4
        _, dataset = get_c4(nsamples=0, seed=0, seqlen=0, model="", trust_remote_code=False)
    else:
        raise NotImplementedError("only support wikitext2 and c4")

    return "\n\n".join(dataset["text"])


def test_ppl(model, tokenizer, text_seq_len=2048, device="cuda"):
    """
    Evaluate perplexity.

    Parameters
    ----------
    device : str
        ``"cuda"``  – single-GPU evaluation (original behaviour).
        ``"auto"``  – multi-GPU dispatch via *accelerate* (for large /
                      MoE models that do not fit on one GPU).
    """
    eval_datasets = ["wikitext2", "c4"]
    multi_gpu = (device == "auto") or hasattr(model, 'hf_device_map')

    # ---- Device placement ----
    if multi_gpu:
        # `device_map="cpu"` also creates hf_device_map; do not mistake it as GPU-dispatched.
        need_dispatch = (not hasattr(model, 'hf_device_map')) or (
            device == "auto" and not _device_map_has_cuda(model)
        )
        if need_dispatch:
            model = dispatch_model_for_eval(model)
        input_device = _first_device_from_map(model)
        if input_device is None:
            input_device = next(model.parameters()).device
        if device == "auto" and input_device.type != "cuda":
            print("[test_ppl] warning: auto device dispatch did not place model on CUDA; evaluation will run on CPU.")
    else:
        model = model.to(device)
        input_device = torch.device(device) if isinstance(device, str) else device

    use_cache = model.config.use_cache
    model.config.use_cache = False

    results = {}
    with torch.no_grad():
        model.eval()
        for dataset_name in eval_datasets:
            text = _load_eval_text(dataset_name)
            encodings = tokenizer(text, return_tensors="pt")
            input_ids = encodings.input_ids[0]  # shape: [total_seq_len]

            seq_len = text_seq_len
            stride = seq_len
            max_length = input_ids.size(0)

            nlls = []
            prev_end_loc = 0
            for begin_loc in tqdm(range(0, max_length - 1, stride), desc=f"ppl:{dataset_name}"):
                end_loc = min(begin_loc + seq_len, max_length)
                trg_len = end_loc - prev_end_loc

                input_chunk = input_ids[begin_loc:end_loc].unsqueeze(0).to(input_device)
                target_ids = input_chunk.clone()
                target_ids[:, :-trg_len] = -100

                outputs = model(input_chunk, labels=target_ids)
                neg_log_likelihood = outputs.loss
                nlls.append(neg_log_likelihood.cpu())   # always collect on CPU

                prev_end_loc = end_loc
                if end_loc == max_length:
                    break

            results[dataset_name] = torch.exp(torch.stack(nlls).mean())

    model.config.use_cache = use_cache

    if not multi_gpu:
        model.to("cpu")

    if len(results) == 1:
        return next(iter(results.values()))
    return results


def create_logger():
    logger = logging.getLogger("GPTQ")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    has_console = any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers)
    has_file = any(isinstance(handler, logging.FileHandler) for handler in logger.handlers)

    if not has_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if not has_file:
        log_dir = os.environ.get("GPTQ_LOG_DIR", "./logs")
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        log_file = log_path / f"gptq-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info(f"Log file: {log_file.resolve()}")

    return logger



def cleanup_memory(verbos=True,logger=None) -> None:
    """Run GC and clear GPU memory."""
    import gc
    import inspect
    caller_name = ''
    try:
        caller_name = f' (from {inspect.stack()[1].function})'
    except (ValueError, KeyError):
        pass

    def total_reserved_mem() -> int:
        return sum(torch.cuda.memory_reserved(device=i) for i in range(torch.cuda.device_count()))

    memory_before = total_reserved_mem()

    # gc.collect and empty cache are necessary to clean up GPU memory if the model was distributed
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_after = total_reserved_mem()
        if verbos and logger:
            logger.info(
                f"GPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> {memory_after / (1024 ** 3):.2f} GB"
                f" ({(memory_after - memory_before) / (1024 ** 3):.2f} GB)"
            )


def get_minq_maxq(bits, sym):
    if sym:
        maxq = torch.tensor(2**(bits-1)-1)
        minq = -maxq -1
    else:
        maxq = torch.tensor(2**bits - 1)
        minq = 0

    return minq, maxq

def asym_quant(x, scale, zero, maxq):
    scale = scale.to(x.device)
    zero = zero.to(x.device)
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return q, scale, zero

def asym_dequant(q, scale, zero):
    return scale * (q - zero)

def asym_quant_dequant(x, scale, zero, maxq):
    return asym_dequant(*asym_quant(x, scale, zero, maxq))

def sym_quant(x, scale, maxq):
    scale = scale.to(x.device)
    q = torch.clamp(torch.round(x / scale), -(maxq+1), maxq)
    return q, scale
def sym_dequant(q, scale):
    return scale * q

def sym_quant_dequant(x, scale, maxq):
    return sym_dequant(*sym_quant(x, scale, maxq))


def two_compl(x, bits: int):
    return torch.where(x < 0, 2 ** bits + x, x)

# Pack the int tensor. Each uint8 stores two int4 value.
def pack_i4(q):
    assert torch.is_signed(q), 'The tensor to be packed should be signed int'
    minq, maxq = get_minq_maxq(4, True)
    assert torch.all(torch.logical_and(q >= minq, q <= maxq))

    q_i8 = two_compl(q.to(dtype=torch.int8), 4).to(torch.uint8)
    q_i4 = q_i8[:, 0::2] | (q_i8[:, 1::2] << 4)
    return q_i4


# Unpack the quantized int4 tensor (stored in uint8) into int32 tensor.
def unpack_i4(x: torch.Tensor):
    assert x.dtype == torch.uint8, 'The tensor to be unpacked should be stored in uint8'

    out_shape = list(x.shape)
    out_shape[-1] *= 2  # Each uint8 packs two numbers

    # Low 4 bits
    x0 = (x & 0x0f).to(torch.int8)
    x0[x0>=8] -= 16
    x0 = x0.view(-1, x0.shape[-1])

    # High 4 bits
    x1 = ((x & 0xf0) >> 4).to(torch.int8)
    x1[x1>=8] -= 16
    x1 = x1.view(-1, x1.shape[-1])

    out = torch.empty(out_shape, device=x.device, dtype=torch.int32)
    out = out.view(-1, out.shape[-1])
    # Interleaving
    out[:, 0::2] = x0
    out[:, 1::2] = x1

    return out.view(out_shape)


class WeightQuantizer(torch.nn.Module):
    '''From GPTQ Repo'''

    def __init__(self, shape=1):
        super(WeightQuantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))

    def configure(
        self,
        bits, perchannel=False, sym=True,
        mse=False, norm=2.4, grid=100, maxshrink=.8,
    ):
        self.bits = bits
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        if sym:
            self.maxq = torch.tensor(2**(bits-1)-1)
        else:
            self.maxq = torch.tensor(2**bits - 1)

    def find_params(self, x):
        if self.bits == 16:
            return
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            x = x.flatten(1)
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax).clamp(min=1e-5)
            self.scale = xmax / self.maxq
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin).clamp(min=1e-5) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float('inf'), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax

                if self.sym:
                    scale1 = xmax1 / self.maxq
                    zero1 = torch.zeros_like(scale1)
                    q = sym_quant_dequant(x, scale1.unsqueeze(1), self.maxq)
                else:

                    scale1 = (xmax1 - xmin1) / self.maxq
                    zero1 = torch.round(-xmin1 / scale1)
                    q = asym_quant_dequant(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)

                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:

            tmp = shape[0]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        shape = [-1] + [1] * (len(shape) - 1)
        self.scale = self.scale.reshape(shape)
        self.zero = self.zero.reshape(shape)
        return

    # TODO: This should be better refactored into `forward`, which applies quantize and dequantize. A new method `quantize` should be added (if needed) to return the quantized integers and scales, like in ActQuantizer.
    def quantize(self, x):
        x_dtype = x.dtype
        if self.ready() and self.bits < 16:
            if self.sym:
                return sym_quant_dequant(x, self.scale, self.maxq).to(x_dtype)
            return asym_quant_dequant(x, self.scale, self.zero, self.maxq).to(x_dtype)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)



def find_qlayers(module, layers=[torch.nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_qlayers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res
