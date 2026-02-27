import logging
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


def dispatch_model_for_eval(model, gpu_ids=(0, 1), no_split_module_classes=None, dtype=None):
    """
    Dispatch *model* across selected GPUs in-memory using ``accelerate``.
    """
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return model

    from accelerate import infer_auto_device_map, dispatch_model

    if gpu_ids is None:
        gpu_ids = tuple(range(min(2, torch.cuda.device_count())))
    gpu_ids = tuple(gpu_ids)
    if len(gpu_ids) == 0:
        return model

    max_memory = {}
    for gpu_id in gpu_ids:
        total_gib = torch.cuda.get_device_properties(gpu_id).total_memory // (1024 ** 3)
        usable_gib = max(1, int(total_gib) - 1)
        max_memory[gpu_id] = f"{usable_gib}GiB"
    max_memory['cpu'] = '120GiB'

    inferred_dtype = dtype if dtype is not None else next(model.parameters()).dtype
    device_map = infer_auto_device_map(
        model,
        max_memory=max_memory,
        no_split_module_classes=no_split_module_classes,
        dtype=inferred_dtype,
    )
    model = dispatch_model(model, device_map=device_map)
    return model


def test_ppl(model, tokenizer, datasets="wikitext2", text_seq_len=2048, device="cuda"):
    """
    Evaluate perplexity.

    Parameters
    ----------
    device : str
        ``"cuda"``  – single-GPU evaluation (original behaviour).
        ``"auto"``  – multi-GPU dispatch via *accelerate* (for large /
                      MoE models that do not fit on one GPU).
    """
    multi_gpu = (device == "auto") or hasattr(model, 'hf_device_map')

    if datasets == "wikitext2":
        datasets = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    elif datasets == "c4":
        datasets = load_dataset('/home/disk2/wsg/llm/bitnet-sliding/datasets/c4', 
                                data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, 
                                split='validation')
    else:
        raise NotImplementedError("only support wikitext2 and c4")

    text = "\n\n".join(datasets["text"])
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids[0]  # shape: [total_seq_len]

    seq_len = text_seq_len
    stride = seq_len
    max_length = input_ids.size(0)

    # ---- Device placement ----
    if multi_gpu:
        if not hasattr(model, 'hf_device_map'):
            model = dispatch_model_for_eval(model)
        input_device = _first_device_from_map(model)
        if input_device is None:
            input_device = next(model.parameters()).device
    else:
        model = model.to(device)
        input_device = torch.device(device) if isinstance(device, str) else device

    use_cache = model.config.use_cache
    model.config.use_cache = False

    nlls = []
    prev_end_loc = 0
    with torch.no_grad():
        model.eval()
        for begin_loc in tqdm(range(0, max_length - 1, stride)):
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

    ppl = torch.exp(torch.stack(nlls).mean())
    model.config.use_cache = use_cache

    if not multi_gpu:
        model.to("cpu")

    return ppl


def create_logger():
    # 创建一个 Logger 对象
    logger = logging.getLogger("GPTQ")
    logger.setLevel(logging.DEBUG)  # 设置日志级别

    # 创建一个 StreamHandler（输出到控制台）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # 定义日志格式
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)

    # 添加处理器到 logger
    logger.addHandler(console_handler)

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
