import math
import time
import tqdm
import torch
import torch.nn as nn
from quant_utils import cleanup_memory
import quant_utils
import jsonlines

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

DEV = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# ---------------------------------------------------------------------------
# MoE-aware helpers
# ---------------------------------------------------------------------------

# Routing-gate layer names that must NEVER be quantized (stay FP16/BF16)
GATE_LAYER_NAMES = {'mlp.gate', 'mlp.shared_expert_gate'}


def is_gate_layer(name: str) -> bool:
    """Return True if *name* is an MoE routing gate that must stay in full precision."""
    return name in GATE_LAYER_NAMES


def build_sequential_for_layer(layer):
    """
    Dynamically build the sequential execution order for GPTQ quantization.
    Returns a list of name-groups.  Each group shares the same calibration
    forward pass; layers inside a group are then quantized individually.

    Works for both dense (LLaMA / Qwen) and MoE (Qwen2MoE) decoder layers.
    """
    all_linear = quant_utils.find_qlayers(layer)
    all_names = set(all_linear.keys())
    is_moe = any('experts.' in n for n in all_names)

    # --- Attention (common) ---
    attn_qkv = sorted(
        [n for n in all_names
         if n in ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj')]
    )
    attn_o = [n for n in all_names if n == 'self_attn.o_proj']

    sequential = []
    if attn_qkv:
        sequential.append(attn_qkv)
    if attn_o:
        sequential.append(attn_o)

    # --- MLP / MoE ---
    if is_moe:
        # Group ALL expert + shared-expert up/gate projections together,
        # then ALL down projections together.  This way a single calibration
        # forward pass through the decoder layer collects Hessians for every
        # expert simultaneously (each expert hook only fires for its routed
        # tokens).  Gate layers are excluded.
        mlp_up_gate = sorted([
            n for n in all_names
            if not is_gate_layer(n) and 'self_attn' not in n
            and ('up_proj' in n or 'gate_proj' in n)
        ])
        mlp_down = sorted([
            n for n in all_names
            if not is_gate_layer(n) and 'self_attn' not in n
            and 'down_proj' in n
        ])
        if mlp_up_gate:
            sequential.append(mlp_up_gate)
        if mlp_down:
            sequential.append(mlp_down)
    else:
        mlp_up_gate = sorted(
            [n for n in all_names if n in ('mlp.up_proj', 'mlp.gate_proj')]
        )
        mlp_down = [n for n in all_names if n == 'mlp.down_proj']
        if mlp_up_gate:
            sequential.append(mlp_up_gate)
        if mlp_down:
            sequential.append(mlp_down)

    # Append any remaining quantizable linears for robustness
    covered = {n for group in sequential for n in group}
    for name in sorted(all_names):
        if name in covered or is_gate_layer(name):
            continue
        sequential.append([name])

    return sequential


def get_layer_bits(name: str, args) -> int:
    """
    Determine the quantization bit-width for a given layer name.
    Supports mixed-precision: independent bits for attention vs. expert layers.
    """
    if is_gate_layer(name):
        return 16
    if 'lm_head' in name:
        return 16
    # MoE expert layers (including shared expert)
    if 'experts.' in name or 'shared_expert.' in name:
        return getattr(args, 'expert_bits', None) or args.w_bits
    # Attention layers
    if 'self_attn.' in name:
        return getattr(args, 'attn_bits', None) or args.w_bits
    return args.w_bits


def _expand_kwargs_for_batch(kwargs, batch_size):
    """Expand cached layer kwargs to match a larger batch dimension."""
    if batch_size <= 1:
        return kwargs
    expanded = {}
    for k, v in kwargs.items():
        if isinstance(v, torch.Tensor):
            if v.shape[0] == 1:
                expanded[k] = v.expand(batch_size, *v.shape[1:])
            else:
                expanded[k] = v
        elif isinstance(v, tuple):
            expanded[k] = tuple(
                t.expand(batch_size, *t.shape[1:])
                if isinstance(t, torch.Tensor) and t.shape[0] == 1
                else t
                for t in v
            )
        else:
            expanded[k] = v
    return expanded


def _batched_layer_forward(layer, inps, outs, batch_size, layer_kwargs):
    """Forward pass through a decoder layer with configurable batch size."""
    nsamples = inps.shape[0]
    for j in range(0, nsamples, batch_size):
        batch_end = min(j + batch_size, nsamples)
        actual_bs = batch_end - j
        expanded = _expand_kwargs_for_batch(layer_kwargs, actual_bs)
        outs[j:batch_end] = layer(inps[j:batch_end], **expanded)[0]


def llama_down_proj_groupsize(model, groupsize,logger):
    
    assert groupsize > 1, 'groupsize should be greater than 1!'
    
    if model.config.intermediate_size % groupsize == 0:
        logger.info(f'(Act.) Groupsiz = Down_proj Groupsize: {groupsize}')
        return groupsize

    group_num = int(model.config.hidden_size/groupsize)
    assert groupsize*group_num == model.config.hidden_size, 'Invalid groupsize for llama!'

    down_proj_groupsize = model.config.intermediate_size//group_num
    assert down_proj_groupsize*group_num == model.config.intermediate_size, 'Invalid groupsize for down_proj!'
    logger.info(f'(Act.) Groupsize: {groupsize}, Down_proj Groupsize: {down_proj_groupsize}')
    return down_proj_groupsize




class GPTQ:
    def __init__(self, layer,logger):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.token_count = 0
        self.logger = logger

    def add_batch(self, inp, out):
        
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            self.token_count += inp.shape[0] * inp.shape[1]
        else:
            self.token_count += inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False
    ):
        W = self.layer.weight.data.clone()
        W = W.float()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        # if static_groups:
        #     import copy
        #     groups = []
        #     for i in range(0, self.columns, groupsize):
        #         quantizer = copy.deepcopy(self.quantizer)
        #         quantizer.find_params(W[:, i:(i + groupsize)])
        #         groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                # if groupsize != -1:
                #     if not static_groups:
                #         if (i1 + i) % groupsize == 0:
                #             self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)])
                #     else:
                #         idx = i1 + i
                #         if actorder:
                #             idx = perm[idx]
                #         self.quantizer = groups[idx // groupsize]

                q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()

        if actorder:
            Q = Q[:, invperm]

        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if torch.any(torch.isnan(self.layer.weight.data)):
            self.logger.info('NaN in weights')
            import pprint
            # pprint.pprint(self.quantizer.bits, self.quantizer.scale, self.quantizer.zero_point)
            raise ValueError('NaN in weights')

    def free(self):
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()
        cleanup_memory(verbos=False)
        
        
@torch.no_grad()
def gptq_fwrd(model, dataloader, dev, args, logger=None):
    '''
    GPTQ quantization forward pass.
    Supports both dense and MoE (Qwen2MoE) architectures with:
      - Mixed-precision (separate attn_bits / expert_bits)
      - MoE gate exemption
      - Configurable calibration batch_size
      - Per-layer statistics logging (bit-width & calibration token count)
    '''
    logger.info('-----GPTQ Quantization-----')

    batch_size = getattr(args, 'batch_size', 1)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, 2048, model.config.hidden_size), dtype=dtype, device=dev
    )

    # Generic Catcher — captures all kwargs regardless of model architecture
    cache = {'i': 0}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['kwargs'] = kwargs
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    layer_kwargs = cache.get('kwargs', {})          # reused for every layer

    quantizers = {}

    for i in range(len(layers)):
        logger.info(f'\n=== Layer {i} ===')
        layer = layers[i].to(dev)
        full = quant_utils.find_qlayers(layer)

        # Dynamically build the sequential execution order (dense or MoE)
        sequential = build_sequential_for_layer(layer)

        for names in sequential:
            subset = {n: full[n] for n in names if n in full}
            if not subset:
                continue

            gptq = {}
            for name in subset:
                layer_weight_bits = get_layer_bits(name, args)
                layer_weight_sym = not(args.w_asym)

                if layer_weight_bits >= 16:
                    logger.info(f'  {name}: skipped (bits={layer_weight_bits})')
                    continue

                gptq[name] = GPTQ(subset[name], logger=logger)
                gptq[name].quantizer = quant_utils.WeightQuantizer()
                gptq[name].quantizer.configure(
                    layer_weight_bits, perchannel=True,
                    sym=layer_weight_sym, mse=args.w_clip
                )

            if not gptq:
                continue

            # --- Register hooks to accumulate Hessian ---
            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in gptq:
                handles.append(subset[name].register_forward_hook(add_batch(name)))

            # Batched calibration forward (configurable batch_size)
            _batched_layer_forward(layer, inps, outs, batch_size, layer_kwargs)

            for h in handles:
                h.remove()

            # --- Quantize each layer & log statistics ---
            for name in gptq:
                layer_weight_bits = get_layer_bits(name, args)
                gptq[name].fasterquant(
                    percdamp=args.percdamp,
                    groupsize=args.w_groupsize,
                    actorder=args.act_order,
                    static_groups=False,
                )
                logger.info(
                    f'  {name}: bits={layer_weight_bits}, '
                    f'calibration_tokens={gptq[name].token_count}'
                )
                quantizers['model.layers.%d.%s' % (i, name)] = gptq[name].quantizer
                gptq[name].free()

        # Final forward to propagate updated weights to next layer
        _batched_layer_forward(layer, inps, outs, batch_size, layer_kwargs)

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    cleanup_memory(verbos=True)
    logger.info('-----GPTQ Quantization Done-----\n')
    return quantizers

@torch.no_grad()
def rtn_fwrd(model, dev, args, logger):
    '''
    RTN (Round-to-Nearest) quantization.
    Supports both dense and MoE architectures with mixed-precision.
    MoE routing gates are automatically excluded.
    '''
    logger.info('-----RTN Quantization-----')
    assert args.w_groupsize == -1, "Groupsize not supported in RTN!"
    layers = model.model.layers
    torch.cuda.empty_cache()

    quantizers = {}

    for i in tqdm.tqdm(range(len(layers)), desc="(RtN Quant.) Layers"):
        layer = layers[i].to(dev)

        subset = quant_utils.find_qlayers(layer)

        for name in subset:
            # Skip MoE routing gates — must stay FP16/BF16
            if is_gate_layer(name):
                logger.info(f'  Layer {i} {name}: skipped (routing gate)')
                continue

            layer_weight_bits = get_layer_bits(name, args)

            if layer_weight_bits >= 16:
                logger.info(f'  Layer {i} {name}: skipped (bits={layer_weight_bits})')
                continue
            
            W = subset[name].weight.data
            quantizer = quant_utils.WeightQuantizer()
            quantizer.configure(
                layer_weight_bits, perchannel=True, sym=not(args.w_asym), mse=args.w_clip
            )
            quantizer.find_params(W)

            subset[name].weight.data = quantizer.quantize(W).to(
                next(iter(layer.parameters())).dtype)
            quantizers['model.layers.%d.%s' % (i, name)] = quantizer.cpu()

            # Log statistics
            logger.info(
                f'  Layer {i} {name}: bits={layer_weight_bits}, '
                f'weight_params={W.numel()}'
            )

        layers[i] = layer.cpu()
        torch.cuda.empty_cache()
        del layer
            
    cleanup_memory(verbos=True)
    logger.info('-----RTN Quantization Done-----\n')

    return quantizers


import datasets
import random
import transformers

def get_wikitext2(nsamples, seed, seqlen, model, hf_token, eval_mode=False):
    
    if hf_token is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False, use_auth_token=hf_token)
        
    if eval_mode:
        testdata = datasets.load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='test')
        testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
        return testenc
    else:
        traindata = datasets.load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='train')
        trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')    
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader

def get_c4_new(nsamples, seed, seqlen, model, hf_token=None, eval_mode=False):

    if hf_token is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False, use_auth_token=hf_token)

    if eval_mode:
        valdata = datasets.load_dataset(
        'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')
        valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
        valenc = valenc.input_ids[:, :(256 * seqlen)]
        class TokenizerWrapper:
            def __init__(self, input_ids):
                self.input_ids = input_ids
        valenc = TokenizerWrapper(valenc)
        return valenc
    else:
        traindata = datasets.load_dataset(
            'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
        
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
                if trainenc.input_ids.shape[1] >= seqlen:
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader

    


def get_ptb_new(nsamples, seed, seqlen, model, hf_token, eval_mode=False):
    
        
    if hf_token is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False, use_auth_token=hf_token)
    
    if eval_mode:
        testdata = datasets.load_dataset('ptb_text_only', 'penn_treebank', split='test')
        testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')
        return testenc
    else:
        traindata = datasets.load_dataset('ptb_text_only', 'penn_treebank', split='train')
        trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader


def get_hard_data(nsamples, seed, seqlen, model,data_path,data_name,from_start=True):
    print(f"get {data_name} from_start:{from_start}")
    # import ipdb;ipdb.set_trace()

    with jsonlines.open(data_path) as f:
        traindata = list(f)


    tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False)

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            line_text = traindata[i]["prompt"] + traindata[i]["output"]
            trainenc = tokenizer(line_text, return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        if from_start is True:
            i = 0
        else:
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader


def get_loaders(
    name, nsamples=128, seed=0, seqlen=2048, model='', hf_token=None, eval_mode=False
):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model, hf_token, eval_mode)
    if 'ptb' in name:
        return get_ptb_new(nsamples, seed, seqlen, model, hf_token, eval_mode)
    if 'c4' in name:
        return get_c4_new(nsamples, seed, seqlen, model, hf_token, eval_mode)
    if 'gsm8k' in name:
        # return get_hard_data(nsamples, seed, seqlen, model,data_path="/home/wsg/llm/bitnet-sliding/gsm8k/gsm8k_train_output_qwen3-8B.jsonl",data_name=name)
        return get_hard_data(nsamples, seed, seqlen, model,data_path="/home/wsg/llm/bitnet-sliding/gsm8k/gsm8k_train_output_qwen3-8B-fewshot-0-V2.jsonl",data_name=name)