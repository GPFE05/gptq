import transformers
import gptq_utils
from quant_utils import create_logger

def gptq_for_model(model, model_path, w_bits=4, nsamples=128, cal_dataset="wikitext2",
                    use_rtn=False, attn_bits=None, expert_bits=None, batch_size=1):
    """
    Apply GPTQ or RTN quantization to a model.

    Parameters
    ----------
    model        : nn.Module   – the model to quantize (on CPU).
    model_path   : str         – HF model id / local path (used by data loaders).
    w_bits       : int         – default quantization bit-width.
    nsamples     : int         – number of calibration samples.
    cal_dataset  : str         – calibration dataset name.
    use_rtn      : bool        – use RTN instead of GPTQ.
    attn_bits    : int | None  – bit-width for attention layers (None ⇒ w_bits).
    expert_bits  : int | None  – bit-width for MoE expert layers (None ⇒ w_bits).
    batch_size   : int         – batch size for calibration forward pass.
    """
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args([])

    logger = create_logger()
    args.seed = 32
    args.model = model_path
    transformers.set_seed(args.seed)
    model.eval()

    # --- Quantization config ---
    args.w_bits = w_bits
    args.attn_bits = attn_bits          # None → falls back to w_bits
    args.expert_bits = expert_bits      # None → falls back to w_bits
    args.batch_size = batch_size
    args.w_clip = True
    args.cal_dataset = cal_dataset
    args.nsamples = nsamples

    args.w_groupsize = -1
    args.w_asym = True
    args.int8_down_proj = True
    args.act_order = True
    args.percdamp = 0.01

    logger.info(
        f'Quantization config: w_bits={w_bits}, '
        f'attn_bits={attn_bits or w_bits}, '
        f'expert_bits={expert_bits or w_bits}, '
        f'batch_size={batch_size}'
    )

    if args.w_bits < 16:
        save_dict = {}        
        if use_rtn is True:
            quantizers = gptq_utils.rtn_fwrd(model, gptq_utils.DEV, args, logger=logger)
            save_dict["w_quantizers"] = quantizers
        else:
            trainloader = gptq_utils.get_loaders(
                args.cal_dataset, nsamples=args.nsamples,
                seed=args.seed, model=args.model,
                seqlen=2048, eval_mode=False
            )
            quantizers = gptq_utils.gptq_fwrd(model, trainloader, gptq_utils.DEV, args, logger=logger)
            save_dict["w_quantizers"] = quantizers
        
    return model