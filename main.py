import argparse
import os
import gc
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from quant_utils import test_ppl


MODEL_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {value}")


def build_parser():
    parser = argparse.ArgumentParser(description="GPTQ quantization entrypoint")
    parser.add_argument("--model-path", default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--model-dtype", default="float16", choices=sorted(MODEL_DTYPES.keys()))
    parser.add_argument("--test-model-ppl", type=str2bool, default=True)
    parser.add_argument("--eval-device", default="auto")
    parser.add_argument("--local-files-only", type=str2bool, default=True)
    parser.add_argument("--trust-remote-code", type=str2bool, default=True)

    parser.add_argument("--wbits", type=int, default=2)
    parser.add_argument("--attn-bits", type=int, default=2)
    parser.add_argument("--expert-bits", type=int, default=2)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--act-order", type=str2bool, default=False)
    parser.add_argument("--force-all-tokens-to-all-experts", type=str2bool, default=False)
    parser.add_argument("--use-last-visible-gpu-for-quant", type=str2bool, default=False)
    parser.add_argument("--cal-dataset", default="c4")
    parser.add_argument("--use-rtn", type=str2bool, default=False)

    parser.add_argument("--save-path", default=None)
    parser.add_argument("--cuda-visible-devices", default="0,1,2")
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--proxy-host", default="http://child-prc.intel.com")
    parser.add_argument("--proxy-port", type=int, default=913)
    parser.add_argument("--log-dir", default="./logs")
    return parser


def print_ppl_results(prefix, ppl_result):
    if isinstance(ppl_result, dict):
        print(f"{prefix} wikitext2 ppl: {ppl_result.get('wikitext2')}")
        print(f"{prefix} c4 ppl: {ppl_result.get('c4')}")
        return
    print(f"{prefix} ppl: {ppl_result}")


def apply_runtime_env(args):
    if args.proxy_host:
        proxy = f"{args.proxy_host}:{args.proxy_port}"
        os.environ["http_proxy"] = proxy
        os.environ["https_proxy"] = proxy
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
    if args.log_dir:
        os.environ["GPTQ_LOG_DIR"] = args.log_dir


def main():
    args = build_parser().parse_args()
    apply_runtime_env(args)

    model_dtype = MODEL_DTYPES[args.model_dtype]
    save_root = Path(__file__).resolve().parent / "models"
    save_root.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )

    if args.test_model_ppl:
        print(f"Loading {args.model_dtype} model with auto device map for evaluation...")
        model_eval = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map="auto",
            trust_remote_code=args.trust_remote_code,
            torch_dtype=model_dtype,
            local_files_only=args.local_files_only,
        )
        model_eval.eval()

        print(f"Evaluating {args.model_dtype} model...")
        ppl_eval = test_ppl(
            model=model_eval,
            tokenizer=tokenizer,
            device=args.eval_device,
        )
        print_ppl_results(f"{args.model_dtype} model", ppl_eval)

        print(f"Releasing {args.model_dtype} evaluation model...")
        del model_eval
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("Loading model to CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="cpu",
        trust_remote_code=args.trust_remote_code,
        torch_dtype=model_dtype,
        local_files_only=args.local_files_only,
    )
    model.eval()

    print("Starting quantization...")
    from gptq import gptq_for_model

    model_quant = gptq_for_model(
        model,
        model_path=args.model_path,
        w_bits=args.wbits,
        nsamples=args.nsamples,
        cal_dataset=args.cal_dataset,
        use_rtn=args.use_rtn,
        attn_bits=args.attn_bits,
        expert_bits=args.expert_bits,
        batch_size=args.batch_size,
        group_size=args.group_size,
        act_order=args.act_order,
        force_all_tokens_to_all_experts=args.force_all_tokens_to_all_experts,
        use_last_visible_gpu_for_quant=args.use_last_visible_gpu_for_quant,
    )

    print(f"save file is {args.save_path}")
    if args.save_path is not None:
        model_quant.save_pretrained(args.save_path)
        tokenizer.save_pretrained(args.save_path)
        print("Quantized model and tokenizer saved successfully.")

    print("Evaluating in-memory quantized model...")
    ppl_quant = test_ppl(model=model_quant, tokenizer=tokenizer, device=args.eval_device)
    print_ppl_results("quant model", ppl_quant)

    print("Executing Garbage Collection...")
    del model
    del model_quant
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.save_path is not None:
        print("Loading quantized model from disk for evaluation...")
        save_lm = AutoModelForCausalLM.from_pretrained(
            args.save_path,
            device_map="cpu",
            trust_remote_code=args.trust_remote_code,
            torch_dtype=model_dtype,
            local_files_only=args.local_files_only,
        )
        ppl_quant_disk = test_ppl(model=save_lm, tokenizer=tokenizer, device=args.eval_device)
        print_ppl_results("quant model from disk", ppl_quant_disk)

        del save_lm
        gc.collect()


if __name__ == "__main__":
    main()