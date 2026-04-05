import argparse
import os
import gc
import json
import logging
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
    parser.add_argument("--tasks", default="")
    parser.add_argument("--lm-eval-batch-size", default="auto")
    parser.add_argument("--gen-kwargs", default=None)
    parser.add_argument("--output-dir", default="./log_model")
    return parser


def parse_gen_kwargs(gen_kwargs):
    if gen_kwargs is None:
        return None
    if isinstance(gen_kwargs, dict):
        return gen_kwargs
    if isinstance(gen_kwargs, str):
        value = gen_kwargs.strip()
        if value == "":
            return None
        # Prefer JSON when supplied; otherwise pass through raw string to lm_eval.
        if value.startswith("{") and value.endswith("}"):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value
    return gen_kwargs


def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("gptq_main")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(os.path.join(log_dir, "main.log"), encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


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
    logger = setup_logger(args.log_dir)

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

    results = {}
    if args.tasks != "":
        task_list = args.tasks.split(",") if isinstance(args.tasks, str) else args.tasks
        task_list = [task.strip() for task in task_list if str(task).strip()]

        if task_list:
            try:
                import lm_eval
                from lm_eval.models.huggingface import HFLM
            except ImportError as exc:
                raise RuntimeError(
                    "lm_eval is required for --tasks evaluation. Please install it first."
                ) from exc

            try:
                task_manager = lm_eval.tasks.TaskManager(
                    include_path="./datasets_local/lm_eval_configs/tasks",
                    include_defaults=True,
                )
            except Exception:
                task_manager = lm_eval.tasks.TaskManager(include_defaults=True)

            eval_batch_size = args.lm_eval_batch_size
            if eval_batch_size is None or str(eval_batch_size).strip() == "":
                eval_batch_size = "auto"

            print(f"Initializing HFLM with batch_size={eval_batch_size}...")
            hflm = HFLM(pretrained=model_quant, tokenizer=tokenizer, batch_size=eval_batch_size)

            t_results = lm_eval.simple_evaluate(
                model=hflm,
                tasks=task_list,
                batch_size=eval_batch_size,
                task_manager=task_manager,
                gen_kwargs=parse_gen_kwargs(args.gen_kwargs),
            )["results"]

            metric_vals = {}
            for task, result in t_results.items():
                metric_vals[task] = round(result.get("acc_norm,none", result.get("acc,none", 0)), 4)

            print(f"Task Results: {metric_vals}")
            from pprint import pprint

            pprint(metric_vals)
            results.update(metric_vals)

            if metric_vals:
                avg_task_score = round(sum(metric_vals.values()) / len(metric_vals), 4)
                avg_msg = f"Task Average Score: {avg_task_score}"
                print(avg_msg)
                logger.info(avg_msg)
                results["task_avg"] = avg_task_score

            reported_metric_vals = {}
            for key, value in metric_vals.items():
                if "mmlu" in key:
                    if key == "mmlu":
                        reported_metric_vals[key] = value
                else:
                    reported_metric_vals[key] = value

            import pandas as pd

            os.makedirs(args.output_dir, exist_ok=True)
            csv_path = f"{args.output_dir}/results.csv"

            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                new_df = pd.DataFrame(reported_metric_vals, index=[0])
                for col in new_df.columns:
                    if col not in df.columns:
                        df[col] = None
                df = pd.concat([df, new_df], ignore_index=True)
            else:
                df = pd.DataFrame(reported_metric_vals, index=[0])

            if len(task_list) >= 5:
                cols = ["piqa", "arc_easy", "arc_challenge", "hellaswag", "winogrande"]
                if all(col in df.columns for col in cols):
                    df["avg-5"] = df[cols].mean(axis=1)
            if len(task_list) >= 6:
                cols = ["piqa", "arc_easy", "arc_challenge", "hellaswag", "winogrande", "boolq"]
                if all(col in df.columns for col in cols):
                    df["avg-6"] = df[cols].mean(axis=1)

            print(f"Saving task results to {csv_path}...")
            print(df)
            df.to_csv(csv_path, index=False)

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