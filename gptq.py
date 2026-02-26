import transformers
import gptq_utils
from quant_utils import create_logger

def gptq_for_model(model,model_path,w_bits=4,nsamples=128,cal_dataset="wikitext2",use_rtn=False):
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args([])

    logger = create_logger()
    args.seed = 32
    args.model = model_path
    transformers.set_seed(args.seed)
    model.eval()


    args.w_bits = w_bits     # quantization bits
    args.w_clip = True
    args.cal_dataset = cal_dataset
    args.nsamples = nsamples
    

    args.w_groupsize = -1
    args.w_asym = True


    args.int8_down_proj = True
    args.act_order = True
    args.percdamp = 0.01
          
                
    if args.w_bits < 16:
        save_dict = {}        
        if use_rtn is True:
            # rtn
            quantizers = gptq_utils.rtn_fwrd(model, gptq_utils.DEV, args,logger=logger)
            save_dict["w_quantizers"] = quantizers
        else:
            # gptq
            trainloader = gptq_utils.get_loaders(
                args.cal_dataset, nsamples=args.nsamples,
                seed=args.seed, model=args.model,
                seqlen=2048, eval_mode=False
            )
            quantizers = gptq_utils.gptq_fwrd(model, trainloader, gptq_utils.DEV, args,logger=logger)
            save_dict["w_quantizers"] = quantizers
        
    return model