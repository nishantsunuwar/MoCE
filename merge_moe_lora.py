import transformers

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

from peft import PeftModel

import torch
import os


def merge_lora_to_base_model(path, save_name, num_cluster, topk):#, drop_cluster):
    from transformers_utils import get_keys_to_not_convert, _load_pretrained_model
    import transformers.utils.bitsandbytes
    import transformers.modeling_utils

    transformers.utils.bitsandbytes.get_keys_to_not_convert = get_keys_to_not_convert
    transformers.modeling_utils.PreTrainedModel._load_pretrained_model = (
        _load_pretrained_model
    )

    from camelidae.configuration_camelidae import CamelidaeConfig
    from camelidae.modeling_camelidae import LlamaForCausalLM
    model_path = "meta-llama/Llama-2-7b-chat-hf"
    config_name = CamelidaeConfig
    model_name = LlamaForCausalLM
    auto_map = {
    "AutoConfig": "configuration_camelidae.CamelidaeConfig",
    "AutoModelForCausalLM": "modeling_camelidae.LlamaForCausalLM"
    }

    peft_path = os.path.join(path, "adapter_model")
    moe_path = os.path.join(path, "moe_model.bin")
    save_path = "./SAVEPATH"

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path, use_fast=False, trust_remote_code=True
    )

    model_config = config_name.from_pretrained(model_path)
    model_config.pretraining_tp = 1
    model_config.auto_map = auto_map

    # Camelidae Config
    model_config.moe_dtype = "bfloat16"
    model_config.adapter_dim = 64
    model_config.topk = topk
    model_config.moe_scaling = 0.25
    model_config.num_experts = 4
    model_config.num_clusters = num_cluster
    model_config.output_router_logits = False

    model = model_name.from_pretrained(
        model_path,
        config=model_config,
        torch_dtype=torch.bfloat16,
        device_map={'': 'cpu'}
    )

    moe_weights = torch.load(moe_path, map_location=torch.device("cpu"))  # 여기가 163840
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in moe_weights.items() if k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model = PeftModel.from_pretrained(
        model,
        peft_path,
        torch_dtype=torch.bfloat16,
        device_map={'': 'cpu'}
    )

    model = model.merge_and_unload()

    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)
    model.push_to_hub(save_name)
    tokenizer.push_to_hub(save_name)


if __name__ == '__main__':
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--save', required=True, type=str, help='save file name: e.g. Sugyeong/my_model')
    argparser.add_argument('--path', required=True, type=str,
                           help='insert trained model path name: e.g. ./train_scripts/result_llama2_7b/checkpoint-117/adapter_model/')
    argparser.add_argument('--num_cluster', required=True, type=int, help='num cluster')
    argparser.add_argument('--topk', required=True, type=int, help='topk')

    args = argparser.parse_args()
    merge_lora_to_base_model(model=args.model, path=args.path, save_name=args.save, num_cluster=args.num_cluster, topk=args.topk)#, drop_cluster=args.drop_cluster)
    print('DONE!')