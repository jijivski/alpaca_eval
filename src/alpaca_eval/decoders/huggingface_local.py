import logging
from typing import Optional, Sequence

import numpy as np
import torch
import transformers
from peft import PeftModel
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .. import constants, utils

__all__ = ["huggingface_local_completions"]


class ListDataset(Dataset):
    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]


def huggingface_local_completions(
    prompts: Sequence[str],
    model_name: str,
    do_sample: bool = False,
    batch_size: int = 1,
    model_kwargs=None,
    cache_dir: Optional[str] = constants.DEFAULT_CACHE_DIR,
    remove_ending: Optional[str] = None,
    is_fast_tokenizer: bool = True,
    adapters_name: Optional[str] = None,
    **kwargs,
) -> dict[str, list]:
    """Decode locally using huggingface transformers pipeline.

    Parameters
    ----------
    prompts : list of str
        Prompts to get completions for.

    model_name : str, optional
        Name of the model (repo on hugging face hub)  to use for decoding.

    do_sample : bool, optional
        Whether to use sampling for decoding.

    batch_size : int, optional
        Batch size to use for decoding. This currently does not work well with to_bettertransformer.

    model_kwargs : dict, optional
        Additional kwargs to pass to from_pretrained.

    cache_dir : str, optional
        Directory to use for caching the model.

    remove_ending : str, optional
        The ending string to be removed from completions. Typically eos_token.

    kwargs :
        Additional kwargs to pass to `InferenceClient.__call__`.
    """
    model_kwargs = model_kwargs or {}
    if "device_map" not in model_kwargs:
        model_kwargs["device_map"] = "auto"
    if "torch_dtype" in model_kwargs and isinstance(model_kwargs["torch_dtype"], str):
        model_kwargs["torch_dtype"] = getattr(torch, model_kwargs["torch_dtype"])

    n_examples = len(prompts)
    if n_examples == 0:
        logging.info("No samples to annotate.")
        return []
    else:
        logging.info(f"Using `huggingface_local_completions` on {n_examples} prompts using {model_name}.")

    if not torch.cuda.is_available():
        model_kwargs["load_in_8bit"] = False
        model_kwargs["torch_dtype"] = None

    #  faster but slightly less accurate matrix multiplications
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        padding_side="left",
        use_fast=is_fast_tokenizer,
        **model_kwargs,
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, **model_kwargs).eval()

    if adapters_name:
        logging.info(f"Merging adapter from {adapters_name}.")
        model = PeftModel.from_pretrained(model, adapters_name)
        model = model.merge_and_unload()

    if batch_size == 1:
        try:
            model = model.to_bettertransformer()
        except:
            # could be not implemented or natively supported
            pass

    logging.info(f"Model memory: {model.get_memory_footprint() / 1e9} GB")

    if batch_size > 1:
        # sort the prompts by length so that we don't necessarily pad them by too much
        # save also index to reorder the completions
        original_order, prompts = zip(*sorted(enumerate(prompts), key=lambda x: len(x[1])))
        prompts = list(prompts)

    if not tokenizer.pad_token_id:
        # set padding token if not set
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token

    default_kwargs = dict(
        do_sample=do_sample,
        model_kwargs={k: v for k, v in model_kwargs.items() if k != "trust_remote_code"},
        batch_size=batch_size,
    )
    default_kwargs.update(kwargs)
    logging.info(f"Kwargs to completion: {default_kwargs}")
    # breakpoint()
    '''(vllm) [wangxidong@pgpu26 alpaca_eval]$ alpaca_eval evaluate_from_model   --model_configs '/mntnfs/med_data5/chenghao/alpaca_eval/src/alpaca_eval/models_configs/llama-2-7b-chat-hf/configs.yaml'   --annotators_config 'alpaca_eval_gpt4_turbo_fn' 
    Using the latest cached version of the module from /home/wangxidong/.cache/huggingface/modules/datasets_modules/datasets/tatsu-lab--alpaca_eval/1e5100c79ad26a4779a5903cfb0148b793bb73b3e114580b73bf868affec2f39 (last modified on Mon Apr  1 17:11:38 2024) since it couldn't be found locally at tatsu-lab/alpaca_eval, or remotely on the Hugging Face Hub.
    Chunking for generation:   0%|                                                                                                                                                                                                             | 0/13 [00:00<?, ?it/s]INFO:root:Using `huggingface_local_completions` on 64 prompts using /mntcephfs/data/med/guimingchen/models/general/llama2-7b-chat-hf.
    Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:19<00:00,  9.83s/it]
    INFO:root:Model memory: 13.543948288 GB█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:19<00:00,  9.06s/it]
    INFO:root:Kwargs to completion: {'do_sample': True, 'model_kwargs': {'torch_dtype': torch.float16, 'device_map': 'auto'}, 'batch_size': 1, 'max_new_tokens': 2048, 'temperature': 0.7, 'top_p': 1.0}
    > /mntnfs/med_data5/chenghao/alpaca_eval/src/alpaca_eval/decoders/huggingface_local.py(130)huggingface_local_completions()
    -> pipeline = transformers.pipeline(
    (Pdb) w
    /mntnfs/med_data5/wangxidong/wangxidong/anaconda3/envs/vllm/bin/alpaca_eval(33)<module>()
    -> sys.exit(load_entry_point('alpaca-eval', 'console_scripts', 'alpaca_eval')())
    /mntnfs/med_data5/chenghao/alpaca_eval/src/alpaca_eval/main.py(607)main()
    -> fire.Fire(ALL_FUNCTIONS)
    /mntnfs/med_data5/wangxidong/wangxidong/anaconda3/envs/vllm/lib/python3.10/site-packages/fire/core.py(143)Fire()
    -> component_trace = _Fire(component, args, parsed_flag_args, context, name)
    /mntnfs/med_data5/wangxidong/wangxidong/anaconda3/envs/vllm/lib/python3.10/site-packages/fire/core.py(477)_Fire()
    -> component, remaining_args = _CallAndUpdateTrace(
    /mntnfs/med_data5/wangxidong/wangxidong/anaconda3/envs/vllm/lib/python3.10/site-packages/fire/core.py(693)_CallAndUpdateTrace()
    -> component = fn(*varargs, **kwargs)
    /mntnfs/med_data5/chenghao/alpaca_eval/src/alpaca_eval/main.py(342)evaluate_from_model()
    -> model_outputs = get_completions(
    /mntnfs/med_data5/chenghao/alpaca_eval/src/alpaca_eval/main.py(327)get_completions()
    -> completions = fn_completions(prompts=prompts, **configs["completions_kwargs"])["completions"]
    > /mntnfs/med_data5/chenghao/alpaca_eval/src/alpaca_eval/decoders/huggingface_local.py(130)huggingface_local_completions()
    -> pipeline = transformers.pipeline(
    (Pdb) 
    (Pdb) model
    LlamaForCausalLM(
    (model): LlamaModel(
        (embed_tokens): Embedding(32000, 4096)
        (layers): ModuleList(
        (0-31): 32 x LlamaDecoderLayer(
            (self_attn): LlamaSdpaAttention(
            (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (rotary_emb): LlamaRotaryEmbedding()
            )
            (mlp): LlamaMLP(
            (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
            (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
            (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
            (act_fn): SiLU()
            )
            (input_layernorm): LlamaRMSNorm()
            (post_attention_layernorm): LlamaRMSNorm()
        )
        )
        (norm): LlamaRMSNorm()
    )
    (lm_head): Linear(in_features=4096, out_features=32000, bias=False)
    )
    (Pdb) model.device
    device(type='cuda', index=0)'''
    pipeline = transformers.pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        **default_kwargs,
        # trust_remote_code=model_kwargs.get("trust_remote_code", False),
        trust_remote_code=model_kwargs.get("trust_remote_code", True),

    )

    ## compute and log the time for completions
    prompts_dataset = ListDataset(prompts)
    completions = []
    print('len(prompts_dataset)',len(prompts_dataset))
    # breakpoint()
    '''

    

okay:

pipe = transformers.pipeline("text-generation", model="/mntnfs/med_data5/fanyaxin/Llama-2-7b-hf", trust_remote_code=True)
print(pipe("Hello, I'm a language model"))

pipe = transformers.pipeline("text-generation", model="/mntcephfs/data/med/chenghao/models/Qwen-1_8B", trust_remote_code=True)
print(pipe("Hello, I'm a language model"))
# 'QWenLMHeadModel' is not supported for text-generation.


pipe = transformers.pipeline("text-generation", model="/mntcephfs/data/med/chenghao/models/phi-2", trust_remote_code=True)
print(pipe("Hello, I'm a language model"))
    (Pdb) pipe = transformers.pipeline("text-generation", model="/mntcephfs/data/med/chenghao/models/phi-2", trust_remote_code=True)
    Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:07<00:00,  3.89s/it]
    Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.██████████████████████████████████████████████████████████| 2/2 [00:07<00:00,  3.34s/it]
    (Pdb) print(pipe("Hello, I'm a language model"))
    /mntnfs/med_data5/wangxidong/wangxidong/anaconda3/envs/vllm2/lib/python3.10/site-packages/transformers/generation/utils.py:1132: UserWarning: Using the model-agnostic default `max_length` (=20) to control the generation length. We recommend setting `max_new_tokens` to control the maximum length of the generation.
    warnings.warn(
    [{'generated_text': "Hello, I'm a language model AI and I'm here to help you with your writing. However"}]
    (Pdb) 

default_kwargs={'model_kwargs': {'device_map': 'auto'}}
pipe = transformers.pipeline("text-generation", model="/mntcephfs/data/med/chenghao/models/phi-2", trust_remote_code=True,**default_kwargs)
print(pipe("Hello, I'm a language model"))


nok
default_kwargs={'model_kwargs': {'device_map': 'auto'}}
pipe = transformers.pipeline("text-generation", model="/mntnfs/med_data5/fanyaxin/Llama-2-7b-hf",**default_kwargs, trust_remote_code=True)
print(pipe("Hello, I'm a language model"))

---

default_kwargs={'do_sample': True, 'model_kwargs': {'torch_dtype': torch.float16, 'device_map': 'auto'}, 'batch_size': 1, 'max_new_tokens': 2048, 'temperature': 0.7, 'top_p': 1.0}


default_kwargs={'do_sample': True, 'model_kwargs': {'torch_dtype': torch.float16, 'device_map': 'auto'}, 'batch_size': 1, 'max_new_tokens': 2048, 'temperature': 0.7, 'top_p': 1.0}


default_kwargs={'do_sample': True, 'batch_size': 1, 'max_new_tokens': 2048, 'temperature': 0.7, 'top_p': 1.0}

default_kwargs={'batch_size': 1, 'max_new_tokens': 200,'model_kwargs': {'device_map': 'auto'}}

pipe = transformers.pipeline("text-generation", model="/mntnfs/med_data5/fanyaxin/Llama-2-7b-hf",**default_kwargs, trust_remote_code=True)
print(pipe("Hello, I'm a language model"))





    pipeline = transformers.pipeline(        task="text-generation",        model=model,        tokenizer=tokenizer,        **default_kwargs,        trust_remote_code=model_kwargs.get("trust_remote_code", True),)
    
    pipeline(prompts_dataset[:1],return_full_text=False,pad_token_id=tokenizer.pad_token_id,)
    pipeline(prompts_dataset[0],return_full_text=False,pad_token_id=tokenizer.pad_token_id,)

    pipeline2(prompts_dataset[0],return_full_text=False,pad_token_id=tokenizer.pad_token_id,)

    prompts_dataset = ListDataset(prompts)

    '''


    with utils.Timer() as t:
        for out in tqdm(
            pipeline(
                prompts_dataset,#:like list[String]
                return_full_text=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        ):
            # breakpoint()
            
            generated_text = out[0]["generated_text"]
            if remove_ending is not None and generated_text.endswith(remove_ending):
                generated_text = generated_text[: -len(remove_ending)]
            completions.append(generated_text)

    logging.info(f"Time for {n_examples} completions: {t}")

    if batch_size > 1:
        # reorder the completions to match the original order
        completions, _ = zip(*sorted(list(zip(completions, original_order)), key=lambda x: x[1]))
        completions = list(completions)

    # local => price is really your compute
    price = [np.nan] * len(completions)
    avg_time = [t.duration / n_examples] * len(completions)

    return dict(completions=completions, price_per_example=price, time_per_example=avg_time)
