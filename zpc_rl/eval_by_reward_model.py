
import json, torch
from vllm import LLM
from transformers.utils import cached_file
from transformers import AutoConfig, AutoModelForTokenClassification
from transformers import AutoTokenizer
import deepspeed
import os

V_HEAD_WEIGHTS_NAME = "value_head.bin"

V_HEAD_SAFE_WEIGHTS_NAME = "value_head.safetensors"
def load_valuehead_params(path_or_repo_id: str) -> dict[str, torch.Tensor]:
    r"""Load value head parameters from Hugging Face Hub or local disk.

    Returns: dict with keys `v_head.summary.weight` and `v_head.summary.bias`.
    """
    kwargs = {"path_or_repo_id": path_or_repo_id}
    try:
        from safetensors import safe_open

        vhead_file = cached_file(filename=V_HEAD_SAFE_WEIGHTS_NAME, **kwargs)
        with safe_open(vhead_file, framework="pt", device="cpu") as f:
            return {key: f.get_tensor(key) for key in f.keys()}
    except Exception as err:
        err_text = str(err)

    try:
        vhead_file = cached_file(filename=V_HEAD_WEIGHTS_NAME, **kwargs)
        return torch.load(vhead_file, map_location="cpu", weights_only=True)
    except Exception as err:
        err_text = str(err)

    return None

if __name__ == "__main__":
    torch.distributed.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    print(f"Rank {local_rank}/{world_size} initialized")

    tokenizer = AutoTokenizer.from_pretrained("/data/k8s/zpc/Qwen2.5-Coder-7B")
    model = AutoModelForTokenClassification.from_pretrained(
                pretrained_model_name_or_path="/data/k8s/zpc/Custom-LLaMA-Factory/arkts_linter_reward_model_merged_v1",
                torch_dtype=torch.bfloat16,
            )
    model.to(torch.bfloat16)
    model.to(f"cuda:{local_rank}")
    with open("record_prompt_and_response.jsonl", "r") as file:
        data_list = [json.loads(data) for data in list[str](file)]
    # with open("ds_config.json", "r") as f:
    #     ds_config = json.load(f)

    ds_engine = deepspeed.init_inference(
        model=model,
        config="ds_config.json",
        mp_size=4,  # 如果单GPU，设为1
        dtype=torch.bfloat16,  # 使用半精度进一步节省内存
        replace_method="auto",
        replace_with_kernel_inject=True,  # 注入优化kernel
    )

    for data in data_list:
        prompt = data["prompt"]
        response = data["response"]
        print(f"input is {prompt+response}\n len is{len(tokenizer.encode(prompt+response))}")
        output = ds_engine(input_ids=torch.tensor(tokenizer.encode(prompt+response)).to("cuda:0").view(1, -1))
            
        rm_score = output.logits  # (batch_size, seq_len, 1)
        rm_score = rm_score.squeeze(-1)[:,-1]
        print(f"rm_score is {rm_score}")