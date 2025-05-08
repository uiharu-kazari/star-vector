import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from torch import cuda


def is_flash_attention_2_supported():
    if not cuda.is_available():
        return False
    major, minor = cuda.get_device_capability()
    compute_capability = float(f"{major}.{minor}")
    # FlashAttention 2 requires compute capability >= 8.0 (Ampere or newer)
    return compute_capability >= 8.0


class StarCoderModel(nn.Module):
    def __init__(self, config, **kwargs):
        super(StarCoderModel, self).__init__()

        self.init_tokenizer(config.starcoder_model_name)

        self.max_length = config.max_length
        model_config = AutoConfig.from_pretrained(
            config.starcoder_model_name, trust_remote_code=True
        )
        kwargs = {}
        kwargs["trust_remote_code"] = True
        kwargs["torch_dtype"] = config.torch_dtype

        # Configure special tokens for generation
        model_config.eos_token_id = self.tokenizer.eos_token_id
        model_config.pad_token_id = self.tokenizer.pad_token_id
        model_config.bos_token_id = self.tokenizer.bos_token_id
        attn_implementation = getattr(config, "attn_implementation", "eager")
        if attn_implementation not in ["eager", "sdpa", "flash_attention_2"]:
            raise ValueError(
                f"attn_implementation {attn_implementation} not supported. Choose from 'eager', 'sdpa', or 'flash_attention_2'"
            )

        if (
            attn_implementation == "flash_attention_2"
            and not is_flash_attention_2_supported()
        ):
            print(
                "Warning: flash_attention_2 requested but not supported by hardware. Falling back to eager implementation."
            )
            attn_implementation = "eager"

        if (
            attn_implementation == "flash_attention_2"
            and is_flash_attention_2_supported()
        ):
            model_config.flash_attention = True
            model_config._attn_implementation = "flash_attention_2"
        else:
            config.use_flash_attn = False
            model_config._attn_implementation = attn_implementation

        # print the attention method
        print(f"Using attention implementation: {model_config._attn_implementation}")

        # model = GPTBigCodeForCausalLM(config=model_config)
        model = AutoModelForCausalLM.from_pretrained(
            config.starcoder_model_name, config=model_config, **kwargs
        )
        model.resize_token_embeddings(len(self.tokenizer))
        self.transformer = model

        # Prompt the model after image
        self.prompt = "<svg"

    def init_tokenizer(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        # Incude padding and eos tokens in the vocabulary
        if self.tokenizer.eos_token_id is None:
            self.tokenizer.add_special_tokens({"eos_token": "[EOS]"})
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        self.svg_start_token = "<svg-start>"
        self.image_start_token = "<image-start>"
        self.text_start_token = "<caption-start>"

        self.tokenizer.add_tokens(
            [self.svg_start_token, self.image_start_token, self.text_start_token]
        )
        self.svg_start_token_id = self.tokenizer.encode(self.svg_start_token)[0]
