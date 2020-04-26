import random
import torch
import torch.nn.functional as F
import warnings
from itertools import chain
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer

from train import SPECIAL_TOKENS, build_input_from_segments, add_special_tokens_
from utils import get_dataset, download_pretrained_model

config = {
    "dataset_path": "", # Path or url of the dataset. If empty download from S3.")
    "dataset_cache": './dataset_cache', # Path or url of the dataset cache")
    "model": "openai-gpt", # Model type (openai-gpt or gpt2)", choices=['openai-gpt', 'gpt2'])  # anything besides gpt2 will load openai-gpt
    "model_checkpoint": "", # Path, url or short name of the model")
    "max_history": 2, # Number of previous utterances to keep in history")
    "device": "cuda" if torch.cuda.is_available() else "cpu", # if torch.cuda.is_available() else "cpu", # Device (cuda or cpu)")
    "no_sample": "", # Set to use greedy decoding instead of sampling")
    "max_length": 20, # Maximum length of the output utterances")
    "min_length": 1, # Minimum length of the output utterances")
    "seed": 0, # Seed")
    "temperature": 0.7, # Sampling softmax temperature")
    "top_k": 0, # Filter top-k tokens before sampling (<=0: no filtering)")
    "top_p": 0.9, # Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
}

class bot:
    def __init__(self):
        if config["model_checkpoint"] == "":
            config["model_checkpoint"] = download_pretrained_model()

        tokenizer_class, model_class = (GPT2Tokenizer, GPT2LMHeadModel) if config["model"] == 'gpt2' else (OpenAIGPTTokenizer, OpenAIGPTLMHeadModel)
        self.tokenizer = tokenizer_class.from_pretrained(config["model_checkpoint"])

        self.model = model_class.from_pretrained(config["model_checkpoint"])
        self.model.to(config["device"]) 
        add_special_tokens_(self.model, self.tokenizer)

    def get_text(self, personality_str, history_str, current_output=None):
        personality = list(map(self.tokenizer.encode, personality_str))
        history = list(map(self.tokenizer.encode, history_str))
        special_tokens_ids = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
        if current_output is None:
            current_output = []

        for i in range(config["max_length"]):
            instance = build_input_from_segments(personality, history, current_output, self.tokenizer, with_eos=False)

            input_ids = torch.tensor(instance["input_ids"], device=config["device"]).unsqueeze(0)
            token_type_ids = torch.tensor(instance["token_type_ids"], device=config["device"]).unsqueeze(0)

            logits = self.model(input_ids, token_type_ids=token_type_ids)
            if isinstance(logits, tuple):  # for gpt2 and maybe others
                logits = logits[0]
            logits = logits[0, -1, :] / config["temperature"]
            logits = self.top_filtering(logits, top_k=config["top_k"], top_p=config["top_p"])
            probs = F.softmax(logits, dim=-1)

            prev = torch.topk(probs, 1)[1] if config["no_sample"] else torch.multinomial(probs, 1)
            if i < config["min_length"] and prev.item() in special_tokens_ids:
                while prev.item() in special_tokens_ids:
                    if probs.max().item() == 1:
                        warnings.warn("Warning: model generating special token with probability 1.")
                        break  # avoid infinitely looping over special token
                    prev = torch.multinomial(probs, num_samples=1)

            if prev.item() in special_tokens_ids:
                break
            current_output.append(prev.item())

        return self.tokenizer.decode(current_output, skip_special_tokens=True)
    
    def top_filtering(self, logits, top_k=0., top_p=0.9, threshold=-float('Inf'), filter_value=-float('Inf')):
        """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
            Args:
                logits: logits distribution shape (vocabulary size)
                top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
                top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                    whose total probability mass is greater than or equal to the threshold top_p.
                    In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                    the threshold top_p.
                threshold: a minimal threshold to keep logits
        """
        assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
        top_k = min(top_k, logits.size(-1))
        if top_k > 0:
            # Remove all tokens with a probability less than the last token in the top-k tokens
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p > 0.0:
            # Compute cumulative probabilities of sorted tokens
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probabilities > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # Back to unsorted indices and set them to -infinity
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = filter_value

        indices_to_remove = logits < threshold
        logits[indices_to_remove] = filter_value

        return logits
    