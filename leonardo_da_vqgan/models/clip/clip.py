import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
from typing import Any, Union, List

from clip.model import build_model
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

BICUBIC = InterpolationMode.BICUBIC
_tokenizer = _Tokenizer()

# source: https://github.com/openai/CLIP
class CLIP:
    def __init__(self, device):
        self.device = device

    def _convert_image_to_rgb(self, image):
        return image.convert("RGB")

    def tokenize(self, texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> torch.LongTensor:
        if isinstance(texts, str):
            texts = [texts]

        sot_token = _tokenizer.encoder["<|startoftext|>"]
        eot_token = _tokenizer.encoder["<|endoftext|>"]
        all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                if truncate:
                    tokens = tokens[:context_length]
                    tokens[-1] = eot_token
                else:
                    raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
            result[i, :len(tokens)] = torch.tensor(tokens)

        return result

    def transform(self, n_px):
        return Compose([
            Resize(n_px, interpolation=BICUBIC),
            CenterCrop(n_px),
            self._convert_image_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def load(self, model_path):
        model = torch.jit.load(model_path, map_location=self.device).eval()

        model = build_model(model.state_dict()).to(self.device)

        device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(self.device)), example_inputs=[])
        device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]

        def patch_device(module):
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []

            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("prim::Constant"):
                    if "value" in node.attributeNames() and str(node["value"]).startswith("cuda"):
                        node.copyAttributes(device_node)

        model.apply(patch_device)
        patch_device(model.encode_image)
        patch_device(model.encode_text)

        return model, self.transform(model.visual.input_resolution)
