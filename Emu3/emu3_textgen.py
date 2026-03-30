# -*- coding: utf-8 -*-
from PIL import Image
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor, AutoModelForCausalLM
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation import LogitsProcessorList, PrefixConstrainedLogitsProcessor, UnbatchedClassifierFreeGuidanceLogitsProcessor
import torch
torch.backends.cudnn.enabled = False
from emu3.mllm.processing_emu3 import Emu3Processor

def get_model():
    # model path
    EMU_HUB = "./checkpoints/BAAI/Emu3-Chat"
    VQ_HUB = "./checkpoints/BAAI/Emu3-VisionTokenizer"

    # prepare model and processor
    model = AutoModelForCausalLM.from_pretrained(
        EMU_HUB,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(EMU_HUB, trust_remote_code=True, padding_side="left")
    image_processor = AutoImageProcessor.from_pretrained(VQ_HUB, trust_remote_code=True)
    image_tokenizer = AutoModel.from_pretrained(VQ_HUB, device_map="cuda", trust_remote_code=True).eval()
    processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)

    return processor, model, tokenizer


def generate_text(text, processor, model, tokenizer):
    # prepare input
    prompt = [text]
    inputs = processor(
        text=prompt,
        image=None,
        mode='U',
        padding_image=True,
        padding="longest",
        return_tensors="pt",
    )

    GENERATION_CONFIG = GenerationConfig(pad_token_id=tokenizer.pad_token_id, bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id)

    # generate
    outputs = model.generate(
        inputs.input_ids.to("cuda"),
        GENERATION_CONFIG,
        max_new_tokens=1024,
        attention_mask=inputs.attention_mask.to("cuda"),
    )

    outputs = outputs[:, inputs.input_ids.shape[-1]:]
    answers = processor.batch_decode(outputs, skip_special_tokens=True)
    
    return answers[0]

if __name__ == "__main__":
    processor, model, tokenizer = get_model()
    question = "Which word is longer: 'Multimodal' or 'Understanding'?"
    answer = generate_text(question, processor, model, tokenizer)
    print(answer)
    