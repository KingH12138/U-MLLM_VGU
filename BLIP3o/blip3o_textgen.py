from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

def get_model():
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    # default processer
    min_pixels = 256*28*28
    max_pixels = 1280*28*28
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
    return processor, model


def generate_answer(question, processor, model):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generation_config = dict(
        max_new_tokens=1024,  # 提升最大生成长度，避免截断
        do_sample=False,  # 关闭采样（采样会降低指令依从性，优先确定性生成）
        repetition_penalty=1.1,  # 轻微重复惩罚，避免废话
        num_beams=4,  # 束搜索（提升生成质量和指令依从性）
        early_stopping=True,  # 束搜索完成后停止，避免无意义生成
    )
    generated_ids = model.generate(**inputs, **generation_config)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]


if __name__ == "__main__":
    processor, model = get_model()
    question = "Who are you?"
    print(generate_answer(question, processor, model))