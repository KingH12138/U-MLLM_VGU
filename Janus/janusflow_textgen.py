import torch
from janus.janusflow.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images

def get_model():
    # specify the path to the model
    model_path = "./checkpoints/deepseek-ai/JanusFlow-1.3B"
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt = MultiModalityCausalLM.from_pretrained(
        model_path, trust_remote_code=True
    )
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

    return vl_chat_processor, tokenizer, vl_gpt

def get_answer(prompt, vl_chat_processor, tokenizer, vl_gpt):
    conversation = [
        {
            "role": "User",
            "content": f"{prompt}",
        },
        {"role": "Assistant", "content": ""},
    ]

    # load images and prepare for inputs
    prepare_inputs = vl_chat_processor(
        conversations=conversation,  images=[], force_batchify=True
    ).to(vl_gpt.device)

    # # run image encoder to get the image embeddings
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    # # run the model to get the response
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False,
        use_cache=True,
    )

    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    
    return answer
