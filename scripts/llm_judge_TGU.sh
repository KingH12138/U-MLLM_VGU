CUDA_VISIBLE_DEVICES=0 python utils/llmasajudge_TGU.py --model_name Bagel --output_dir ./workdirs/COMSC_llm_judge/TGU
CUDA_VISIBLE_DEVICES=1 python utils/llmasajudge_TGU.py --model_name BLIP3o --output_dir ./workdirs/COMSC_llm_judge/TGU
CUDA_VISIBLE_DEVICES=2 python utils/llmasajudge_TGU.py --model_name Janus --output_dir ./workdirs/COMSC_llm_judge/TGU
CUDA_VISIBLE_DEVICES=3 python utils/llmasajudge_TGU.py --model_name JanusFlow --output_dir ./workdirs/COMSC_llm_judge/TGU
CUDA_VISIBLE_DEVICES=4 python utils/llmasajudge_TGU.py --model_name Show-o --output_dir ./workdirs/COMSC_llm_judge/TGU
CUDA_VISIBLE_DEVICES=5 python utils/llmasajudge_TGU.py --model_name UniLIP --output_dir ./workdirs/COMSC_llm_judge/TGU
CUDA_VISIBLE_DEVICES=6 python utils/llmasajudge_TGU.py --model_name Emu3 --output_dir ./workdirs/COMSC_llm_judge/TGU