# Baseline without Fine-tuning

# python main.py --push_to_hub True --exp_save_name mistral-7b-instruct-v0.3-bnb-4bit-base --wandb_project mistral-7Bv0.3-Instruct-FTandEvalv2 --pretrained_model unsloth/mistral-7b-instruct-v0.3-bnb-4bit --per_device_train_batch_size 4 --test True --hf_model_path unsloth/mistral-7b-instruct-v0.3-bnb-4bit

# python main.py --push_to_hub True --exp_save_name Meta-Llama-3.1-8B-Instruct-bnb-4bit-base --wandb_project unsloth_Llama-3.1-8B-Instruct-FTandEval --pretrained_model unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit --per_device_train_batch_size 4 --test True --hf_model_path unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit

# python main.py --push_to_hub True --exp_save_name Llama-3.2-3B-Instruct-base --wandb_project unsloth_Llama-3.2-3B-Instruct-FTandEval --pretrained_model unsloth/Llama-3.2-3B-Instruct --per_device_train_batch_size 4 --test True --hf_model_path unsloth/Llama-3.2-3B-Instruct

# Finetuning

main.py --push_to_hub True --exp_save_name llama-3.1-8B-Finetuned-v2 --wandb_project unsloth_Llama-3.1-8B-Instruct-FTandEval --pretrained_model unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit --per_device_train_batch_size 4


main.py --push_to_hub True --exp_save_name llama-3.2-3B-Finetuned-v2 --wandb_project unsloth_Llama-3.2-3B-Instruct-FTandEval --pretrained_model unsloth/Llama-3.2-3B-Instruct --per_device_train_batch_size 4