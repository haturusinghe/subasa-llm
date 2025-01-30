# Baseline without Fine-tuning

# python main.py --push_to_hub True --exp_save_name mistral-7b-instruct-v0.3-bnb-4bit-base --wandb_project mistral-7Bv0.3-Instruct-FTandEvalv2 --pretrained_model unsloth/mistral-7b-instruct-v0.3-bnb-4bit --per_device_train_batch_size 4 --test True --hf_model_path unsloth/mistral-7b-instruct-v0.3-bnb-4bit

# python main.py --push_to_hub True --exp_save_name Meta-Llama-3.1-8B-Instruct-bnb-4bit-base --wandb_project unsloth_Llama-3.1-8B-Instruct-FTandEval --pretrained_model unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit --per_device_train_batch_size 4 --test True --hf_model_path unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit

# python main.py --push_to_hub True --exp_save_name Llama-3.2-3B-Instruct-base --wandb_project unsloth_Llama-3.2-3B-Instruct-FTandEval --pretrained_model unsloth/Llama-3.2-3B-Instruct --per_device_train_batch_size 4 --test True --hf_model_path unsloth/Llama-3.2-3B-Instruct

# Finetuning for Augmented Dataset
# rm -rf s-haturusinghe/

# # Llama 3.2 3B with Augmented Dataset
# python main.py --push_to_hub True --exp_save_name llama-3.2-3B-Finetuned-v2.1-Augmented --wandb_project unsloth_Llama-3.2-3B-Instruct-FTandEval --pretrained_model unsloth/Llama-3.2-3B-Instruct --per_device_train_batch_size 4 --use_augmented_dataset True --test True

# rm -rf s-haturusinghe/
# # Llama 3.1 8B with Augmented Dataset
# python main.py --push_to_hub True --exp_save_name llama-3.1-8B-Finetuned-v2.1-Augmented --wandb_project unsloth_Llama-3.1-8B-Instruct-FTandEval --pretrained_model unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit --per_device_train_batch_size 4 --use_augmented_dataset True --test True

# # For Mistral Models
# rm -rf s-haturusinghe/

python main.py --push_to_hub True --exp_save_name mistral-7Bv0.3-ft-v3.1-DEBUG --wandb_project mistral-7Bv0.3-Instruct-FTandEvalv2 --pretrained_model unsloth/mistral-7b-instruct-v0.3-bnb-4bit --per_device_train_batch_size 4 --test True --debug True --max_steps 2

# rm -rf s-haturusinghe/
# # Mistral buth with Augmented Dataset
# python main.py --push_to_hub True --exp_save_name mistra-7Bv0.3-ft-v3.1-Augmented --wandb_project mistral-7Bv0.3-Instruct-FTandEvalv2 --pretrained_model unsloth/mistral-7b-instruct-v0.3-bnb-4bit --per_device_train_batch_size 4 --use_augmented_dataset True --test True


## FOR SUHS Dataset 

# python main.py --exp_save_name Meta-Llama-3.1-8B-Instruct-bnb-4bit-base-SUHS --wandb_project unsloth_Llama-3.1-8B-Instruct-FTandEval-SUHS --pretrained_model unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit --per_device_train_batch_size 4 --test True --hf_model_path unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit --dataset suhs --debug True

# python main.py --exp_save_name Llama-3.2-3B-Instruct-base-SUHS --wandb_project unsloth_Llama-3.2-3B-Instruct-FTandEval-SUHS --pretrained_model unsloth/Llama-3.2-3B-Instruct --per_device_train_batch_size 4 --test True --hf_model_path unsloth/Llama-3.2-3B-Instruct --dataset suhs --debug True

# python main.py --exp_save_name mistral-7b-instruct-v0.3-bnb-4bit-base-SUHS --wandb_project mistral-7Bv0.3-Instruct-FTandEvalv2-SUHS --pretrained_model unsloth/mistral-7b-instruct-v0.3-bnb-4bit --per_device_train_batch_size 4 --test True --hf_model_path unsloth/mistral-7b-instruct-v0.3-bnb-4bit --dataset suhs --debug True