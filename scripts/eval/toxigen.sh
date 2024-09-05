# # # Evaluating DExperts
# size=7
# python -m eval.toxigen.run_eval \
#     --base_model_name_or_path meta-llama/Llama-2-${size}b-hf \
#     --expert_model_name_or_path meta-llama/Llama-2-7b-chat-hf \
#     --save_dir results/toxigen/dexperts-${size}B \
#     --eval_batch_size 32 \
#     --max_examples_per_group 200


# d experts llama3
# size=8
# python -m eval.toxigen.run_eval \
#     --base_model_name_or_path andrijdavid/Llama3-1B-Base \
#     --expert_model_name_or_path meta-llama/Meta-Llama-3-${size}b-Instruct \
#     --antiexpert_model_name_or_path meta-llama/Meta-Llama-3-${size}b \
#     --save_dir results/toxigen/dexperts-llama3_${size}B \
#     --eval_batch_size 32 \
#     --max_examples_per_group 200


# anti expert to boost toxicity
size=8
python -m eval.toxigen.run_eval \
    --base_model_name_or_path meta-llama/Meta-Llama-3-${size}b \
    --expert_model_name_or_path  bunnycore/NSFWLlama3-8B \
    --antiexpert_model_name_or_path  meta-llama/Meta-Llama-3-${size}b-Instruct\
    --save_dir results/toxigen/toxicv2 \
    --eval_batch_size 64 \
    --max_examples_per_group 100
# # Evaluating Llama 2
# size=13
# python -m eval.toxigen.run_eval \
#     --model_name_or_path meta-llama/Llama-2-${size}b-hf \
#     --save_dir results/toxigen/llama2-${size}B \
#     --eval_batch_size 32 \
#     --max_examples_per_group 200

# Evaluating Llama 3
# size=8
# python -m eval.toxigen.run_eval \
#     --model_name_or_path meta-llama/Meta-Llama-3-${size}b \
#    --save_dir results/toxigen/llama3-${size}B \
#    --eval_batch_size 32 \
#    --max_examples_per_group 200

# # Evaluating Llama 3 instruct
# size=70
# python -m eval.toxigen.run_eval \
#     --model_name_or_path meta-llama/Meta-Llama-3-${size}b-Instruct \
#    --save_dir results/toxigen/llama3-instruct-${size}B \
#    --eval_batch_size 32 \
#    --max_examples_per_group 200


# evaluating llama3 1b


# python -m eval.toxigen.run_eval \
#    --model_name_or_path andrijdavid/Llama3-1B-Base \
#    --save_dir results/toxigen/llama3-1B \
#    --eval_batch_size 256 \
#    --max_examples_per_group 100
# # Evaluating Llama 3 chat
# size=70
# python -m eval.toxigen.run_eval \
#     --model_name_or_path meta-llama/Meta-Llama-3-${size}b \
#    --save_dir results/toxigen/llama3-${size}B \
#    --eval_batch_size 1 \
#    --max_examples_per_group 1

# # Evaluating Llama 2 chat
# size=13
# python -m eval.toxigen.run_eval \
#     --model_name_or_path meta-llama/Llama-2-${size}b-chat-hf \
#     --save_dir results/toxigen/llama2-chat-${size}B \
#     --eval_batch_size 32 \
#     --max_examples_per_group 200 \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format
