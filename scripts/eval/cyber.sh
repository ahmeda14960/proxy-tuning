# Evaluating Llama 3
# size=8
# python -m eval.cyber_wmd_single \
#     --model_name_or_path meta-llama/Meta-Llama-3-${size}b \
#    --save_dir results/cyber_wmd/llama3-${size}B \
#    --eval_batch_size 16 \
#    --sample

# Evaluating Llama 3 1b
# size=8
# python -m eval.cyber_wmd_single \
#     --model_name_or_path andrijdavid/Llama3-1B-Base \
#    --save_dir results/cyber_wmd/llama3-1B \
#    --eval_batch_size 16 \
#    --sample

# evaluate experts

size=8
python -m eval.cyber_wmd \
    --save_dir results/cyber_wmd/dexperts_llama3-1B_top0.9 \
    --base_model_name_or_path andrijdavid/Llama3-1B-Base \
    --expert_model_name_or_path  meta-llama/Meta-Llama-3-${size}b \
    --antiexpert_model_name_or_path  meta-llama/Meta-Llama-3-${size}b-Instruct\
    --eval_batch_size 8 \
    --mode dexperts \
    --sample \


    # batch size 1

    