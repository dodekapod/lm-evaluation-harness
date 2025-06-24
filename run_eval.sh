# \$1 - model name
# \$2 - ip address with port

lm_eval --model ragondin \
        --tasks hellaswag,squadv2,triviaqa,truthfulqa_gen,truthfulqa_mc1,truthfulqa_mc2,mmlu,mmlu_pro,openbookqa,gpqa,commonsense_qa \
        --output_path results \
        --use_cache cache \
        --cache_requests true \
        --log_samples \
        --model_args model=$1,max_length=32768,base_url=http://$2/v1/completions,num_concurrent=8,max_retries=3,tokenized_requests=False

