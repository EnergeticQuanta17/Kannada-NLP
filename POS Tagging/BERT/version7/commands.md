wget https://objectstore.e2enetworks.net/ai4b-public-nlu-nlg/indic-corp-frozen-for-the-paper-oct-2022/kn.txt

python IndicBERT/tokenization/build_tokenizer.py \
    --input kn.txt \
    --output TOKENIZER_OUTPUT \
    --vocab_size 23000000

python IndicBERT/process_data/create_mlm_data.py \
    --input_file kn.txt \
    --output_file PREPROCESS_OUTPUT \
    --input_file_type monolingual \
    --tokenizer TOKENIZER_OUTPUT/config.json \
    --max_seq_length 512 \
    --max_predictions_per_seq 50 \
    --do_whole_word_mask False \
    --masked_lm_prob 0.2 \
    --random_seed 42 \
    --dupe_factor 1 \
# not masking whole words
# dupe_factor=1

python IndicBERT/train/run_pretraining.py \
--input_file kn_10k.txt \
--output_dir PRETRAINING_OUTPUT \
--do_train True \
--bert_config_file config.json \
--train_batch_size 32 \
--max_seq_length 512 \
--max_predictions_per_seq 50 \
--num_train_steps 1000 \
--num_warmup_steps 10 \
--learning_rate 0.0005 \
--save_checkpoints_steps 1 \
--use_tpu False


python IndicBERT/fine-tuning/ner/ner.py \
    --model_name_or_path aquorio15/KannadaBERT-lamb \
    --do_train True