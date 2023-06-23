wget https://objectstore.e2enetworks.net/ai4b-public-nlu-nlg/indic-corp-frozen-for-the-paper-oct-2022/kn.txt

python IndicBERT/tokenization/build_tokenizer.py \
    --input kn.txt \
    --output TOKENIZER_OUTPUT \
    --vocab_size 23000000

python IndicBERT/process_data/create_mlm_data.py \
    --input_file kn.txt \
    --output_file PREPROCESS_OUTPUT \
    --input_file_type monolingual \
    --tokenizer TOKENIZER_OUT/config.json \
    --max_seq_length 512 \
    --max_predictions_per_seq 50 \
    --do_whole_word_mask False \
    --masked_lm_prob 0.2 \
    --random_seed 42 \
    --dupe_factor 2 \
# not masking whole words
# dupe_factor=2