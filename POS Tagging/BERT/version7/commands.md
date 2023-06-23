wget https://objectstore.e2enetworks.net/ai4b-public-nlu-nlg/indic-corp-frozen-for-the-paper-oct-2022/kn.txt

python IndicBERT/tokenization/build_tokenizer.py \
    --input kn.txt \
    --output_dir TOKENIZER_OUTPUT \
    --vocab_size 23000000