#### To train a POS tagger and chunker, 2 programs are run in the following sequence
1. create_index_to_char_and_index_to_label_dictionary_from_training_data.py
	- this creates required index to character, index to label dictionaries to be used while training the BILSTM model
	- how to run this
	python create_index_to_char_and_index_to_label_dictionary_from_training_data.py --train train.data --lang eng
2. train_bilstm_c2w_pos_chunk_with_generators.py
	- this trains the BILSTM model, this code is based on BILSTM C2W model, uses fasttext embeddings of 300 dimensions (epochs can be increased to 100), fastTextVectorFile is a pretrained fastext embeddings binary file
	- this also predicts for Validation and Test data
	- how to run
	python train_bilstm_c2w_pos_chunk_with_generators.py --train train.data --val validation.data --test test.data --embed fastTextVectorFile --lang eng --wt weight-eng-bilstm-c2w --epoch 1
3. Another code for predicting tags
	- how to run
	python predict_tags_using_model_and_generators.py --test test.data --embed fastTextVectorFile --lang eng --model weight-eng-bilstm-c2w --output output_file.txt

4. This work is a direct implementation of the [paper](https://arxiv.org/pdf/1808.03175.pdf)
5. To cite this work:
use this citation
``
@misc{todi2018building,
      title={Building a Kannada POS Tagger Using Machine Learning and Neural Network Models}, 
      author={Ketan Kumar Todi and Pruthwik Mishra and Dipti Misra Sharma},
      year={2018},
      eprint={1808.03175},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
``
6. Fasttext vectors can be downloaded from [Fastext vectors](https://fasttext.cc/docs/en/crawl-vectors.html)