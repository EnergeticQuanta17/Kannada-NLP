import datasets
from pathlib import Path

from transformers import (DataCollatorForLanguageModeling, Trainer,
			 TrainingArguments, DistilBertConfig, 
			DistilBertForMaskedLM, DistilBertTokenizerFast)


import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp


config = DistilBertConfig(vocab_size=30000)
model = xmp.MpModelWrapper(DistilBertForMaskedLM(config))
SERIAL_EXEC = xmp.MpSerialExecutor()


tokenizer = DistilBertTokenizerFast.from_pretrained("/content/Tokenizer", do_lower_case=False)


def get_tokenized_dataset():
  data_files = [str(x) for x in Path('data').glob('**/*.txt')]
  tokenized_datasets = datasets.load_dataset('text', data_files=data_files)
  
  def tokenize_function(examples):
    # Remove empty lines
    examples["text"] = [line for line in examples["text"] if len(line) > 0 and not line.isspace()]
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128,
        return_special_tokens_mask=True,
    )
  return tokenized_datasets.with_transform(tokenize_function)


def get_data_collator():
  return DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)


def map_fn(index):  
  device = xm.xla_device()
  model.to(device)
  xm.rendezvous("Model moved to device")


  # Defining arbitrary training arguments
  args = TrainingArguments(output_dir="/content/TPUCheckpoints", do_train=True, per_device_train_batch_size=32,weight_decay=0.01, 
                    num_train_epochs=3, save_total_limit=2, save_steps=500,
                    disable_tqdm=False, remove_unused_columns=False, ignore_data_skip=False)
                    
  tokenized_datasets = SERIAL_EXEC.run(get_tokenized_dataset)
  xm.rendezvous("Tokenized dataset loaded")
  data_collator = SERIAL_EXEC.run(get_data_collator)
  xm.rendezvous("DataCollator loaded")
  
  trainer = Trainer(
      model=model,
      args=args,
      train_dataset=tokenized_datasets["train"],
      tokenizer=tokenizer,
      data_collator=data_collator,
  )
  trainer.train()


if __name__ == "__main__":
  xmp.spawn(map_fn, args=(), nprocs=8, start_method='fork')
