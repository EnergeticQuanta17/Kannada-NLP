import datasets
from pathlib import Path

from transformers import DataCollatorForLanguageModeling, DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizerFast, Trainer
from transformers.training_args import TrainingArguments

# import xla.torch_xla.core.xla_model as xm
# import xla.torch_xla.distributed.parallel_loader as pl
# import xla.torch_xla.distributed.xla_multiprocessing as xmp

#


config = DistilBertConfig(vocab_size=30000)
model = DistilBertForMaskedLM(config)


tokenizer = DistilBertTokenizerFast.from_pretrained("./kanbert", do_lower_case=False)


def get_tokenized_dataset():
  data_files = [str(x) for x in Path('data').glob('**/*.txt')]
  tokenized_datasets = datasets.load_dataset('text', data_files=data_files)
  
  def tokenize_function(examples):
    
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
  model.to('cuda')
  
  trainer = Trainer(
      model=model,
      train_dataset=tokenized_datasets["train"],
      tokenizer=tokenizer,
      data_collator=data_collator,
  )
  trainer.train()