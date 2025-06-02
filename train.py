from datasets import load_dataset
from transformers import (
    BertConfig, BertForMaskedLM, BertTokenizerFast,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
import os



checkpoint_path = "/content/drive/MyDrive/rq2_thesis/bert-hindi-part-5gb-validation-loss/checkpoint-45000"
model = BertForMaskedLM.from_pretrained(checkpoint_path)


train_size = 0.9
split = train_dataset.train_test_split(train_size=train_size)
train_dataset_split = split['train']
eval_dataset_split = split['test']


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir="/content/drive/MyDrive/rq2_thesis/bert-hindi-part-5gb-validation-loss",
    overwrite_output_dir=False, 
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=1,
    eval_strategy="steps",
    eval_steps=15000,
    logging_steps=1000,
    save_steps=15000,
    save_total_limit=5,
    fp16=True,
    warmup_ratio=0.05,
    logging_dir="/content/drive/MyDrive/rq2_thesis/bert-hindi-part-5gb-validation-loss/logs"
)


trainer = Trainer(
    model=model,  
    args=training_args,
    train_dataset=train_dataset_split,
    eval_dataset=eval_dataset_split,
    tokenizer=tokenizer,
    data_collator=data_collator,
)


trainer.train(resume_from_checkpoint=checkpoint_path)


trainer.save_model("/content/drive/MyDrive/rq2_thesis/bert-hindi-part-5gb-validation-loss")
tokenizer.save_pretrained("/content/drive/MyDrive/rq2_thesis/bert-hindi-part-5gb-validation-loss")
