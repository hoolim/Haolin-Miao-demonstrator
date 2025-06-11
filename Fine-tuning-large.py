import torch
from datasets import load_dataset, DatasetDict, Audio
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    WhisperFeatureExtractor,
    WhisperTokenizer,
)
import jiwer 


MODEL_NAME = "openai/whisper-large-v3"
MODEL_OUTPUT_DIR = "./whisper-large-v3-cantonese-english-finetuned" 
METADATA_PATH = "./metadata.csv" 



raw_dataset = DatasetDict()
raw_dataset['train'] = load_dataset('csv', data_files=METADATA_PATH, split='train[:95%]')
raw_dataset['test'] = load_dataset('csv', data_files=METADATA_PATH, split='train[95%:]')

print("Dataset structure:", raw_dataset)
print("A sample from training set:", raw_dataset["train"][0])



feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME) 
processor = WhisperProcessor.from_pretrained(MODEL_NAME) 



raw_dataset = raw_dataset.cast_column("audio", Audio(sampling_rate=16000))


def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    
    
    sentence = batch["sentence"]
    lang_prompt = "<|zh|en|> " + sentence
    batch["labels"] = tokenizer(lang_prompt).input_ids
    return batch


vectorized_dataset = raw_dataset.map(prepare_dataset, remove_columns=raw_dataset.column_names["train"], num_proc=1)



class DataCollatorSpeechSeq2SeqWithPadding:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
            
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

import jiwer

transform = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip()
])

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    
    
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    valid_pred = []
    valid_label = []

    for ref, hyp in zip(label_str, pred_str):
        try:
            ref_t = transform(ref)
            hyp_t = transform(hyp)

            if len(ref_t.split()) > 0 and len(hyp_t.split()) > 0:
                valid_label.append(ref_t)
                valid_pred.append(hyp_t)
        except Exception as e:
            
            continue

    
    if not valid_label:
        return {"wer": None, "cer": None}

    wer = 100 * jiwer.wer(valid_label, valid_pred)
    cer = 100 * jiwer.cer(valid_label, valid_pred)


    return {"wer": wer, "cer": cer}


model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)


model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

training_args = Seq2SeqTrainingArguments(
    output_dir=MODEL_OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=True,
    eval_strategy="steps",  
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=4000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
)


trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=vectorized_dataset["train"],
    eval_dataset=vectorized_dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

print("--- Starting Multilingual Code-Mixing Fine-tuning ---")
trainer.train()


print("--- Training Finished ---")
print("Saving final model and processor...")
trainer.save_model(MODEL_OUTPUT_DIR)
processor.save_pretrained(MODEL_OUTPUT_DIR)
print(f"Model and processor saved to {MODEL_OUTPUT_DIR}")