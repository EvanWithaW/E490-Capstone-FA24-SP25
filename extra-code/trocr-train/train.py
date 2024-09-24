import torch
from torch.utils.data import DataLoader
from transformers import VisionEncoderDecoderModel
from transformers import TrOCRProcessor
import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
from datasets import load_metric
from transformers import AdamW


def compute_cer(pred_ids, label_ids):
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return cer

class LPRDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=128):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get file name + text 
        file_name = self.df['file_name'][idx]
        text = self.df['text'][idx]
        # prepare image (i.e. resize + normalize)
        image = Image.open(os.path.join(self.root_dir, file_name)).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(text, 
                                          padding="max_length", 
                                          max_length=self.max_target_length).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding




# slate_dataset_path = '/N/slate/jdmckean/TRAIN-LP/train/images'
slate_dataset_path = r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\TRAIN-CHAR\trocr-train"
df = pd.read_csv('train-set.csv', usecols=["text", "file_name"])
df.columns = ["text", "file_name"]
df['file_name'] = df['file_name'].apply(lambda x: x + '.png')	#add 'png' to all file names
print(df.head())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-printed")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-printed")
# model = VisionEncoderDecoderModel.from_pretrained("train.pt")
# processor = TrOCRProcessor.from_pretrained("processor.pt")
model.to(device)

# test train spilit with random seed to get the same split everytime
train_df, test_df = train_test_split(df, test_size=0.2, random_state = 1)
# we reset the indices to start from zero
train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

train_dataset = LPRDataset(root_dir=slate_dataset_path,
                           df=train_df,
                           processor=processor)
eval_dataset = LPRDataset(root_dir=slate_dataset_path,
                           df=test_df,
                           processor=processor)
     


print("Number of training examples:", len(train_dataset))
print("Number of validation examples:", len(eval_dataset))
# encoding = train_dataset[0]
# for k,v in encoding.items():
#       print(k, v.shape)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=32)

# set special tokens used for creating the decoder_input_ids from the labels
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
# make sure vocab size is set correctly
model.config.vocab_size = model.config.decoder.vocab_size

# set beam search parameters
model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.max_length = 64
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4

cer_metric = load_metric("cer")
optimizer = AdamW(model.parameters(), lr=5e-5)

for epoch in range(45):  # loop over the dataset multiple times
    # train
    print("epock: ", epoch)
    model.train()
    train_loss = 0.0
    for batch in train_dataloader:
        # get the inputs
        for k,v in batch.items():
            batch[k] = v.to(device)
        # forward + backward + optimize
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.item()
        break

    print(f"Loss after epoch {epoch}:", train_loss/len(train_dataloader))
    
            # evaluate
    model.eval()
    valid_cer = 0.0
    with torch.no_grad():
        for batch in eval_dataloader:
            # run batch generation
            outputs = model.generate(batch["pixel_values"].to(device))
            # compute metrics
            cer = compute_cer(pred_ids=outputs, label_ids=batch["labels"])
            valid_cer += cer 
            break
    print("Validation CER:", valid_cer / len(eval_dataloader))

model.save_pretrained("./model.pt")
processor.save_pretrained("./processor.pt")
