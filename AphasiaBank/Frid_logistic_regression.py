'''
Use Best transcript from wavlm-Transformer
-extract BERT features
'''

import torch
import torch.nn as nn
from transformers import BertModel,DataCollatorWithPadding,AutoTokenizer
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset,Dataset,DatasetDict
from transformers import AdamW
import torch.nn.functional as F


class ParaphasiaDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        self.max_length = 128

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # an encoding can have keys such as input_ids and attention_mask
        # item is a dictionary which has the same keys as the encoding has
        # and the values are the idxth value of the corresponding key (in PyTorch's tensor format)
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

        # pad_len = self.max_length - self.labels[idx]
        item['labels'] = torch.tensor(self.labels[idx])
        return item

class PD_LogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased', return_dict=False)
        self.dropout1 = nn.Dropout()
        self.linear1 = nn.Linear(in_features=768, out_features=256)
        self.linear2 = nn.Linear(in_features=256, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,tokens, attention_mask):
        _, cls_out = self.bert(input_ids=tokens, attention_mask=attention_mask)
        x = F.relu(self.linear1(cls_out))
        x = self.dropout1(x)
        x = F.sigmoid(self.linear2(x))
        return x

def align_and_pad_labels(labels, input_ids, pad_token_label=-100):
    max_length = input_ids.shape[1]
    padded_labels = []

    for label, ids in zip(labels, input_ids):
        # Aligning labels with word pieces, if necessary
        # If your tokenizer splits words into multiple tokens, you'll need to decide how to label these sub-tokens
        # For simplicity, let's assume each word is tokenized into a single token

        # Padding the label if it's shorter than the max length
        label.extend([pad_token_label] * (max_length - len(label)))
        padded_labels.append(label[:max_length])  # Truncate if necessary

    return padded_labels


def align_labels_with_tokens(sentences, word_labels, tokenizer):
    tokenized_inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, is_split_into_words=True)
    aligned_labels = []

    for i, label in enumerate(word_labels):
        print(f"label: {label}")
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Get word id for each token
        print(f"word_ids: {word_ids}")
        token_labels = []

        for word_id in word_ids:
            if word_id is None:  # Special tokens have no word id
                token_labels.append(-100)  # Use -100 to ignore these tokens during loss calculation
            else:
                token_labels.append(label[word_id])

        aligned_labels.append(token_labels)

    return tokenized_inputs, aligned_labels

def data_prep(csv_path, tokenizer, batch_size, shuffle):
    # Create train, dev, and test sets
    PARA_DICT = {'C':0, 'P':1}
    df = pd.read_csv(csv_path)
    labels_list = [x.split("/")[1] for row in df['aug_para'] for x in row.split()]
    text_list = [x.split("/")[0] for row in df['aug_para'] for x in row.split()]
    text = " ".join(text_list)
    df['words'] = split_words = df['aug_para'].apply(lambda s: " ".join([word.split('/')[0] for word in s.split()]))
    df['labels'] = split_words = df['aug_para'].apply(lambda s: [PARA_DICT[word.split('/')[1]] for word in s.split()])

    tokenized_inputs, aligned_labels = align_labels_with_tokens(df['words'].to_list(), df['labels'].to_list(), tokenizer)

    print(tokenized_inputs)
    exit()
    # encodings = tokenizer(df['words'].to_list(), truncation=True, padding=True, return_tensors="pt")
    # labels = df['labels'].to_list()
    # padded_labels = align_and_pad_labels(labels, encodings['input_ids'])
    print(encodings['input_ids'][0])
    print(padded_labels[0])
    exit()
    
    dataset = ParaphasiaDataset(encodings,labels)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collator)



def train_step(model,device, train_loader):
    model.train()
    loss_fn = nn.BCEWithLogitsLoss()
    for i, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        print(f"labels: {labels}")
        exit()
        outputs = torch.flatten(model(tokens=input_ids, attention_mask=attention_mask))
        print(f"outputs: {outputs}")
        print(f"labels: {labels}")
        exit()

def train_model(model, train_loader, dev_loader, epochs):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Device used: {}.".format(device))
    model.to(device)

    for epoch in range(1,epochs+1):
        model = train_step(model,device,train_loader)
        
        
   

if __name__ == "__main__":
    DATA_ROOT = "/home/mkperez/speechbrain/AphasiaBank/data/Fridriksson_para_best_Word"

    TRAIN_FLAG = True
    EVAL_FLAG = True
    OUTPUT_NEURONS=500
    PARA_TYPE=['pn','p','n'][0]
    W2V_MODEL=['wavlm-large', 'wav2vec2-large-960h-lv60-self','hubert-large-ls960-ft'][2]
    EXP_DIR = f"results/Table2/ASR-PT_PD-FT/{W2V_MODEL.split('-')[0]}/{PARA_TYPE}-{OUTPUT_NEURONS}-unigram"

    
    model_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    BATCH_SIZE=16
    LR=1e-3
    EPOCHS=20

    for i in range(1,2):
        data_fold_dir = f"{DATA_ROOT}/Fold_{i}"

        # Prep data
        train_loader = data_prep(f"{data_fold_dir}/train_{PARA_TYPE}.csv", tokenizer, BATCH_SIZE, True)
        dev_loader = data_prep(f"{data_fold_dir}/dev_{PARA_TYPE}.csv", tokenizer, BATCH_SIZE, False)

        # TODO: change this to load text from ASR
        test_loader = data_prep(f"{data_fold_dir}/dev_{PARA_TYPE}.csv", tokenizer, BATCH_SIZE, False)

        # init model
        model = PD_LogisticRegression()
        optimizer = AdamW(model.parameters(), lr=LR)

        # Train model
        model = train_model(model, train_loader, dev_loader, EPOCHS)