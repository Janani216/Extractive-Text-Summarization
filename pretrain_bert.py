import torch
import pandas as pd
from transformers import BertTokenizer, AdamW
from transformers import BertForMaskedLM
import numpy as np
from statistics import mean

from reader import DataReader


def read_dataset():
    reader = DataReader('cnn_data.csv')
    return reader.load_dataset_from_csv()


def split_into_chunks(documents, chunk_size):
    np_documents = np.array(list(documents.values()))
    return np.array_split(np_documents, chunk_size)


def divide_chunks(documents, n):
    for i in range(0, len(documents), n):
        yield documents[i:i + n]


if __name__ == '__main__':

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    model = BertForMaskedLM.from_pretrained('bert-base-uncased')

    model.to(device)
    model.train()

    lr = 1e-2

    optimizer = AdamW(model.parameters(), lr=lr, correct_bias=False)
    max_len = 82
    chunk_size = 64
    epochs = 5

    documents, summaries = read_dataset()
    print('Original ', len(documents))
    document_splits = list(divide_chunks(documents, chunk_size))
    print(len(document_splits))

    for epoch in range(epochs):
        print(epoch)
        epoch_losses = []
        for split in document_splits:
            try:
                encoded_dict = tokenizer.batch_encode_plus(
                    split,  # Sentence to encode.
                    add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                    max_length=max_len,  # Pad & truncate all sentences.
                    pad_to_max_length=True,
                    truncation=True,
                    return_attention_mask=True,  # Construct attn. masks.
                    return_tensors='pt',  # Return pytorch tensors.
                )
                input_ids = encoded_dict['input_ids'].to(device)
                loss_value, scores = model(input_ids, labels=input_ids,
                                           return_dict=False)
                epoch_losses.append(loss_value.item())
                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            except Exception as e:
                print(str(e))
                continue
        print('Avg epoch loss ', epoch, 'Loss: ', mean(epoch_losses))
        path = 'bert_model/'
        model.save_pretrained(path + "Fine_Tuned_Bert_5Epochs")
