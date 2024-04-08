import pytorch_lightning as pl
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, DeviceStatsMonitor
from torchmetrics.functional import accuracy
from lightning.pytorch.loggers import TensorBoardLogger
from datasets import load_metric
from transformers import BertTokenizer, BertModel, BertTokenizerFast
from torchcrf import CRF
pl.seed_everything(42,workers=True)
# Constants
MODEL_PATH = "/home/hjz/544/CSCI544-FinalProject/data/mBERT/fine"
TRAIN_DATA_PATH = "/home/hjz/544/CSCI544-FinalProject/data/merge/train.parquet"
VAL_DATA_PATH = "/home/hjz/544/CSCI544-FinalProject/data/merge/dev.parquet"
TEST_DATA_PATH = "/home/hjz/544/CSCI544-FinalProject/data/merge/test.parquet"


WIKINEURAL_TAGS_TO_INT = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3,
                          'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8, 'PAD': 9}
WIKINEURAL_INT_TO_TAGS = {v: k for k, v in WIKINEURAL_TAGS_TO_INT.items()}

# Hyperparameters
EMBEDDING_DIM = 768
HIDDEN_DIM = 512
OUTPUT_DIM = 1024
DROPOUT = 0.33
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 60
NUM_LABELS = 10

# Load tokenizer and model
tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
bert_model = BertModel.from_pretrained(MODEL_PATH)
bert_model.eval()
bert_model.to('cuda:0')
bert_model.requires_grad_(False)
# Utility functions


def pooling_embedding(tokenized_input, embeddings):
    processed_embedding = []
    current_embedding = []
    previous_word_idx = None

    for i, word_idx in enumerate(tokenized_input):
        if word_idx is None:
            continue

        if word_idx == previous_word_idx:
            current_embedding.append(embeddings[i])
        else:
            if current_embedding:
                processed_embedding.append(torch.mean(
                    torch.stack(current_embedding), dim=0))
                current_embedding.clear()
            current_embedding.append(embeddings[i])
            previous_word_idx = word_idx

    if current_embedding:
        processed_embedding.append(torch.mean(
            torch.stack(current_embedding), dim=0))

    return torch.stack(processed_embedding)


def collate_fn(batch):
    input_ids, label_ids = zip(*batch)
    input_ids = pad_sequence(
        [ids.clone().detach() for ids in input_ids], batch_first=True, padding_value=0)
    label_ids = pad_sequence(
        [ids.clone().detach() for ids in label_ids], batch_first=True, padding_value=WIKINEURAL_TAGS_TO_INT['PAD'])
    return input_ids, label_ids

# Dataset


class NERDataset(Dataset):
    def __init__(self, data_file_path, tokenizer, bert_model):
        self.raw_dataset = pd.read_parquet(data_file_path)
        self.tokenizer = tokenizer
        self.bert_model = bert_model

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, index):
        current_row = self.raw_dataset.iloc[index]
        sentence_words = current_row['tokens'].tolist()
        encoded_words = tokenizer(sentence_words, return_tensors='pt',
                                  is_split_into_words=True, truncation=True).to("cuda:0")
        embeddings = self.bert_model(**encoded_words)
        pooled_embeddings = pooling_embedding(
            encoded_words.word_ids(), embeddings.last_hidden_state[0])
        labels = torch.tensor(current_row['ner_tags'].astype(int)).to("cuda:0")
        if pooled_embeddings.shape[0] < labels.shape[0]:
            labels = labels[:pooled_embeddings.shape[0]]
        assert pooled_embeddings.shape[0] == labels.shape[
            0], f"pooled_embeddings shape {pooled_embeddings.shape} and labels shape {labels.shape} are not equal, index {index}"
        return pooled_embeddings, labels

# Data module


class NERDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, test_dataset, batch_size):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)

# Model


class AttentionCRFModel(pl.LightningModule):
    def __init__(self, embedding_dim, hidden_dim, output_dim, num_labels, dropout, num_attention_layers=5):
        super(AttentionCRFModel, self).__init__()
        self.predictions = []
        self.num_labels = num_labels
        self.criterion = nn.CrossEntropyLoss()
        self.transform_embeddings = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ELU(),
            nn.LayerNorm(hidden_dim)
        )
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads=8) for _ in range(num_attention_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_embeddings, labels=None):
        transformed_embeddings = self.transform_embeddings(input_embeddings)
        attention_out = transformed_embeddings.transpose(0, 1)  # Shape: (seq_len, batch_size, hidden_dim)

        for attention_layer in self.attention_layers:
            attention_out, _ = attention_layer(attention_out, attention_out, attention_out)

        attention_out = attention_out.transpose(0, 1)  # Shape: (batch_size, seq_len, hidden_dim)
        attention_out = self.dropout(attention_out)
        emissions = self.classifier(attention_out)

        if labels is not None:
            mask = (labels != WIKINEURAL_TAGS_TO_INT['PAD'])
            loss = -self.crf(emissions, labels, mask=mask.byte())
            return loss
        else:
            decoded_sequence = self.crf.decode(emissions)
            return decoded_sequence

    def training_step(self, batch, batch_idx):
        input_embeddings, labels = batch
        loss = self(input_embeddings, labels)
        self.log("train_loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_embeddings, labels = batch
        loss = self(input_embeddings, labels)
        decoded_labels = self(input_embeddings)

        mask = labels != WIKINEURAL_TAGS_TO_INT['PAD']
        true_labels = [
            [WIKINEURAL_INT_TO_TAGS[label.item()]
            for label, m in zip(label_seq, mask_seq) if m]
            for label_seq, mask_seq in zip(labels, mask)
        ]

        pred_labels = [
            [WIKINEURAL_INT_TO_TAGS[label]
            for label, m in zip(label_seq, mask_seq) if m]
            for label_seq, mask_seq in zip(decoded_labels, mask)
        ]
        results = seqeval_metric.compute(
            predictions=pred_labels, references=true_labels)

        self.log("val_seqeval_f1", results['overall_f1'],
                 on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        input_embeddings, labels = batch
        loss = self(input_embeddings, labels)
        decoded_labels = self(input_embeddings)

        mask = labels != WIKINEURAL_TAGS_TO_INT['PAD']
        true_labels = [
            [WIKINEURAL_INT_TO_TAGS[label.item()]
            for label, m in zip(label_seq, mask_seq) if m]
            for label_seq, mask_seq in zip(labels, mask)
        ]

        pred_labels = [
            [WIKINEURAL_INT_TO_TAGS[label]
            for label, m in zip(label_seq, mask_seq) if m]
            for label_seq, mask_seq in zip(decoded_labels, mask)
        ]
        self.predictions.extend(pred_labels)
        results = seqeval_metric.compute(
            predictions=pred_labels, references=true_labels)

        self.log("test_loss", loss, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log("test_seqeval_f1", results['overall_f1'],
                 on_step=False, on_epoch=True, prog_bar=True)
    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=LEARNING_RATE)

    def configure_gradient_clipping(self, optimizer: optim.Optimizer, gradient_clip_val: int | float | None = None, gradient_clip_algorithm: str | None = None) -> None:
        self.clip_gradients(
            optimizer,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm
        )
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        input_embeddings, labels = batch
        decoded_labels = self(input_embeddings)

        mask = labels != WIKINEURAL_TAGS_TO_INT['PAD']
        pred_labels = [
            [WIKINEURAL_INT_TO_TAGS[label]
            for label, m in zip(label_seq, mask_seq) if m]
            for label_seq, mask_seq in zip(decoded_labels, mask)
        ]

        # Get the original tokens from the dataset used by the dataloader
        dataset = self.trainer.predict_dataloaders.dataset
        sentence_words = dataset.raw_dataset.iloc[batch_idx]['tokens'].tolist()

        # Truncate the tokens if necessary
        if len(sentence_words) > len(pred_labels[0]):
            sentence_words = sentence_words[:len(pred_labels[0])]

        return {"tokens": sentence_words, "predictions": pred_labels[0]}

# Main
if __name__ == "__main__":
    # Load datasets
    train_dataset = NERDataset(TRAIN_DATA_PATH, tokenizer, bert_model)
    train_dataset, _ = random_split(train_dataset, [int(
        0.1 * len(train_dataset)), len(train_dataset) - int(0.1 * len(train_dataset))])
    val_dataset = NERDataset(VAL_DATA_PATH, tokenizer, bert_model)
    val_dataset, _ = random_split(val_dataset, [int(
        0.1 * len(val_dataset)), len(val_dataset) - int(0.1 * len(val_dataset))])
    # val_dataset = torch.utils.data.Subset(val_dataset, range(1000))
    test_dataset = NERDataset(TEST_DATA_PATH, tokenizer, bert_model)
    # test_dataset, _ = random_split(test_dataset, [int(
    #     0.1 * len(test_dataset)), len(test_dataset) - int(0.1 * len(test_dataset))])
    # Create data module
    data_module = NERDataModule(
        train_dataset, val_dataset, test_dataset, BATCH_SIZE)

    # Create model
    # Create model
    attention_crf_model = AttentionCRFModel(EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_LABELS, DROPOUT)

    # Set up logger and callbacks
    logger = TensorBoardLogger("logs/", name="my_model_lstm")
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/attention_crf",
        filename="attention-crf-8000",
        save_top_k=3,
        verbose=True,
        monitor="val_seqeval_f1",
        mode="max"
    )
    device_stats = DeviceStatsMonitor()
    seqeval_metric = load_metric("seqeval")

    # Set up trainer
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, device_stats],
        max_epochs=NUM_EPOCHS,
        accelerator="gpu",
        devices=1,
        enable_checkpointing=True,
        enable_progress_bar=True,
        logger=logger,
        gradient_clip_val=1.0,
    )

    # Train the model
    trainer.fit(attention_crf_model, datamodule=data_module,)

    # Evaluate the model
    #trainer.test(attention_crf_model, datamodule=data_module, ckpt_path="/home/hjz/544/CSCI544-FinalProject/models/LSTM/checkpoints/lstm-crf-full-v4.ckpt")
    
    
    
    # # aggregation
    # data_module = NERDataModule(
    #     train_dataset, val_dataset, test_dataset, 1)
    # #print(seqeval_metric.compute(predictions=all_pred_labels, references=all_true_labels))
    # current_predictions=trainer.predict(attention_crf_model, dataloaders=data_module.train_dataloader(), ckpt_path="/home/hjz/544/CSCI544-FinalProject/models/LSTM/checkpoints/lstm-crf-full-final.ckpt")
    # current_df = pd.DataFrame(current_predictions)
    # current_df.to_csv("BiLSTM_CRF_train.csv")
    # # Predict on the validation dataset
    # current_predictions = trainer.predict(attention_crf_model, dataloaders=data_module.val_dataloader(), ckpt_path="/home/hjz/544/CSCI544-FinalProject/models/LSTM/checkpoints/lstm-crf-full-final.ckpt")
    # current_df = pd.DataFrame(current_predictions)
    # current_df.to_csv("BiLSTM_CRF_dev.csv")
    # # Predict on the testing dataset
    # current_predictions = trainer.predict(attention_crf_model, dataloaders=data_module.test_dataloader(), ckpt_path="/home/hjz/544/CSCI544-FinalProject/models/LSTM/checkpoints/lstm-crf-full-final.ckpt")
    # current_df = pd.DataFrame(current_predictions)
    # current_df.to_csv("BiLSTM_CRF_test.csv")
    
    
    