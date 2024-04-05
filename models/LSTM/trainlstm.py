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

# Constants
MODEL_PATH = "/home/hjz/544/CSCI544-FinalProject/data/mBERT/fine"
TRAIN_DATA_PATH = "/home/hjz/544/CSCI544-FinalProject/data/merge/train.parquet"
VAL_DATA_PATH = "/home/hjz/544/CSCI544-FinalProject/data/merge/dev.parquet"
TEST_DATA_PATH = "/home/hjz/544/CSCI544-FinalProject/data/merge/test.parquet"

WIKINEURAL_TAGS_LIST = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
WIKINEURAL_TAGS_TO_INT = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
WIKINEURAL_INT_TO_TAGS = {v: k for k, v in WIKINEURAL_TAGS_TO_INT.items()}

# Hyperparameters
EMBEDDING_DIM = 768
HIDDEN_DIM = 512
OUTPUT_DIM = 1024
DROPOUT = 0.33
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 60
NUM_LABELS = 9

# Load tokenizer and model
tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
bert_model = BertModel.from_pretrained(MODEL_PATH)
bert_model.eval()
bert_model.to('cuda:0')

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
                processed_embedding.append(torch.mean(torch.stack(current_embedding), dim=0))
                current_embedding.clear()
            current_embedding.append(embeddings[i])
            previous_word_idx = word_idx

    if current_embedding:
        processed_embedding.append(torch.mean(torch.stack(current_embedding), dim=0))

    return torch.stack(processed_embedding)

def collate_fn(batch):
    input_ids, label_ids = zip(*batch)
    input_ids = pad_sequence([ids.clone().detach() for ids in input_ids], batch_first=True, padding_value=0)
    label_ids = pad_sequence([ids.clone().detach() for ids in label_ids], batch_first=True, padding_value=-100)
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
        encoded_words = tokenizer(sentence_words, return_tensors='pt', is_split_into_words=True, truncation=True).to("cuda:0")
        embeddings = self.bert_model(**encoded_words)
        pooled_embeddings = pooling_embedding(encoded_words.word_ids(), embeddings.last_hidden_state[0])
        labels = torch.tensor(current_row['ner_tags'].astype(int)).to("cuda:0")
        if pooled_embeddings.shape[0] < labels.shape[0]:
            labels = labels[:pooled_embeddings.shape[0]]
        assert pooled_embeddings.shape[0] == labels.shape[0], f"pooled_embeddings shape {pooled_embeddings.shape} and labels shape {labels.shape} are not equal, index {index}"
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
class BLSTMModelLightning(pl.LightningModule):
    def __init__(self, embedding_dim, hidden_dim, output_dim, num_labels, dropout):
        super(BLSTMModelLightning, self).__init__()
        self.num_labels = num_labels
        self.criterion = nn.CrossEntropyLoss()
        
        self.transform_embeddings = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ELU(),
            nn.LayerNorm(embedding_dim * 2)
        )
        self.transform_embeddings2 = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ELU(),
            nn.LayerNorm(embedding_dim)
        )
        
        self.blstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, input_embeddings):
        transformed_embeddings = self.transform_embeddings(input_embeddings)
        transformed_embeddings = self.transform_embeddings2(transformed_embeddings)
        
        # Add residual connection
        # transformed_embeddings = transformed_embeddings + input_embeddings
        
        blstm_out, _ = self.blstm(transformed_embeddings)
        blstm_out = self.dropout(blstm_out)
        logits = self.classifier(blstm_out)
        return logits

    def training_step(self, batch, batch_idx):
        input_embeddings, labels = batch
        logits = self(input_embeddings)
        loss = self.criterion(logits.view(-1, self.num_labels), labels.view(-1))
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_embeddings, labels = batch
        logits = self(input_embeddings)
        loss = self.criterion(logits.view(-1, self.num_labels), labels.view(-1))
        preds = torch.argmax(logits, dim=2).flatten()
        mask = labels.view(-1) != -100
        preds = torch.argmax(logits, dim=2).view(-1)[mask]
        labels_flat = labels.view(-1)[mask]
        true_labels = [WIKINEURAL_INT_TO_TAGS[label.item()] for label in labels_flat]
        pred_labels = [WIKINEURAL_INT_TO_TAGS[pred.item()] for pred in preds]
        results = seqeval_metric.compute(predictions=[pred_labels], references=[true_labels])
        self.log("val_seqeval_f1", results['overall_f1'], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        input_embeddings, labels = batch
        logits = self(input_embeddings)
        loss = self.criterion(logits.view(-1, self.num_labels), labels.view(-1))
        preds = torch.argmax(logits, dim=2).flatten()
        mask = labels.view(-1) != -100
        preds = torch.argmax(logits, dim=2).view(-1)[mask]
        labels_flat = labels.view(-1)[mask]
        acc = accuracy(preds, labels_flat, task="multiclass", num_classes=self.num_labels)
        true_labels = [WIKINEURAL_INT_TO_TAGS[label.item()] for label in labels_flat]
        pred_labels = [WIKINEURAL_INT_TO_TAGS[pred.item()] for pred in preds]
        results = seqeval_metric.compute(predictions=[pred_labels], references=[true_labels])
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_seqeval_f1", results['overall_f1'], on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=LEARNING_RATE)
    def configure_gradient_clipping(self, optimizer: optim.Optimizer, gradient_clip_val: int | float | None = None, gradient_clip_algorithm: str | None = None) -> None:
        self.clip_gradients(
                    optimizer,
                    gradient_clip_val=gradient_clip_val,
                    gradient_clip_algorithm=gradient_clip_algorithm
                )
    
# Main
if __name__ == "__main__":
    # Load datasets
    train_dataset = NERDataset(TRAIN_DATA_PATH, tokenizer, bert_model)
    train_dataset, _ = random_split(train_dataset, [int(0.1 * len(train_dataset)), len(train_dataset) - int(0.1 * len(train_dataset))])
    val_dataset = NERDataset(VAL_DATA_PATH, tokenizer, bert_model)
    val_dataset, _ = random_split(val_dataset, [int(0.01 * len(val_dataset)), len(val_dataset) - int(0.01 * len(val_dataset))])
    test_dataset = NERDataset(TEST_DATA_PATH, tokenizer, bert_model)

    # Create data module
    data_module = NERDataModule(train_dataset, val_dataset, test_dataset, BATCH_SIZE)

    # Create model
    lstm_model = BLSTMModelLightning(EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_LABELS, DROPOUT)

    # Set up logger and callbacks
    logger = TensorBoardLogger("logs/", name="my_model_lstm")
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="lstm-8000v1",
        save_top_k=3,
        verbose=True,
        monitor="val_seqeval_f1",
        mode="max"
    )
    device_stats = DeviceStatsMonitor()
    seqeval_metric = load_metric("seqeval")

    # Set up trainer
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback,device_stats],
        max_epochs=NUM_EPOCHS,
        accelerator="gpu",
        devices=1,
        enable_checkpointing=True,
        enable_progress_bar=True,
        logger=logger,
        gradient_clip_val=1.0
    )

    # Train the model
    trainer.fit(lstm_model, datamodule=data_module)

    # Evaluate the model
   # trainer.test(lstm_model, datamodule=data_module, ckpt_path="checkpoints/best-checkpoint.ckpt")