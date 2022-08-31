import os

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar

from torch_GAT import GATModel

from utils import data_loader, dataset, CHECKPOINT_PATH, BATCH_SIZE

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

class GATTrainer(pl.LightningModule):
    def __init__(self,  **model_kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = GATModel(**model_kwargs)
        self.loss_module = nn.CrossEntropyLoss()

    def forward(self, data, mode='train'):
        x, edge_index = data.x, data.edge_index

        x = self.model(x, edge_index)

        # Only calculate the loss on the nodes corresponding to the mask
        if mode == "train":
            mask = data.train_mask
        elif mode == "val":
            mask = data.val_mask
        elif mode == "test":
            mask = data.test_mask
        else:
            assert False, f"Unknown forward mode: {mode}"

        loss = self.loss_module(x[mask], data.y[mask])
        acc = (x[mask].argmax(dim=-1) == data.y[
            mask]).sum().float() / mask.sum()
        return loss, acc

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=0.005,
            weight_decay=0.0005
        )

        return optimizer

    def training_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, mode="train")
        self.log('train_loss', loss, batch_size=BATCH_SIZE)
        self.log('train_acc', acc, batch_size=BATCH_SIZE)
        return loss

    def validation_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="val")
        self.log('val_acc', acc, batch_size=BATCH_SIZE)

    def test_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="test")
        self.log('test_acc', acc, batch_size=BATCH_SIZE)


def train_model(model_name, max_epochs, **model_kwargs):
    pl.seed_everything(42)

    # Create a PyTorch Lightning trainer with the generation callback
    root_dir = os.path.join(CHECKPOINT_PATH, "NodeLevel" + model_name)
    os.makedirs(root_dir, exist_ok=True)

    trainer = pl.Trainer(
        default_root_dir=root_dir,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
            TQDMProgressBar(refresh_rate=0)
        ],
        gpus=1 if str(device).startswith("cuda") else 0,
        max_epochs=max_epochs,
        log_every_n_steps=1,
    )

    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    model = GATTrainer(
        dataset=dataset,
        **model_kwargs
    )

    trainer.fit(model, data_loader, data_loader)
    best_model = GATTrainer.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path)

    # Test best model on the test set
    test_result = trainer.test(model,data_loader, verbose=False)
    batch = next(iter(data_loader))
    batch = batch.to(model.device)
    _, train_acc = best_model.forward(batch, mode="train")
    _, val_acc = best_model.forward(batch, mode="val")
    result = {"train": train_acc,
              "val": val_acc,
              "test": test_result[0]['test_acc']}
    return model, result

# Small function for printing the test scores
def print_results(result_dict):
    if "train" in result_dict:
        print(f"Train accuracy: {(100.0*result_dict['train']):4.2f}%")
    if "val" in result_dict:
        print(f"Val accuracy:   {(100.0*result_dict['val']):4.2f}%")
    print(f"Test accuracy:  {(100.0*result_dict['test']):4.2f}%")

if __name__ == "__main__":
    model, results = train_model(
        model_name='GAT1',
        max_epochs=100,
        num_layers=2,
        num_heads=[8, 1],
        c_hidden=64,
        dropout=0.1
    )
    print_results(results)