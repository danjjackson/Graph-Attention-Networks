## PyTorch
import torch
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

import jax
from jax import random

from flax.training import train_state, checkpoints
import optax

import numpy as np
import os
from tqdm.auto import tqdm

from JAX_GAT import GATModel
from utils import data_loader, dataset, CHECKPOINT_PATH

# Seeding for random operations
main_rng = random.PRNGKey(42)

class TrainerModule:

    def __init__(self, model_name, graph, lr, seed, **model_kwargs):
        """

        :param model_name: Name of the model for saving and checkpointing
        :param exmp_batch: Example batch to initialise the model
        :param lr: Learning rate
        :param model_kwargs:
        """

        self.model_name = model_name
        self.lr = lr
        self.seed = seed
        self.model = GATModel(**model_kwargs)

        self.log_dir = os.path.join(CHECKPOINT_PATH, self.model_name)
        self.logger = SummaryWriter(log_dir = self.log_dir)

        self.create_functions()

        self.init_model(graph)

    def create_functions(self):

        def calculate_loss(params, rng, data, mode='train'):
            x = data.x
            rng, dropout_apply_rng = random.split(rng)
            if mode == "train":
                mask = data.train_mask
            elif mode == "val":
                mask = data.val_mask
            elif mode == "test":
                mask = data.test_mask
            else:
                assert False, f"Unknown forward mode: {mode}"

            mask = mask.numpy()

            logits = self.model.apply(
                {'params': params},
                x,
                rngs={'dropout': dropout_apply_rng}
            )
            masked_logits = logits[mask]
            masked_y = data.y[mask]
            loss = optax.softmax_cross_entropy(masked_logits, masked_y)
            acc = (x[mask].argmax(dim=-1) == data.y[
                mask]).sum().float() / mask.sum()
            return loss, acc, rng

        # Training function
        def train_step(state, rng, batch):
            loss_fn = lambda params: calculate_loss(params, rng, batch, mode='train')
            loss, acc, rng, grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            state = state.apply(grads=grads)
            return state, rng, loss, acc
        self.train_step = train_step

        # Evaluation function
        def eval_step(state, rng, batch):
            loss, acc, rng = calculate_loss(state.params, rng, batch, mode='val')
            return acc, rng
        self.eval_step = eval_step

    def init_model(self, graph):
        self.rng = random.PRNGKey(self.seed)
        self.rng, init_rng, dropout_init_rng = random.split(self.rng, 3)
        params = self.model.init({'params': init_rng, 'dropout': dropout_init_rng}, graph)['params']
        optimiser = optax.adam(self.lr)

        self.state = train_state.TrainState.create(apply_fn=self.model.apply, params=params, tx=optimiser)

    def train_epoch(self, train_loader, epoch):
        accs, losses = [], []
        for batch in tqdm(train_loader, desc='Training', leave=False):
            x = batch.x
            self.state, self.rng, loss, accuracy = self.train_step(self.state, self.rng, batch)
            losses.append(loss)
            accs.append(accuracy)
        avg_loss = np.stack(jax.device_get(losses)).mean()
        avg_acc = np.stack(jax.device_get(accs)).mean()
        self.logger.add_scalar('train/loss', avg_loss, global_step=epoch)
        self.logger.add_scalar('train/accuracy', avg_acc, global_step=epoch)

    def train_model(self, train_loader, val_loader, num_epochs=500):
        best_acc = 0.0
        for epoch in tqdm(range(num_epochs)):
            self.train_epoch(train_loader, epoch=epoch)
            if epoch % 5 == 0:
                eval_acc = self.eval_model(val_loader)
                self.logger.add_scalar('val/accuracy', eval_acc, global_step=epoch)
                if eval_acc >= best_acc:
                    best_acc = eval_acc
                    self.save_model(step=epoch)
                self.logger.flush()

    def eval_model(self, data_loader):
        # Test model on all data points of a data loader and return avg accuracy
        correct_class, count = 0, 0
        for batch in data_loader:
            acc, self.rng = self.eval_step(self.state, self.rng, batch)
            correct_class += acc * batch[0].shape[0]
            count += batch[0].shape[0]
        eval_acc = (correct_class / count).item()
        return eval_acc

    def save_model(self, step=0):
        # Save current model at certain training iteration
        checkpoints.save_checkpoint(ckpt_dir=self.log_dir, target=self.state.params, step=step)

    def load_model(self, pretrained=False):
        # Load model. We use different checkpoint for the pretrained model
        if not pretrained:
            params = checkpoints.restore_checkpoint(ckpt_dir=self.log_dir, target=self.state.params)
        else:
            params = checkpoints.restore_checkpoint(ckpt_dir=os.path.join(CHECKPOINT_PATH, f'{self.model_name}.ckpt'), target=self.state.params)
        self.state = train_state.TrainState.create(apply_fn=self.model.apply, params=params, tx=self.state.tx)

    def checkpoint_exists(self):
        # Check whether a pretrained model exist for this Transformer
        return os.path.isfile(os.path.join(CHECKPOINT_PATH, f'{self.model_name}.ckpt'))

def train_model(model_name, num_epochs, **model_kwargs):
    root_dir = os.path.join(CHECKPOINT_PATH, 'JAX', model_name)
    os.makedirs(root_dir, exist_ok=True)


    trainer = TrainerModule(
        model_name=model_name,
        graph=dataset.data.x,
        lr=1e-4,
        seed=42,
        num_nodes=dataset.data.num_nodes,
        input_size=dataset.num_node_features,
        num_classes=dataset.num_classes,
        edge_index = dataset.data.edge_index,
        **model_kwargs
    )

    if not trainer.checkpoint_exists():  # Skip training if pretrained model exists
        trainer.train_model(
            data_loader,
            data_loader,
            num_epochs=num_epochs)
        trainer.load_model()
    else:
        trainer.load_model(pretrained=True)

    train_acc = trainer.eval_model(data_loader)
    val_acc = trainer.eval_model(data_loader)
    test_acc = trainer.eval_model(data_loader)

    # Bind parameters to model for easier inference
    trainer.model_bd = trainer.model.bind({'params': trainer.state.params})
    return trainer, {'train_acc': train_acc, 'val_acc': val_acc,
                     'test_acc': test_acc}

if __name__ == "__main__":
    trainer, results = train_model(
        model_name='JAX_GAT1',
        num_epochs=100,
        num_heads=4,
        c_hidden=24,
        dropout=0.4,
        bias=True
    )

    print(f"Train accuracy: {(100.0 * results['train_acc']):4.2f}%")
    print(f"Val accuracy:   {(100.0 * results['val_acc']):4.2f}%")
    print(f"Test accuracy:  {(100.0 * results['test_acc']):4.2f}%")