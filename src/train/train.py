import torch
from tqdm import tqdm
import pandas as pd
import os

class ModelTrainer:
    def __init__(self, 
                 model, 
                 train_loader, 
                 val_loader, 
                 test_loader, 
                 optimizer,
                 loss_fn,
                 scheduler,
                 device,
                 csv_log_path,
                 checkpoints_path):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.csv_log_path = csv_log_path
        self.checkpoints_path = checkpoints_path

    def train_step(self):
        self.model.train()
        train_loss = 0.0
        for data in self.train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            preds = self.model(data)
            loss = self.loss_fn(preds, data.y)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
        return train_loss / len(self.train_loader)

    def test_step(self, loader):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)
                preds = self.model(data)
                loss = self.loss_fn(preds, data.y)
                test_loss += loss.item()
        return test_loss/len(loader)

    def fit(self, epochs):
        os.makedirs(os.path.dirname(self.csv_log_path), exist_ok=True)
        os.makedirs(self.checkpoints_path, exist_ok=True)
        results = {
            'train_loss': [],
            'val_loss': [],
            'test_loss': []
        }
        for epoch in tqdm(range(1, epochs+1)):
            lr = self.scheduler.optimizer.param_groups[0]['lr']
            train_loss = self.train_step()
            val_loss = self.test_step(self.val_loader)
            test_loss = self.test_step(self.test_loader)
            path = f'{self.checkpoints_path}/checkpoint_{epoch}.pt'
            checkpoint = self.save_model(path, epoch, val_loss, test_loss)
            self.scheduler.step(val_loss)
            results['train_loss'].append(train_loss)
            results['val_loss'].append(val_loss)
            results['test_loss'].append(test_loss)
            print(f'Epoch: {epoch:03d}, LR: {lr:7f}, Loss: {train_loss:.7f}, '
              f'Val loss: {val_loss:.7f}, Test loss: {test_loss:.7f}')
            
        train_log = pd.DataFrame(results)
        train_log.to_csv(self.csv_log_path)
        return train_log

    def save_model(self, path, epoch, val_loss, test_loss):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved at epoch {epoch} with val_loss: {val_loss:.7f} to {path}")

    def load_model(self, path):
         checkpoint = torch.load(path, map_location=self.device)
         return checkpoint