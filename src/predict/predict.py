import torch
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from src.model.model import MEGNETModel, CGCNNModel
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt

class Prediction:
    def __init__(self,
                 model,
                 train_loader,
                 val_loader,
                 test_loader,
                 device,
                 path_predictions,
                 ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.path_predictions = path_predictions

    def predict(self):
        train_y, train_predict = self._get_predictions(self.train_loader)
        val_y, val_predict = self._get_predictions(self.val_loader)
        test_y, test_predict = self._get_predictions(self.test_loader)

        metrics = {
            'train' : self._get_evaluation_metrics(train_y, train_predict),
            'val' : self._get_evaluation_metrics(val_y, val_predict),
            'test' : self._get_evaluation_metrics(test_y, test_predict)
        }
        
        train = {'train_y' : train_y,
                 'train_predict': train_predict}
        val = {'val_y': val_y,
               'val_predict': val_predict}
        test = {'test_y': test_y,
                'test_predict': test_predict}
    
        df_train = pd.DataFrame(train)
        df_val = pd.DataFrame(val)
        df_test = pd.DataFrame(test)

        os.makedirs(os.path.dirname(self.path_predictions), exist_ok=True)
        df_train.to_csv(self.path_predictions.replace('.csv', '_train.csv'), index=False)
        df_val.to_csv(self.path_predictions.replace('.csv', '_val.csv'), index=False)
        df_test.to_csv(self.path_predictions.replace('.csv', '_test.csv'), index=False)

        for split, metric in metrics.items():
            print(f"{split.capitalize()} Metrics:")
            print(f"  MSE: {metric['mse']:.4f}")
            print(f"  MAE: {metric['mae']:.4f}")
            print(f"  R2: {metric['r2_score']:.4f}")

        return {'train':df_train, 'val':df_val, 'test': df_test}, metrics
    
    def _get_predictions(self, dataloader):
        self.model.eval()
        y_reals = []
        y_preds = []
        with torch.no_grad():
            for data in dataloader:
                data = data.to(self.device)
                pred = self.model(data)
                y_reals.append(data.y.cpu())
                y_preds.append(pred.cpu())
        y_real = torch.cat(y_reals).squeeze().numpy()
        y_pred = torch.cat(y_preds).squeeze().numpy()
        return y_real, y_pred
    
    def _get_evaluation_metrics(self, y_real, y_pred):
        mae = mean_absolute_error(y_real, y_pred)
        mse = mean_squared_error(y_real, y_pred)
        r2 = r2_score(y_real, y_pred)
        return {'mae':mae, 'mse':mse, 'r2_score':r2}
    

class EnsembleLearning:
    def __init__(self, models, device, checkpoints_dir):
        self.models = models
        self.device = device
        self.checkpoints_dir = checkpoints_dir

    def ensemble_predict(self, data_loader, top_k):
        predictions = []
        with torch.no_grad():
            for data in data_loader:
                data = data.to(self.device)
                preds = [model(data) for model in self.models[:top_k]]
                avg_pred = torch.mean(torch.stack(preds), dim=0)
                predictions.append(avg_pred)
        return torch.cat(predictions, dim=0).squeeze().cpu().numpy()
    
    def plot_evaluation(self, data_loader, top_k):
        mae_errors = []
        y_real = self._get_y_real_labels(data_loader)
        for k in range(1, top_k+1):
            preds = self.ensemble_predict(data_loader, k)
            mae = mean_absolute_error(y_real, preds)
            mae_errors.append(mae)
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, top_k + 1), mae_errors, marker='o', label='MAE')
        plt.xlabel('Number of Models in Ensemble (Top-K)')
        plt.ylabel('Mean Absolute Error (MAE)')
        plt.title('Ensemble Performance vs. Number of Models')
        plt.legend()
        plt.grid(True)
        plt.show()

    def evaluate_ensemble_metrics(self, data_loader, top_k):
        mae_errors = []
        mse_errors = []
        r2_scores = []
        y_real = self._get_y_real_labels(data_loader)
        for k in range(1, top_k+1):
            preds = self.ensemble_predict(data_loader, k)
            mae = mean_absolute_error(y_real, preds)
            mse = mean_squared_error(y_real, preds)
            r2 = r2_score(y_real, preds)
            mae_errors.append(mae)
            mse_errors.append(mse)
            r2_scores.append(r2)
        results_df = pd.DataFrame({
        'Top_K': range(1, top_k + 1),
        'MAE': mae_errors,
        'MSE': mse_errors,
        'R2_Score': r2_scores
        })
        return results_df
        
    def save_predictions(self, y_real, y_pred):
        df = pd.DataFrame({'y_real':y_real, 'y_predict':y_pred})
        return df
        
    def _get_y_real_labels(self, data_loader):
        y_reals = []
        for data in data_loader:
            y_reals.append(data.y.cpu())
        y_real = torch.cat(y_reals).squeeze().numpy()
        return y_real
    
    def _get_evaluation_metrics(self, y_real, y_pred):
        mae = mean_absolute_error(y_real, y_pred)
        mse = mean_squared_error(y_real, y_pred)
        r2 = r2_score(y_real, y_pred)
        return {'mae':mae, 'mse':mse, 'r2_score':r2}