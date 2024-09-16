import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from src.model import MLP, EarlyStopping

def train_models(
                X_train, y_train, X_test, y_test, 
                 architecture: list =[6, 20, 20, 15, 3], 
                 n_models: int =2, steps: int =10000, lr: float =1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = []

    for i in range(n_models):
        tqdm.write(f'Training model {i}')
        model = MLP(np.array(architecture)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        es = EarlyStopping()

        train_losses, test_losses = [], []

        for epoch in range(steps):
            # Train step
            model.train()
            optimizer.zero_grad()
            loss = loss_fn(model(X_train), y_train)
            loss.backward()
            optimizer.step()

            # Evaluation step
            if epoch % 100 == 0:
                model.eval()
                with torch.no_grad():
                    test_loss = loss_fn(model(X_test), y_test)
                    train_losses.append(loss.item())
                    test_losses.append(test_loss.item())
                    tqdm.write(f'Epoch: {epoch}, Train Loss: {loss.item():.3f}, Test Loss: {test_loss.item():.3f}')

                if es(loss.item(), test_loss.item()): 
                    tqdm.write(f'Early stopping at epoch {epoch}')
                    break

        results.append((model, epoch, train_losses, test_losses))

    return results

# Usage
# X_train, y_train = X_train.to(device), y_train.to(device)
# X_test, y_test = X_test.to(device), y_test.to(device)

# results = train_models(X_train, y_train, X_test, y_test, n_models=4, steps=5000, lr=1e-2)

# # Unpack results
# model_list, epoch, train_loss_list, test_loss_list = zip(*results)