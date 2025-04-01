import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pdb
from tqdm import tqdm

class UpProjection(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[500, 1000, 10000]):
        super(UpProjection, self).__init__()
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:  # No activation on the last layer
                layers.append(nn.GELU())

        self.model = nn.Sequential(*layers)
        self.model = self.model.to('cuda')

    def forward(self, x):
        return self.model(x)

# Training function
def train_model(model, train_dataloader, val_dataloader, epochs=100, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in tqdm(range(epochs)):
        for x_batch, y_batch in train_dataloader:
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

        train_bs = x_batch.shape[0]

        if epoch % 10 == 0:
            # Evaluate
            model.eval()
            val_loss = []
            with torch.no_grad():
                for x_batch, y_batch in val_dataloader:
                    y_pred = model(x_batch)
                    loss = criterion(y_pred, y_batch)
                    val_loss.append(loss.item())
            test_bs = x_batch.shape[0]
            val_loss = sum(val_loss) / (test_bs * len(val_loss))
            model.train()

            print(f"Epoch {epoch}, Train Loss: {loss.item() / train_bs:.6f}")
            print(f"Epoch {epoch}, Test Loss: {val_loss:.6f}")

