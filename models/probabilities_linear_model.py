from datetime import datetime
import hashlib
from io import BytesIO
import os
import sys
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

base_directory = os.getcwd()
sys.path.insert(0, base_directory)
from utility.minio import cmd


class ProbabilitiesModel:
    def __init__(self, input_size):
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        self._device = torch.device(device)

        self.model_type = 'probabilities-linear'
        self.loss_func_name = ''
        self.file_path = ''
        self.model_hash = ''
        self.date = datetime.now().strftime("%Y-%m-%d")

        self.training_loss = 0.0
        self.validation_loss = 0.0
        
        self.input_size = input_size
        self.model = nn.Linear(input_size, 1)
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.tanh = nn.Tanh()

    def _hash_model(self):
        """
        Hashes the current state of the model, and stores the hash in the
        instance of the classifier.
        """
        model_str = str(self.model.state_dict())
        self.model_hash = hashlib.sha256(model_str.encode()).hexdigest()

    def save(self, minio_client, datasets_bucket, model_output_path):
        # Hashing the model with its current configuration
        self._hash_model()
        self.file_path = model_output_path
        # Preparing the model to be saved
        model = {}
        model['model_dict'] = self.model.state_dict()
        # Adding metadata
        model['model-type'] = self.model_type
        model['file-path'] = self.file_path
        model['model-hash'] = self.model_hash
        model['date'] = self.date

        # Saving the model to minio
        buffer = BytesIO()
        torch.save(model, buffer)
        buffer.seek(0)
        
        # upload the model
        cmd.upload_data(minio_client, datasets_bucket, model_output_path, buffer)
    
    def load(self, model_buffer):
        # Loading state dictionary
        model = torch.load(model_buffer)
        # Restoring model metadata
        self.model_type = model['model-type']
        self.file_path = model['file-path']
        self.model_hash = model['model-hash']
        self.date = model['date']
        self.model.load_state_dict(model['model_dict'])

    def train(self,
              X, y,  # Pass your data and labels as arguments
              training_batch_size=10,
              epochs=100,
              learning_rate=0.05,
              weight_decay=0.01,
              normalize=False):
        
        training_loss_per_epoch = []
        validation_loss_per_epoch = []

        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.model_type = 'probabilities-model'
        self.loss_func_name = "L1"

        if normalize:
            # If normalize is set to True, normalize the features.
            X = (X - X.mean()) / X.std()

        # Split the data into training and testing sets.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Convert data to PyTorch tensors.
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train).view(-1, 1)
        X_test = torch.FloatTensor(X_test)
        y_test = torch.FloatTensor(y_test).view(-1, 1)

        # Create DataLoader for batch training
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=training_batch_size, shuffle=True)

        # Training loop
        for epoch in range(epochs):
            self.model.train()  # Set the model to training mode
            epoch_training_loss = 0.0

            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.mse_loss(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_training_loss += loss.item()

            # Calculate validation loss
            self.model.eval()  # Set the model to evaluation mode
            with torch.no_grad():
                validation_outputs = self.model(X_test)
                validation_loss = self.mse_loss(validation_outputs, y_test).item()

            epoch_training_loss /= len(train_loader)
            training_loss_per_epoch.append(epoch_training_loss)
            validation_loss_per_epoch.append(validation_loss)

            print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {epoch_training_loss:.4f}, Validation Loss: {validation_loss:.4f}')

        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        # Plot Validation Loss vs. Epochs
        axs[0, 0].plot(range(epoch + 1), validation_loss_per_epoch, label='Validation Loss')
        axs[0, 0].set_xlabel('Epoch')
        axs[0, 0].set_ylabel('Loss')
        axs[0, 0].legend()

        # Plot Actual vs. Predicted Values
        with torch.no_grad():
            test_predictions = self.model(X_test).numpy()
        axs[0, 1].scatter(y_test.numpy(), test_predictions, alpha=0.5)
        axs[0, 1].set_xlabel('Actual Values')
        axs[0, 1].set_ylabel('Predicted Values')

        # Plot Residuals
        residuals = (y_test - validation_outputs.view(-1, 1)).numpy()
        axs[1, 0].scatter(test_predictions, residuals, alpha=0.5)
        axs[1, 0].axhline(0, color='red', linestyle='--')
        axs[1, 0].set_xlabel('Predicted Values')
        axs[1, 0].set_ylabel('Residuals')

        # Plot Distribution of Residuals
        axs[1, 1].hist(residuals, bins=30, color='blue', alpha=0.7)
        axs[1, 1].set_xlabel('Residuals')
        axs[1, 1].set_ylabel('Frequency')

        plt.tight_layout()

        plt.show()
        return self.model, training_loss_per_epoch, validation_loss_per_epoch

    def predict(self, X):
        X = torch.FloatTensor(X)
        with torch.no_grad():
            return self.model(X).detach().numpy()
