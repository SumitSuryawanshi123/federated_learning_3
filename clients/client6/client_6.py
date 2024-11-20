import flwr as fl
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import ast

# Set client ID
client_id = 6

def log_accuracy(file_path, accuracy):
    with open(file_path, "a") as f:
        f.write(f"{accuracy:.4f}\n")

# # Load and prepare data
# train_data = pd.read_csv('client-5.csv')

# # Prepare features and labels
# x = np.array([np.array(ast.literal_eval(img)) for img in train_data['Image']])
# y = np.array([np.array(ast.literal_eval(label)) for label in train_data['Label']])

# # Split data
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


train_data = pd.read_csv('client-6.csv')

# Prepare training features and labels
x_train = np.array([np.array(ast.literal_eval(img)) for img in train_data['Image']])
y_train = np.array([np.array(ast.literal_eval(label)) for label in train_data['Label']])


# Load test data
test_data = pd.read_csv('client-6.csv').head(500)

# Prepare test features and labels
x_test = np.array([np.array(ast.literal_eval(img)) for img in test_data['Image']])
y_test = np.array([np.array(ast.literal_eval(label)) for label in test_data['Label']])

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if x_train.shape[1] == 128 * 128:
    # Reshape and preprocess data
    x_train = x_train.reshape((-1, 1, 128, 128))
    x_test = x_test.reshape((-1, 1, 128, 128))
    
    # Convert grayscale to RGB by repeating channels
    x_train = np.repeat(x_train, 3, axis=1)
    x_test = np.repeat(x_test, 3, axis=1)
    
    # Convert to tensors and normalize
    x_train = torch.tensor(x_train, dtype=torch.float32) / 255.0
    x_test = torch.tensor(x_test, dtype=torch.float32) / 255.0
    y_train = torch.tensor(y_train, dtype=torch.float32)  # Changed to float32 for BCELoss
    y_test = torch.tensor(y_test, dtype=torch.float32)    # Changed to float32 for BCELoss

    # Create DenseNet model with modified output layer for multi-label
    model = models.densenet121(pretrained=True)
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 15),
        nn.Sigmoid()  # Added Sigmoid for multi-label classification
    )
    model.to(device)
    
    # Loss and optimizer
    criterion = nn.BCELoss()  # Changed to BCELoss
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    

    def calculate_metrics(outputs, targets):
        """Calculate accuracy for each class separately

        Parameters
        ----------
        outputs: torch.Tensor
            model predictions
        targets: torch.Tensor
            ground truth labels

        Returns
        -------
        float
            average accuracy across all classes
        dict
            accuracies for each class
        """
        accuracies = []
        outputs_np = outputs.cpu().numpy()
        targets_np = targets.cpu().numpy()

        # Calculate accuracy for each class
        for i in range(targets_np.shape[1]):
            class_acc = accuracy_score(
                targets_np[:, i], 
                np.where(outputs_np[:, i] >= 0.5, 1, 0)
            )
            accuracies.append(class_acc)

        # Return both mean accuracy and per-class accuracies
        return np.mean(accuracies), {f"class_{i}_acc": acc for i, acc in enumerate(accuracies)}

    # ... (model definition remains the same)

    class FlowerClient(fl.client.NumPyClient):
        def get_parameters(self, config):
            return [val.cpu().numpy() for val in model.state_dict().values()]

        def fit(self, parameters, config):
            # Set the parameters
            params_dict = zip(model.state_dict().keys(), parameters)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            model.load_state_dict(state_dict)

            # Train the model
            model.train()
            for epoch in range(5):
                running_loss = 0.0
                for i in range(0, len(x_train), 32):
                    batch_x = x_train[i:i+32].to(device)
                    batch_y = y_train[i:i+32].to(device)

                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

            # Calculate final training accuracy
            model.eval()
            with torch.no_grad():
                outputs = model(x_train.to(device))
                mean_accuracy, class_accuracies = calculate_metrics(outputs, y_train.to(device))

            # Log accuracies
            log_file_path = f"client_{client_id}_densenet_accuracy.log"
            with open(log_file_path, "a") as f:
                f.write(f"Mean accuracy: {mean_accuracy:.4f}\n")
                for class_name, acc in class_accuracies.items():
                    f.write(f"{class_name}: {acc:.4f}\n")

            return self.get_parameters(config), len(x_train), {}

        def evaluate(self, parameters, config):
            # Set the parameters
            params_dict = zip(model.state_dict().keys(), parameters)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            model.load_state_dict(state_dict)

            # Evaluate the model
            model.eval()
            with torch.no_grad():
                outputs = model(x_test.to(device))
                loss = criterion(outputs, y_test.to(device)).item()
                mean_accuracy, class_accuracies = calculate_metrics(outputs, y_test.to(device))

            # Return results including per-class accuracies
            metrics = {"accuracy": mean_accuracy}
            metrics.update(class_accuracies)

            return float(loss), len(x_test), metrics

    fl.client.start_numpy_client(
            server_address="127.0.0.1:8080",
            client=FlowerClient()
    )

else:
    print("Error: Input data shape mismatch. Cannot reshape.")