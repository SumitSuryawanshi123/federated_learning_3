import flwr as fl
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from sklearn.metrics import accuracy_score
import ast
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

def weighted_average(metrics: List[Tuple[int, Dict]]) -> Dict:
    """Aggregation function for (federated) evaluation metrics."""
    # Calculate weighted average for overall accuracy
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    # Calculate weighted average for each class
    result = {"accuracy": sum(accuracies) / sum(examples)}
    
    # Get all class keys from the first metric
    class_keys = [k for k in metrics[0][1].keys() if k.startswith("class_")]
    
    # Calculate weighted average for each class
    for key in class_keys:
        class_accuracies = [num_examples * m[key] for num_examples, m in metrics]
        result[key] = sum(class_accuracies) / sum(examples)
    
    return result

def calculate_metrics(outputs, targets):
    """Calculate accuracy for each class separately"""
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

def get_evaluate_fn(device):
    """Return an evaluation function for server-side evaluation."""
    def evaluate(server_round: int, parameters: List[np.ndarray], config: Dict[str, fl.common.Scalar]):
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 15),
            nn.Sigmoid()  # Added Sigmoid for multi-label classification
        )
        model.to(device)

        # Load test data
        test = pd.read_csv('client-6.csv').head(500)
        
        x_test = np.array([np.array(ast.literal_eval(img)) for img in test['Image']])
        y_test = np.array([np.array(ast.literal_eval(label)) for label in test['Label']])

        # Reshape and preprocess test data
        x_test = x_test.reshape((-1, 1, 128, 128))
        x_test = np.repeat(x_test, 3, axis=1)
        
        x_test = torch.tensor(x_test, dtype=torch.float32) / 255.0
        y_test = torch.tensor(y_test, dtype=torch.float32)  # Changed to float32 for BCELoss

        # Update model parameters
        state_dict = model.state_dict()
        keys = list(state_dict.keys())
        
        for k, v in zip(keys, parameters):
            state_dict[k] = torch.tensor(v)
        
        model.load_state_dict(state_dict)
        
        # Evaluate the model
        model.eval()
        criterion = nn.BCELoss()
        with torch.no_grad():
            x_test_device = x_test.to(device)
            y_test_device = y_test.to(device)
            
            outputs = model(x_test_device)
            loss = criterion(outputs, y_test_device).item()
            mean_accuracy, class_accuracies = calculate_metrics(outputs, y_test_device)

        # Log the accuracies
        with open('densenet-accuracy.txt', 'a') as file:
            file.write(f"Round {server_round}:\n")
            file.write(f"Mean accuracy: {mean_accuracy:.4f}\n")
            for class_name, acc in class_accuracies.items():
                file.write(f"{class_name}: {acc:.4f}\n")
            file.write("\n")

        # Return results including per-class accuracies
        metrics = {"accuracy": mean_accuracy}
        metrics.update(class_accuracies)
        return float(mean_accuracy), metrics

    return evaluate

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        min_available_clients=5,
        min_fit_clients=5,
        evaluate_metrics_aggregation_fn=weighted_average,
        evaluate_fn=get_evaluate_fn(device)
    )

    history = fl.server.start_server(
        server_address="127.0.0.1:8080", 
        config=fl.server.ServerConfig(num_rounds=20), 
        strategy=strategy
    )

    # Plot results
    global_accuracy_centralized = history.metrics_centralized["accuracy"]
    rounds = [data[0] for data in global_accuracy_centralized]
    acc = [100.0 * data[1] for data in global_accuracy_centralized]

    # Create figure with multiple subplots
    plt.figure(figsize=(15, 10))
    
    # Plot overall accuracy
    plt.subplot(2, 1, 1)
    plt.plot(rounds, acc)
    plt.grid(True)
    plt.ylabel("Mean Accuracy (%)")
    plt.xlabel("Round")
    plt.title("Overall Accuracy Across All Classes")
    
    # Plot per-class accuracies
    plt.subplot(2, 1, 2)
    for i in range(15):  # For each class
        class_key = f"class_{i}_acc"
        if class_key in history.metrics_centralized:
            class_acc = history.metrics_centralized[class_key]
            rounds_class = [data[0] for data in class_acc]
            acc_class = [100.0 * data[1] for data in class_acc]
            plt.plot(rounds_class, acc_class, label=f'Class {i}')
    
    plt.grid(True)
    plt.ylabel("Class Accuracy (%)")
    plt.xlabel("Round")
    plt.title("Per-class Accuracies")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('densenet_accuracy_plot.png', bbox_inches='tight')
    plt.show()