import torch
import flwr
from typing import Dict, List, Optional, Tuple
import numpy as np
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Context, Parameters, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.simulation import run_simulation

from models.feature_extractor import ImageBindTransformerExtractor
from models.detector import TransformerViolenceDetector
from data.video_loader import load_video_data, load_test_data
from fl.client import IFLVADDSAClient
from fl.strategy import IFLVADDSAStrategy
from utils.helpers import get_parameters, set_parameters, evaluate_on_test_set

# Detect device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training device: {DEVICE}")
print(f"Flower {flwr.__version__} / PyTorch {torch.__version__}")

# Flower client function
def client_fn(context: Context) -> Client:
    """Create Flower client"""
    # Extract client information
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    # Create models
    feature_extractor = ImageBindTransformerExtractor().to(DEVICE)
    model = TransformerViolenceDetector(input_dim=128, hidden_dim=256).to(DEVICE)

    # Load video data
    try:
        videos = load_video_data(partition_id, num_partitions)
    except Exception as e:
        print(f"Client {partition_id} failed to load data: {str(e)}")
        # Use random data for testing
        videos = [torch.rand(16, 3, 224, 224) for _ in range(3)]

    # Initial PCA matrix (will be updated by the server)
    feature_dim = 128
    reduced_dim = 64
    pca_matrix = np.eye(feature_dim, reduced_dim)

    # Create client
    client = IFLVADDSAClient(
        partition_id=partition_id,
        videos=videos,
        feature_extractor=feature_extractor,
        model=model,
        pca_matrix=pca_matrix,
        window_length=5,
        anomaly_threshold=3.0,
        percentile=80,
        device=DEVICE
    )

    return client.to_client()

def main():
    # Number of clients
    NUM_PARTITIONS = 5

    # Create client application
    client_app = ClientApp(client_fn=client_fn)

    # Create server with custom strategy
    def server_fn(context: Context) -> ServerAppComponents:
        # Initialize model parameters
        model = TransformerViolenceDetector(input_dim=128, hidden_dim=256).to(DEVICE)
        model_parameters = get_parameters(model)
        parameters = ndarrays_to_parameters(model_parameters)

        # Create strategy
        strategy = IFLVADDSAStrategy(
            fraction_fit=0.8,
            fraction_evaluate=0.5,
            min_fit_clients=2,
            min_evaluate_clients=2,
            min_available_clients=2,
            alpha=0.7,
            beta=0.3,
            global_lr=0.1
        )
        
        # Set initial parameters
        strategy.parameters = parameters

        # Create server configuration
        config = ServerConfig(num_rounds=10)
        
        return ServerAppComponents(config=config, strategy=strategy)

    server_app = ServerApp(server_fn=server_fn)

    # Configure resources
    backend_config = {"client_resources": None}
    if torch.cuda.is_available():
        backend_config = {"client_resources": {"num_gpus": 1}}

    # Run simulation
    print("Starting federated learning simulation...")
    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=NUM_PARTITIONS,
        backend_config=backend_config,
    )
    
    # Evaluate final model on test set
    print("\n--- Final Evaluation on Test Set ---")
    
    # Load test data
    try:
        test_videos, test_labels = load_test_data()
        
        # Create and initialize models for evaluation
        feature_extractor = ImageBindTransformerExtractor().to(DEVICE)
        global_model = TransformerViolenceDetector(input_dim=128, hidden_dim=256).to(DEVICE)
        
        # Get final parameters from the server
        final_parameters = server_app._server.strategy.parameters
        numpy_params = parameters_to_ndarrays(final_parameters)
        set_parameters(global_model, numpy_params)
        
        # Evaluate on test set using AUC metric
        test_auc, predictions, _ = evaluate_on_test_set(
            model=global_model,
            feature_extractor=feature_extractor,
            test_videos=test_videos,
            test_labels=test_labels,
            device=DEVICE
        )
        
        print(f"Final model evaluation on test set - AUC: {test_auc:.4f}")
        
    except Exception as e:
        print(f"Error during test evaluation: {str(e)}")
        print("Skipping test evaluation.")

if __name__ == "__main__":
    main() 