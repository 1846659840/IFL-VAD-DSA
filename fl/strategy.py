import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg

class IFLVADDSAStrategy(Strategy):
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        alpha: float = 0.7,
        beta: float = 0.3,
        global_lr: float = 0.1
    ) -> None:
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.alpha = alpha  # Shapley weight parameter alpha
        self.beta = beta    # Shapley weight parameter beta
        self.global_lr = global_lr
        self.global_statistical_model = None
        self.pca_matrix = None
        self.anomaly_threshold = 3.0  # Initial anomaly threshold
        self.reduced_dim = 64  # Initial reduced dimension
        self.clients_per_round = 0  # Number of clients per round
        self.force_label_update = False  # Whether to force clients to update labels
        self.parameters = None  # Store current global model parameters

    def __repr__(self) -> str:
        return "IFLVADDSAStrategy"

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters"""
        # In actual implementation, should create and initialize model
        # Here, assume parameters are provided externally
        return self.parameters

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure next training round"""
        # Store current parameters
        self.parameters = parameters
        
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Create configurations
        fit_configurations = []
        for client in clients:
            config = {
                "lr": 0.001,
                "epochs": 1,
                "server_round": server_round,
                "force_label_update": getattr(self, "force_label_update", False),
                "anomaly_threshold": self.anomaly_threshold
            }

            # If available, send PCA matrix
            if self.pca_matrix is not None:
                config["pca_matrix"] = self.pca_matrix.tolist()

            fit_configurations.append((client, FitIns(parameters, config)))

        return fit_configurations

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Implement Phase III: Global Aggregation with Decoupled Shapley Contributions"""
        if not results:
            return None, {}

        # Record number of clients in this round, for pseudo-label rule updates
        self.clients_per_round = len(results)

        # Build global statistical model
        self._update_global_statistical_model(results)

        # Calculate Shapley-based weights
        client_weights = self._compute_shapley_weights(results)

        # Get current global model parameters
        current_global_parameters = parameters_to_ndarrays(self.parameters)

        # Calculate updates (delta) for each client and weight them
        deltas = []
        for (_, fit_res), weight in zip(results, client_weights):
            client_parameters = parameters_to_ndarrays(fit_res.parameters)
            # Calculate update: client parameters - global parameters
            delta = [(client_param - global_param) for client_param, global_param
                    in zip(client_parameters, current_global_parameters)]
            deltas.append((delta, weight))

        # Apply Shapley-weighted aggregation and global learning rate
        updated_parameters = self._aggregate_with_lr(deltas, current_global_parameters)
        parameters_aggregated = ndarrays_to_parameters(updated_parameters)

        # Update PCA matrix and pseudo-label generation rules
        updated_pca = False
        updated_rules = False

        if server_round % 5 == 0:  # Update PCA matrix every 5 rounds
            self._update_pca_matrix()
            updated_pca = True

        # Update pseudo-label generation rules
        updated_rules = self.update_pseudolabel_rules(results)

        # If rules were updated, force clients to update labels in next round
        self.force_label_update = updated_rules or server_round % 10 == 0

        # Calculate metrics
        metrics = {
            "round": server_round,
            "updated_pca": updated_pca,
            "updated_rules": updated_rules,
            "force_label_update": self.force_label_update,
            "anomaly_threshold": self.anomaly_threshold
        }

        return parameters_aggregated, metrics

    def update_pseudolabel_rules(self, results):
        """Update pseudo-label generation rules based on clients' label update statistics"""
        total_updates = {"videos": 0, "segments": 0}
        clients_with_updates = 0

        for _, fit_res in results:
            metrics = fit_res.metrics
            if "label_updates" in metrics and metrics["label_updates"]:
                updates = metrics["label_updates"]
                total_updates["videos"] += updates.get("changed_video_labels", 0)
                total_updates["segments"] += updates.get("changed_segment_labels", 0)
                clients_with_updates += 1

        if clients_with_updates > 0:
            # Adjust thresholds based on label update statistics
            # If many labels were updated, may need to adjust pseudo-label generation thresholds
            avg_video_updates = total_updates["videos"] / clients_with_updates
            avg_segment_updates = total_updates["segments"] / clients_with_updates

            print(f"Average updates per client: {avg_video_updates:.2f} video labels and {avg_segment_updates:.2f} segment labels")

            # Dynamically adjust PCA matrix dimensions - e.g., if large label changes, increase dimensions to capture more variance
            if avg_segment_updates > 10:  # Assumed threshold
                self.reduced_dim = min(128, self.reduced_dim + 4)  # Increase dimensions
                print(f"Increased PCA reduced dimensions to {self.reduced_dim}")
            elif avg_segment_updates < 2:  # Few updates
                self.reduced_dim = max(32, self.reduced_dim - 2)  # Decrease dimensions
                print(f"Decreased PCA reduced dimensions to {self.reduced_dim}")

            # Adjust pseudo-label generation thresholds
            if avg_video_updates > 0.3 * self.clients_per_round:  # If most clients updated labels
                self.anomaly_threshold *= 0.9  # Lower anomaly threshold
                print(f"Lowered anomaly threshold to {self.anomaly_threshold}")
            elif avg_video_updates < 0.1 * self.clients_per_round:
                self.anomaly_threshold *= 1.1  # Increase anomaly threshold
                print(f"Increased anomaly threshold to {self.anomaly_threshold}")

            return True  # Indicate rules were updated

        return False  # No rules were updated

    def _update_global_statistical_model(self, results):
        """Update global statistical model using client information"""
        client_stats = []
        for _, fit_res in results:
            metrics = fit_res.metrics
            if "avg_sigma" in metrics and "avg_g" in metrics:
                client_stats.append([metrics["avg_sigma"], metrics["avg_g"]])

        if client_stats:
            self.global_statistical_model = np.mean(client_stats, axis=0)

    def _compute_shapley_weights(self, results):
        """Calculate Shapley weights for clients"""
        if self.global_statistical_model is None:
            # If no global model yet, use equal weights
            return [1.0 for _ in results]

        n_clients = len(results)
        weights = []

        # For each client, calculate individual importance
        for i, (_, fit_res) in enumerate(results):
            # Use AUC instead of accuracy as quality metric
            auc = fit_res.metrics.get("auc", 0.5)
            individual_importance = auc - 0.5  # Baseline is random guessing

            # Calculate interaction importance with other clients
            interaction_importance = 0
            for j, (_, other_res) in enumerate(results):
                if i != j:
                    other_auc = other_res.metrics.get("auc", 0.5)
                    # Simplified interaction calculation
                    interaction = ((auc + other_auc) / 2) - 0.5
                    interaction_importance += interaction

            # Combine alpha and beta weights
            weight = self.alpha * individual_importance + self.beta * interaction_importance

            # Ensure weight is positive
            weights.append(max(0.1, weight))

        return weights

    def _aggregate_with_lr(self, deltas, current_global_parameters):
        """Aggregate client updates according to Shapley values and apply to global model"""
        # Aggregate client updates (weighted by Shapley values)
        aggregated_delta = []
        for i in range(len(current_global_parameters)):
            weighted_sum_delta = np.zeros_like(current_global_parameters[i])
            weight_sum = 0.0

            for delta_params, weight in deltas:
                weighted_sum_delta += delta_params[i] * weight
                weight_sum += weight

            # Apply global learning rate and add to current global parameters
            if weight_sum > 0:
                aggregated_delta.append(
                    current_global_parameters[i] + (weighted_sum_delta / weight_sum) * self.global_lr
                )
            else:
                aggregated_delta.append(current_global_parameters[i])

        return aggregated_delta

    def _update_pca_matrix(self):
        """Update PCA matrix"""
        feature_dim = 128  # Feature dimension
        reduced_dim = self.reduced_dim  # Use dynamically adjusted reduced dimension

        # Create random matrix and orthogonalize
        matrix = np.random.randn(feature_dim, reduced_dim)
        q, r = np.linalg.qr(matrix)
        self.pca_matrix = q
        print(f"Updated PCA matrix, reduced dimension: {reduced_dim}")

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure next evaluation round"""
        if self.fraction_evaluate == 0.0:
            return []

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Create evaluation instructions
        config = {
            "round": server_round,
            "anomaly_threshold": self.anomaly_threshold
        }
        if self.pca_matrix is not None:
            config["pca_matrix"] = self.pca_matrix.tolist()

        return [(client, EvaluateIns(parameters, config)) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results"""
        if not results:
            return None, {}

        loss_aggregated = weighted_loss_avg(
            [(evaluate_res.num_examples, evaluate_res.loss) for _, evaluate_res in results]
        )

        # Calculate average AUC
        aucs = [res.metrics["auc"] for _, res in results if "auc" in res.metrics]
        avg_auc = sum(aucs) / len(aucs) if aucs else 0.5

        return loss_aggregated, {"auc": avg_auc}

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate global model parameters"""
        # No central evaluation in this strategy
        return None

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients"""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation"""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients 