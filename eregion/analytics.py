import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class EregionAnalytics:
    def __init__(self):
        pass

    def compute_metrics(self, y_true, y_pred, multi_class=False):
        if multi_class:
            return {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='macro'),
                'recall': recall_score(y_true, y_pred, average='macro'),
                'f1_score': f1_score(y_true, y_pred, average='macro')
            }
        else:
            return {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred),
                'recall': recall_score(y_true, y_pred),
                'f1_score': f1_score(y_true, y_pred)
            }

    def entropy_of_predictions(self, probabilities):
        """
        Calculates the entropy of the predictions to measure uncertainty.
        """
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-9), axis=1)  # 1e-9 to avoid log(0)
        return np.mean(entropy)

    def dead_neurons_detection(self, activations):
        """
        Detects dead neurons by analyzing the activations of neurons in a layer.
        A neuron is considered 'dead' if its activation is 0 for the entire batch.
        """
        dead_neurons = np.mean(np.all(activations == 0, axis=0))
        return dead_neurons

    def gradient_norm(self, gradients):
        """
        Computes the norm of gradients (useful for monitoring optimization progress).
        """
        grad_norms = [np.linalg.norm(grad) for grad in gradients]
        return np.mean(grad_norms)

    def weight_sparsity(self, weights):
        """
        Calculates the sparsity of model weights (percentage of zero weights).
        """
        total_weights = np.prod(weights.shape)
        zero_weights = np.sum(weights == 0)
        return zero_weights / total_weights

    def confidence_score(self, probabilities):
        """
        Measures the confidence of the model by calculating the mean max probability.
        """
        return np.mean(np.max(probabilities, axis=1))

    def overfitting_detection(self, train_loss, val_loss):
        """
        Detects overfitting by comparing train and validation loss.
        """
        if val_loss > train_loss:
            overfit_percentage = (val_loss - train_loss) / train_loss * 100
            return {'overfit_detected': True, 'overfit_percentage': overfit_percentage}
        return {'overfit_detected': False, 'overfit_percentage': 0}

    def layer_activation_distribution(self, activations):
        """
        Analyzes the activation distribution for each layer.
        """
        distributions = {}
        for layer_name, activation in activations.items():
            activation_mean = np.mean(activation)
            activation_std = np.std(activation)
            non_zero_ratio = np.mean(activation != 0)
            distributions[layer_name] = {
                'mean': activation_mean,
                'std': activation_std,
                'non_zero_ratio': non_zero_ratio  # Used to detect dead neurons
            }
        return distributions

    def compute_efficiency(self, training_time, memory_usage):
        """
        Compute efficiency metrics like time and memory.
        """
        return {
            'training_time': training_time,
            'memory_usage': memory_usage
        }

    def compute_bias(self, y_true, y_pred, sensitive_attribute):
        """
        Computes fairness and bias metrics.
        """
        groups = np.unique(sensitive_attribute)
        parity_ratio = {}
        for group in groups:
            group_mask = (sensitive_attribute == group)
            group_accuracy = accuracy_score(y_true[group_mask], y_pred[group_mask])
            parity_ratio[group] = group_accuracy
        return parity_ratio

    def compute_robustness(self, perturbed_input, original_output, perturbed_output):
        """
        Computes model robustness by comparing outputs under perturbations.
        """
        return {
            'output_shift': np.mean(np.abs(original_output - perturbed_output))
        }