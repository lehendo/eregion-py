import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

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

    def entropy_of_predictions(self, outputs):
        entropy_values = []
        for output in outputs:
            if isinstance(output, (float, int)):
                # For single float/int values, entropy is 0
                entropy_values.append(0.0)
            elif isinstance(output, np.ndarray) and output.ndim == 1:
                # For 1D numpy arrays, calculate entropy
                entropy = -np.sum(output * np.log(output + 1e-10))  # Avoid log(0)
                entropy_values.append(entropy)
            else:
                raise ValueError('Unsupported output type: must be float, int, or 1D numpy array.')
        return np.array(entropy_values)

    def dead_neurons_detection(self, outputs):
        """
        Detects dead neurons by analyzing the activations of neurons in a layer.
        A neuron is considered 'dead' if its activation is 0 for the entire batch.
        """
        if not outputs:
            return None

        activations = np.array(outputs)
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

    def layer_activation_distribution(self, outputs):
        """
        Analyzes the activation distribution for each layer.
        """
        if not outputs:
            return None

        activations = np.array(outputs)
        activation_mean = np.mean(activations)
        activation_std = np.std(activations)
        non_zero_ratio = np.mean(activations != 0)

        return {
            'mean': activation_mean,
            'std': activation_std,
            'non_zero_ratio': non_zero_ratio  # Used to detect dead neurons
        }

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
    def entropy_distribution(self, probabilities, n_bins=10):
        """
        Generates data for an entropy distribution histogram.
        """
        entropy_values = -np.sum(probabilities * np.log(probabilities + 1e-9), axis=1)
        hist, bin_edges = np.histogram(entropy_values, bins=n_bins, range=(0, 1))
        return hist, bin_edges

    def calibration_curve_entropy(self, probabilities, labels, n_bins=10):
        """
        Prepares data for a calibration curve.
        """
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        accuracies = []
        confidences = []

        for i in range(n_bins):
            bin_mask = (probabilities >= bin_edges[i]) & (probabilities < bin_edges[i + 1])
            bin_confidence = np.mean(probabilities[bin_mask]) if bin_mask.any() else 0
            bin_accuracy = np.mean(labels[bin_mask] == (probabilities[bin_mask] > 0.5)) if bin_mask.any() else 0
            accuracies.append(bin_accuracy)
            confidences.append(bin_confidence)

        return bin_centers, accuracies, confidences

    def uncertainty_trend_over_time(self, predictions, timestamps):
        """
        Tracks entropy trend over time.
        """
        entropy_values = -np.sum(predictions * np.log(predictions + 1e-9), axis=1)
        trend_data = pd.DataFrame({'timestamp': timestamps, 'entropy': entropy_values})
        trend_df = trend_data.groupby('timestamp')['entropy'].mean().reset_index()
        return trend_df

    def confidence_vs_entropy_data(self, probabilities):
        """
        Prepares data for a confidence vs. entropy scatter plot.
        """
        confidences = np.max(probabilities, axis=1)
        entropies = -np.sum(probabilities * np.log(probabilities + 1e-9), axis=1)
        return confidences, entropies

    def entropy_distribution_across_classes(self, probabilities, labels, n_classes):
        """
        Gathers entropy distribution data across each class.
        """
        entropies = -np.sum(probabilities * np.log(probabilities + 1e-9), axis=1)
        class_entropy_data = {class_id: [] for class_id in range(n_classes)}

        for label, entropy in zip(labels, entropies):
            class_entropy_data[label].append(entropy)

        return class_entropy_data

    def entropy_progress_over_epochs(self, epoch_entropies):
        """
        Prepares data for the average entropy of model predictions over training epochs.
        """
        return {
            'epochs': list(range(len(epoch_entropies))),
            'average_entropy': epoch_entropies
        }

    def loss_curve_data(self, labeled_losses, unlabeled_losses):
        """
        Prepares data for the loss curve comparison of labeled and unlabeled data.
        """
        return {
            'epochs': list(range(len(labeled_losses))),
            'labeled_loss': labeled_losses,
            'unlabeled_loss': unlabeled_losses
        }

    def confidence_distribution(self, probabilities, n_bins=10):
        """
        Prepares data for the confidence distribution histogram.
        """
        confidences = np.max(probabilities, axis=1)
        hist, bin_edges = np.histogram(confidences, bins=n_bins, range=(0, 1))
        return hist, bin_edges

    def tsne_or_pca_data(self, data, n_components=2):
        """
        Prepares data for a t-SNE or PCA scatter plot for labeled and unlabeled data representations.
        """
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE

        if n_components == 2:
            tsne = TSNE(n_components=2)
            reduced_data = tsne.fit_transform(data)
        else:
            pca = PCA(n_components=n_components)
            reduced_data = pca.fit_transform(data)

        return reduced_data