import numpy as np
from typing import List


class FeedbackNeuralNetwork:
    """Simple feedforward neural network for generating feedback"""

    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int):
        self.layers = []
        self.input_size = input_size
        self.output_size = output_size

        # Build network architecture with smaller, more stable initialization
        layer_sizes = [input_size] + hidden_sizes + [output_size]

        for i in range(len(layer_sizes) - 1):
            # Much smaller initialization to prevent explosion
            fan_in = layer_sizes[i]
            std = 0.1 / np.sqrt(fan_in)  # Very conservative initialization

            layer = {
                'weights': np.random.normal(0, std, (layer_sizes[i], layer_sizes[i + 1])),
                'biases': np.zeros(layer_sizes[i + 1])
            }
            self.layers.append(layer)

    def _validate_and_clean_input(self, x: np.ndarray) -> np.ndarray:
        """Validate and clean input data"""
        # Replace NaN and inf values
        x = np.where(np.isnan(x), 0.0, x)
        x = np.where(np.isinf(x), 0.0, x)

        # Clip extreme values
        x = np.clip(x, -100, 100)

        return x

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the network"""
        if x.ndim != 1:
            raise ValueError(f"Expected 1D input, got shape {x.shape}")
        if x.shape[0] != self.input_size:
            raise ValueError(f"Expected input size {self.input_size}, got {x.shape[0]}")

        # Clean input
        activation = self._validate_and_clean_input(x)

        for i, layer in enumerate(self.layers):
            # Compute linear transformation
            z = np.dot(activation, layer['weights']) + layer['biases']

            # Clean intermediate values
            z = self._validate_and_clean_input(z)

            # Apply activation function
            if i < len(self.layers) - 1:
                activation = np.maximum(0, z)  # ReLU
            else:
                activation = self._safe_softmax(z)  # Safe softmax

        return activation

    def _safe_softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable and safe softmax"""
        if x.ndim != 1:
            raise ValueError(f"Expected 1D input to softmax, got shape {x.shape}")

        # Clean input
        x = self._validate_and_clean_input(x)

        # Clip to prevent overflow
        x = np.clip(x, -500, 500)

        # Subtract max for stability
        x_max = np.max(x)
        if np.isnan(x_max) or np.isinf(x_max):
            x_max = 0.0

        x_shifted = x - x_max

        # Compute exponentials
        exp_x = np.exp(x_shifted)

        # Handle edge cases
        sum_exp = np.sum(exp_x)
        if sum_exp <= 1e-15 or np.isnan(sum_exp) or np.isinf(sum_exp):
            # Return uniform distribution if unstable
            return np.ones(len(x)) / len(x)

        return exp_x / sum_exp

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000,
              learning_rate: float = 0.01):
        """Train the network using backpropagation"""

        if X.ndim != 2:
            raise ValueError(f"Expected X to be 2D, got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"Expected y to be 1D, got shape {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Mismatched samples: X has {X.shape[0]}, y has {y.shape[0]}")
        if X.shape[1] != self.input_size:
            raise ValueError(f"Expected input features {self.input_size}, got {X.shape[1]}")

        # Validate input data
        print("Validating input data...")
        X_clean = np.zeros_like(X)
        for i in range(X.shape[0]):
            X_clean[i] = self._validate_and_clean_input(X[i])

        # Check for completely invalid data
        if np.all(X_clean == 0):
            print("Warning: All input data was invalid and set to zero")
            return

        # Simple normalization to prevent gradient explosion
        X_std = np.std(X_clean, axis=0)
        X_std = np.where(X_std < 1e-8, 1.0, X_std)  # Prevent division by zero
        X_mean = np.mean(X_clean, axis=0)
        X_normalized = (X_clean - X_mean) / X_std

        # Store normalization parameters
        self.X_mean = X_mean
        self.X_std = X_std

        n_samples = X.shape[0]

        # Validate labels
        unique_labels = np.unique(y)
        if np.min(unique_labels) < 0 or np.max(unique_labels) >= self.output_size:
            raise ValueError(
                f"Labels must be in range [0, {self.output_size - 1}], got range [{np.min(unique_labels)}, {np.max(unique_labels)}]")

        # Very conservative learning rate
        max_lr = min(learning_rate, 0.001)

        print(f"Starting training with {n_samples} samples, max LR: {max_lr}")

        for epoch in range(epochs):
            total_loss = 0
            valid_samples = 0

            # Adaptive learning rate with more aggressive decay
            current_lr = max_lr * (0.9 ** (epoch // 5))

            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_normalized[indices]
            y_shuffled = y[indices]

            for i in range(n_samples):
                x_i = X_shuffled[i]
                y_i = int(y_shuffled[i])

                try:
                    # Forward pass
                    activations = [x_i]

                    for j, layer in enumerate(self.layers):
                        z = np.dot(activations[-1], layer['weights']) + layer['biases']

                        # Check for numerical issues early
                        if np.any(np.isnan(z)) or np.any(np.isinf(z)):
                            raise ValueError(f"Numerical instability in layer {j}")

                        if j < len(self.layers) - 1:
                            activation = np.maximum(0, z)  # ReLU
                        else:
                            activation = self._safe_softmax(z)  # Safe softmax

                        activations.append(activation)

                    # Calculate loss with safety checks
                    y_pred = activations[-1]
                    pred_value = y_pred[y_i]

                    # Ensure prediction is valid
                    if np.isnan(pred_value) or np.isinf(pred_value) or pred_value <= 0:
                        continue  # Skip this sample

                    # Safe log calculation
                    pred_value = max(pred_value, 1e-15)  # Prevent log(0)
                    loss = -np.log(pred_value)

                    if np.isnan(loss) or np.isinf(loss):
                        continue  # Skip this sample

                    total_loss += loss
                    valid_samples += 1

                    # Backward pass with extensive safety checks
                    success = self._safe_backward_pass(activations, y_i, current_lr)
                    if not success:
                        continue  # Skip if backward pass failed

                except Exception as e:
                    # Skip problematic samples
                    continue

            # Calculate average loss only from valid samples
            if valid_samples > 0:
                avg_loss = total_loss / valid_samples
                accuracy = self._calculate_safe_accuracy(X_normalized, y)

                if epoch % 20 == 0 or epoch == epochs - 1:
                    print(f"Epoch {epoch}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.3f}, "
                          f"Valid samples: {valid_samples}/{n_samples}, LR: {current_lr:.6f}")
            else:
                print(f"Epoch {epoch}: No valid samples processed")
                break

    def _safe_backward_pass(self, activations: List[np.ndarray], y_true: int,
                            learning_rate: float) -> bool:
        """Safe backward pass with extensive error checking"""
        try:
            # Output layer error
            y_pred = activations[-1]
            y_one_hot = np.zeros(self.output_size)
            y_one_hot[y_true] = 1

            # Calculate delta with safety checks
            delta = y_pred - y_one_hot

            if np.any(np.isnan(delta)) or np.any(np.isinf(delta)):
                return False

            # Aggressive gradient clipping
            delta = np.clip(delta, -1.0, 1.0)

            # Backpropagate through layers
            for i in range(len(self.layers) - 1, -1, -1):
                layer = self.layers[i]
                a_prev = activations[i]

                # Check activations
                if np.any(np.isnan(a_prev)) or np.any(np.isinf(a_prev)):
                    return False

                # Compute gradients
                grad_weights = np.outer(a_prev, delta)
                grad_biases = delta.copy()

                # Check gradients
                if (np.any(np.isnan(grad_weights)) or np.any(np.isinf(grad_weights)) or
                        np.any(np.isnan(grad_biases)) or np.any(np.isinf(grad_biases))):
                    return False

                # Very aggressive gradient clipping
                grad_weights = np.clip(grad_weights, -0.5, 0.5)
                grad_biases = np.clip(grad_biases, -0.5, 0.5)

                # Update parameters
                layer['weights'] -= learning_rate * grad_weights
                layer['biases'] -= learning_rate * grad_biases

                # Clip parameters to prevent explosion
                layer['weights'] = np.clip(layer['weights'], -5.0, 5.0)
                layer['biases'] = np.clip(layer['biases'], -5.0, 5.0)

                # Compute delta for next layer
                if i > 0:
                    delta = np.dot(layer['weights'], delta)

                    if np.any(np.isnan(delta)) or np.any(np.isinf(delta)):
                        return False

                    # Apply ReLU derivative
                    delta = delta * (activations[i] > 0)

                    # Clip delta
                    delta = np.clip(delta, -1.0, 1.0)

            return True

        except Exception:
            return False

    def _calculate_safe_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy with error handling"""
        correct = 0
        total = 0

        for i in range(X.shape[0]):
            try:
                pred = self._safe_predict(X[i])
                if pred == y[i]:
                    correct += 1
                total += 1
            except:
                continue  # Skip failed predictions

        return correct / total if total > 0 else 0.0

    def _safe_predict(self, x: np.ndarray) -> int:
        """Safe prediction with error handling"""
        try:
            output = self.forward(x)
            if np.any(np.isnan(output)) or np.any(np.isinf(output)):
                return 0  # Default prediction
            return int(np.argmax(output))
        except:
            return 0  # Default prediction

    def predict(self, x: np.ndarray) -> int:
        """Make prediction for a single sample or batch"""
        if x.ndim == 1:
            # Normalize if we have training statistics
            if hasattr(self, 'X_mean') and hasattr(self, 'X_std'):
                x_normalized = (x - self.X_mean) / self.X_std
                return self._safe_predict(x_normalized)
            else:
                return self._safe_predict(x)
        elif x.ndim == 2:
            # Batch prediction
            predictions = []
            for sample in x:
                predictions.append(self.predict(sample))
            return np.array(predictions)
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")