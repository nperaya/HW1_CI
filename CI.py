import numpy as np
import matplotlib.pyplot as plt

# DataLoader class for loading and normalizing data
class DataLoader:
    @staticmethod
    def load_and_normalize(file_path, rows, columns, factor=600, skip_header=True):
        """โหลดข้อมูลจากไฟล์และทำการ Normalize"""
        data = np.zeros((rows, columns))
        with open(file_path, 'r') as file:
            if skip_header:
                next(file)
            for l, line in enumerate(file):
                if l >= rows:
                    break
                values = line.strip().split()
                try:
                    data[l] = [float(value) for value in values]
                except ValueError:
                    print(f"Skipping line {l+1} due to conversion error: {values}")
        return data / factor

# Custom neural network class
class NeuralNetworkCustom:
    def __init__(self, layer_config, learning_rate=0.01, momentum=0.9):
        self.layer_config = layer_config
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weights = [np.random.randn(layer_config[i], layer_config[i + 1]) for i in range(len(layer_config) - 1)]
        self.biases = [np.zeros((1, layer_config[i + 1])) for i in range(len(layer_config) - 1)]
        self.weight_momentum = [np.zeros_like(w) for w in self.weights]
        self.bias_momentum = [np.zeros_like(b) for b in self.biases]

    def activation_function(self, x):
        return 1 / (1 + np.exp(-x))
    
    def activation_derivative(self, x):
        return x * (1 - x)
    
    def forward_pass(self, inputs):
        self.activations = [inputs]
        self.pre_activations = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(self.activations[-1], w) + b
            self.pre_activations.append(z)
            self.activations.append(self.activation_function(z))
        return self.activations[-1]
    
    def backward_pass(self, inputs, targets):
        num_samples = inputs.shape[0]
        error = self.activations[-1] - targets
        for i in reversed(range(len(self.weights))):
            grad_w = np.dot(self.activations[i].T, error) / num_samples
            grad_b = np.sum(error, axis=0, keepdims=True) / num_samples
            self.weight_momentum[i] = self.momentum * self.weight_momentum[i] + (1 - self.momentum) * grad_w
            self.bias_momentum[i] = self.momentum * self.bias_momentum[i] + (1 - self.momentum) * grad_b
            self.weights[i] -= self.learning_rate * self.weight_momentum[i]
            self.biases[i] -= self.learning_rate * self.bias_momentum[i]
            if i != 0:
                error = np.dot(error, self.weights[i].T) * self.activation_derivative(self.activations[i])
    
    def train(self, inputs, targets, epochs):
        for _ in range(epochs):
            self.forward_pass(inputs)
            self.backward_pass(inputs, targets)
    
    def predict(self, inputs):
        output = self.forward_pass(inputs)
        return np.argmax(output, axis=1)
    
    def evaluate_accuracy(self, inputs, targets):
        predictions = self.predict(inputs)
        true_labels = np.argmax(targets, axis=1)
        return np.mean(predictions == true_labels)

# Function to generate and display confusion matrix
def generate_confusion_matrix(true_labels, predicted_labels, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(true_labels, predicted_labels):
        matrix[true][pred] += 1
    return matrix

def visualize_confusion_matrix(matrix, fold_number, accuracy):
    fig, ax = plt.subplots()
    ax.set_facecolor('#D1EAF5')  # เปลี่ยนสีพื้นหลังของกราฟ
    cax = ax.matshow(matrix, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, value, ha='center', va='center')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for Fold {fold_number}\nAccuracy: {accuracy * 100:.2f}%')
    plt.show()

# Function to load and preprocess the data
def fetch_data(filepath='c:\\Users\\Pc\\Desktop\\CI\\cross.pat.txt'):
    records = []
    with open(filepath, 'r') as file:
        lines = file.readlines()
        for i in range(1, len(lines), 3):
            features = list(map(float, lines[i].strip().split()))
            labels = list(map(float, lines[i+1].strip().split()))
            records.append(features + labels)
    
    data_array = np.array(records)
    features = data_array[:, :-2]
    targets = data_array[:, -2:]
    return features, targets

def shuffle_dataset(features, targets):
    indices = np.arange(features.shape[0])
    np.random.shuffle(indices)
    return features[indices], targets[indices]

def partition_data(features, targets, train_ratio=0.8):
    split_index = int(len(features) * train_ratio)
    return features[:split_index], features[split_index:], targets[:split_index], targets[split_index:]

# K-Fold Cross-Validation
def k_fold_validation(features, targets, k=10):
    fold_size = len(features) // k
    accuracies = []

    for i in range(k):
        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size
        X_test = features[start_idx:end_idx]
        y_test = targets[start_idx:end_idx]
        X_train = np.concatenate((features[:start_idx], features[end_idx:]), axis=0)
        y_train = np.concatenate((targets[:start_idx], targets[end_idx:]), axis=0)
        
        model = NeuralNetworkCustom(layer_config=[2, 5, 2], learning_rate=0.25, momentum=0.9)
        model.train(X_train, y_train, epochs=15000)
        accuracy = model.evaluate_accuracy(X_test, y_test)
        accuracies.append(accuracy)
        
        y_test_labels = np.argmax(y_test, axis=1)
        y_test_predictions = model.predict(X_test)
        cm = generate_confusion_matrix(y_test_labels, y_test_predictions, 2)
        visualize_confusion_matrix(cm, i + 1, accuracy)
    
    return np.array(accuracies)

# Main workflow for custom neural network
features, targets = fetch_data()
shuffled_features, shuffled_targets = shuffle_dataset(features, targets)
fold_accuracies = k_fold_validation(shuffled_features, shuffled_targets, k=10)
mean_accuracy = np.mean(fold_accuracies)
std_accuracy = np.std(fold_accuracies)

print(f"Mean accuracy: {mean_accuracy * 100:.2f}%")
print(f"Standard deviation of accuracy: {std_accuracy * 100:.2f}%")

final_model = NeuralNetworkCustom(layer_config=[2, 5, 2], learning_rate=0.25, momentum=0.9)
final_model.train(shuffled_features, shuffled_targets, epochs=15000)
final_accuracy = final_model.evaluate_accuracy(shuffled_features, shuffled_targets)
print(f"Final accuracy on the entire dataset: {final_accuracy * 100:.2f}%")

# Plot fold accuracies with mean and final accuracy
plt.figure()
plt.gcf().set_facecolor('#D1EAF5')  # เปลี่ยนสีพื้นหลังของกราฟ
plt.plot(range(1, 11), fold_accuracies * 100, marker='s', linestyle='-', color='#AE7AB4', label='Fold Accuracy')
plt.axhline(y=mean_accuracy * 100, color='#CF7B7E', linestyle='-.', label='Mean Accuracy')
plt.axhline(y=final_accuracy * 100, color='#88B288', linestyle='-.', label='Final Accuracy')
plt.xlabel('Fold')
plt.ylabel('Accuracy (%)')
plt.title('Fold-wise Accuracy with Average and Overall Accuracy')
plt.legend()
plt.show()

# Workflow for MLP with data loader
source_data = 'c:\\Users\\Pc\\Desktop\\CI\\flood_dataset.txt'
data = DataLoader.load_and_normalize(source_data, 252, 9)
input_data = data[:, :8]
output_data = data[:, 8:]

unseen_data = DataLoader.load_and_normalize(source_data, 63, 9)
unseen_input_data = unseen_data[:, :8]
unseen_output_data = unseen_data[:, 8:]

# MLP class for a different neural network
class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.0001, momentum_rate=0.2):
        """สร้างโมเดล MLP"""
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self._initialize_weights(input_size, hidden_size, output_size)

    def _initialize_weights(self, input_size, hidden_size, output_size):
        """กำหนดค่าเริ่มต้นของน้ำหนักและอคติ"""
        np.random.seed(42)
        self.weights_hidden = np.random.rand(input_size, hidden_size)
        self.biases_hidden = np.random.rand(hidden_size)
        self.weights_output = np.random.rand(hidden_size, output_size)
        self.biases_output = np.random.rand(output_size)
        self.prev_weight_changes = {
            'hidden': np.zeros_like(self.weights_hidden),
            'output': np.zeros_like(self.weights_output),
        }
        self.prev_bias_changes = {
            'hidden': np.zeros_like(self.biases_hidden),
            'output': np.zeros_like(self.biases_output),
        }

    def _relu(self, x):
        """ฟังก์ชัน ReLU"""
        return np.maximum(0, x)

    def _relu_derivative(self, x):
        """อนุพันธ์ของฟังก์ชัน ReLU"""
        return np.where(x > 0, 1, 0)

    def _forward(self, inputs):
        """Forward propagation"""
        self.hidden_input = np.dot(inputs, self.weights_hidden) + self.biases_hidden
        self.hidden_output = self._relu(self.hidden_input)
        self.output_input = np.dot(self.hidden_output, self.weights_output) + self.biases_output
        return self.hidden_output, self.output_input

    def _backward(self, inputs, targets):
        """Backpropagation"""
        output_error = self.output_input - targets
        output_gradient = output_error

        weight_output_change = (self.learning_rate * np.dot(self.hidden_output.T, output_gradient)) + (self.momentum_rate * self.prev_weight_changes['output'])
        bias_output_change = (self.learning_rate * np.sum(output_gradient, axis=0)) + (self.momentum_rate * self.prev_bias_changes['output'])

        self.weights_output -= weight_output_change
        self.biases_output -= bias_output_change

        hidden_error = np.dot(output_gradient, self.weights_output.T) * self._relu_derivative(self.hidden_output)
        weight_hidden_change = (self.learning_rate * np.dot(inputs.T, hidden_error)) + (self.momentum_rate * self.prev_weight_changes['hidden'])
        bias_hidden_change = (self.learning_rate * np.sum(hidden_error, axis=0)) + (self.momentum_rate * self.prev_bias_changes['hidden'])

        self.weights_hidden -= weight_hidden_change
        self.biases_hidden -= bias_hidden_change

        self.prev_weight_changes['hidden'] = weight_hidden_change
        self.prev_bias_changes['hidden'] = bias_hidden_change
        self.prev_weight_changes['output'] = weight_output_change
        self.prev_bias_changes['output'] = bias_output_change

    def train(self, inputs, targets, epochs):
        """ฝึกฝนโมเดล"""
        history = {'epoch': [], 'loss': []}
        for epoch in range(epochs):
            self._forward(inputs)
            loss = np.mean((self.output_input - targets) ** 2)
            loss = round(loss, 8)
            self._backward(inputs, targets)

            if epoch % 100 == 0:
                history['epoch'].append(epoch)
                history['loss'].append(loss)
        
        print("Final loss:", loss)
        return history

    def plot_training_results(self, history, targets, predictions):
        """แสดงผลลัพธ์การฝึกฝน"""
        plt.figure(figsize=(10, 8))
        plt.gcf().set_facecolor('#D1EAF5')  # เปลี่ยนสีพื้นหลังของกราฟ

        plt.subplot(2, 1, 1)
        plt.plot(history['epoch'], history['loss'], linestyle='-', color='red')  
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.ylim(0.0001, 0.009)
        plt.title(f"Learning Rate: {self.learning_rate}, Hidden Size: {self.weights_hidden.shape[1]}, Momentum: {self.momentum_rate}")

        plt.subplot(2, 1, 2)
        plt.plot(targets, label="Desired Output", linestyle='-', color='yellow', marker='s')  
        plt.plot(predictions, label="Predicted Output", linestyle='-', color='pink', marker='s')  
        plt.title("Desired Output vs Predicted Output")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

# สร้างและฝึกฝนโมเดล MLP
hidden_size = 3
epochs = 80000
learning_rate = 0.0001
momentum_rate = 0.2

mlp = MLP(input_size=input_data.shape[1], hidden_size=hidden_size, output_size=output_data.shape[1], learning_rate=learning_rate, momentum_rate=momentum_rate)
history = mlp.train(input_data, output_data, epochs)

# การทำนายและแสดงผลลัพธ์
_, predicted_output = mlp._forward(input_data)
predicted_output = predicted_output * 600
print("Predicted output : \n", predicted_output)

mlp.plot_training_results(history, output_data * 600, predicted_output)

# การทดสอบโมเดลกับข้อมูลที่ไม่เคยเห็น
_, predicted_output_unseen = mlp._forward(unseen_input_data)
predicted_output_unseen = predicted_output_unseen * 600
print("Unseen data predicted output: \n", predicted_output_unseen)

# แสดงผลลัพธ์สำหรับข้อมูลที่ไม่เคยเห็น
plt.figure(figsize=(10, 8))
plt.gcf().set_facecolor('#D1EAF5')  # เปลี่ยนสีพื้นหลังของกราฟ
plt.plot(unseen_output_data * 600, label="Desired Output", linestyle='-', color='yellow', marker='o')  
plt.plot(predicted_output_unseen, label="Predicted Output", linestyle='-', color='pink', marker='o')  
plt.title("Desired Output vs Predicted Output for Unseen Data")
plt.legend()
plt.grid(True)
plt.show()
