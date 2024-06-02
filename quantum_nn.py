import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import qiskit
from qiskit_aer import Aer
from qiskit import QuantumCircuit, transpile
import pandas as pd

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

df = pd.read_csv(r"spam.csv", encoding="ISO-8859-1")
df.rename(columns={"v1": "class_label", "v2": "message"}, inplace=True)
df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1, inplace=True)

df_spam = df[df.class_label == "spam"]
df_ham = df[df.class_label == "ham"]

spam_msg_list = df_spam["message"].tolist()
ham_msg_list = df_ham["message"].tolist()[: len(spam_msg_list)]

spam_list = df_spam["class_label"].tolist()
ham_list = df_ham["class_label"].tolist()[: len(spam_list)]

sentences = spam_msg_list + ham_msg_list
base_labels = spam_list + ham_list

labels = []
for i in base_labels:
    if i == "spam":
        labels.append(1)
    else:
        labels.append(0)


vocab = set(" ".join(sentences).split())
word_to_index = {word: i + 1 for i, word in enumerate(vocab)}  # Reserve 0 for padding

max_len = max(len(sentence.split()) for sentence in sentences)


def sentence_to_indices(sentence):
    indices = [word_to_index.get(word, 0) for word in sentence.split()]
    padding = [0] * (max_len - len(indices))
    return padding + indices


X = [sentence_to_indices(sentence) for sentence in sentences]
X = torch.tensor(X, dtype=torch.long)
y = torch.tensor(LabelEncoder().fit_transform(labels), dtype=torch.long)

print(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


def advanced_quantum_circuit(num_qubits=10, num_neurons=None):
    if num_neurons is None:
        num_neurons = num_qubits // 2

    qc = QuantumCircuit(num_qubits, num_neurons)

    # Layers: Initialization, Rotation, Entanglement, Multi-Level, Non-Linear, Measurement-Based
    qc.h(range(num_qubits))

    angles = [
        np.pi / 2,
        np.pi / 3,
        np.pi / 5,
        np.pi / 7,
        np.pi / 11,
        np.pi / 13,
        np.pi / 17,
        np.pi / 19,
        np.pi / 23,
        np.pi / 29,
    ]
    for i, angle in zip(range(num_qubits), angles):
        qc.rx(angle, i)
        qc.ry(angle * 1.1, i)
        qc.rz(angle * 0.9, i)

    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)
        qc.cp(np.pi / 4, i, i + 1)
    for i in range(0, num_qubits - 2, 2):
        qc.ccx(i, i + 1, i + 2)

    for i in range(num_qubits):
        qc.rx(np.pi / 3, i)
        qc.cx(i, (i + 1) % num_qubits)
        qc.ry(np.pi / 4, i)
        qc.cx(i, (i + 2) % num_qubits)
        qc.rz(np.pi / 5, i)

    return qc


# Function to generate a quantum state
def generate_quantum_state(num_qubits, num_neurons):
    qc = advanced_quantum_circuit(num_qubits, num_neurons)
    simulator = Aer.get_backend("statevector_simulator")
    t_qc = transpile(qc, simulator)
    result = simulator.run(t_qc).result()
    statevector = result.get_statevector()
    return np.asarray(statevector)


# Function to get quantum tensor of desired size
def get_quantum_tensor(desired_size, num_neurons=31):
    num_qubits = int(np.ceil(np.log2(desired_size)))
    actual_size = 2**num_qubits
    quantum_state = generate_quantum_state(num_qubits, num_neurons)
    quantum_tensor = torch.tensor(
        np.concatenate((quantum_state.real, quantum_state.imag)),
        dtype=torch.float32,
    )[
        :actual_size
    ]  # Only take the first 2^n elements
    return quantum_tensor, actual_size


# Function to initialize matrix with quantum tensor
def initialize_matrix_with_quantum(matrix, quantum_tensor, actual_size):
    if matrix.numel() <= actual_size:
        # If the matrix is smaller or equal, just copy the first elements
        matrix.data = quantum_tensor[: matrix.numel()].view(matrix.shape)
    else:
        # If the matrix is larger, tile the quantum tensor
        repeats = [max(s // actual_size, 1) for s in matrix.shape]
        expanded_tensor = quantum_tensor.repeat(*repeats)
        matrix.data[
            : expanded_tensor.shape[0], : expanded_tensor.shape[1]
        ] = expanded_tensor


# Quantum influenced model
class QuantumNLPModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(QuantumNLPModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embedding_dim, padding_idx=0)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.initialize_weights(embedding_dim, hidden_dim)

    def initialize_weights(self, embedding_dim, hidden_dim):
        # Initialize embedding weights
        quantum_tensor, _ = get_quantum_tensor(embedding_dim)
        initialize_matrix_with_quantum(self.embedding.weight, quantum_tensor, _)

        # Initialize RNN weights
        ih_tensor, ih_size = get_quantum_tensor(self.rnn.weight_ih_l0.numel())
        hh_tensor, hh_size = get_quantum_tensor(self.rnn.weight_hh_l0.numel())

        initialize_matrix_with_quantum(self.rnn.weight_ih_l0, ih_tensor, ih_size)
        initialize_matrix_with_quantum(self.rnn.weight_hh_l0, hh_tensor, hh_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        last_output = output[:, -1, :]  # Use the last output for classification
        return torch.sigmoid(self.fc(last_output))


# Standard PyTorch model (unchanged)
class StandardNLPModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(StandardNLPModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embedding_dim, padding_idx=0)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        last_output = output[:, -1, :]  # Use the last output for classification
        return torch.sigmoid(self.fc(last_output))


# Training and evaluation functions (unchanged)
def train_model(model, X_train, y_train, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(epochs):
        model.train()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def evaluate_model(model, X_test, y_test):
    model.eval()
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test).sum().item() / len(y_test)
    return accuracy


# Model parameters
vocab_size = len(word_to_index)
embedding_dim = 16
hidden_dim = 32
output_dim = 2

# Initialize models
quantum_model = QuantumNLPModel(vocab_size, embedding_dim, hidden_dim, output_dim)
standard_model = StandardNLPModel(vocab_size, embedding_dim, hidden_dim, output_dim)

# Train models
train_model(quantum_model, X_train, y_train)
train_model(standard_model, X_train, y_train)

# Evaluate models
quantum_accuracy = evaluate_model(quantum_model, X_test, y_test)
standard_accuracy = evaluate_model(standard_model, X_test, y_test)

# Print results
print(f"Quantum-influenced model accuracy: {quantum_accuracy}")
print(f"Standard model accuracy: {standard_accuracy}")

indices = sentence_to_indices("hey bro, wana hangout tonight?")
inputs = torch.tensor([indices], dtype=torch.long)

outputs = quantum_model(inputs)
outputs_2 = standard_model(inputs)

_, predicted = torch.max(outputs, 1)

_2, predicted_2 = torch.max(outputs_2, 1)

print(outputs)
print(outputs_2)
