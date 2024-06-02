# Quantum-Influenced Natural Language Processing (NLP) Model

This project demonstrates the implementation of a Quantum-Influenced NLP Model alongside a Standard PyTorch NLP Model for text classification tasks, specifically spam detection. The Quantum-Influenced model utilizes concepts from quantum computing to influence the initialization of its embedding and recurrent neural network (RNN) layers. The project also compares the performance of this quantum-influenced model with a standard NLP model.

## Project Structure

- `spam.csv`: Dataset containing SMS messages labeled as spam or ham (not spam).
- `README.md`: This file.
- `quantum_nn.py`: Python script containing the implementation of the quantum-influenced and standard NLP models.
- `requirements.txt`: File listing the required Python libraries for running the code.

## Prerequisites

- Python 3.x
- PyTorch
- NumPy
- scikit-learn
- pandas
- qiskit
- qiskit-aer

## Installation

Clone the repository and install the dependencies using pip:

```bash
git clone https://github.com/Meshwa428/neural-network-with-qiskit.git
cd neural-network-with-qiskit
pip install -r requirements.txt
python quantum_nn.py
```
