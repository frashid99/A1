import numpy as np
import random

class A1:
    def __init__(self):
        self.states = []
        self.transition_matrix = None

    def generate_markov_chain(self, states, sequence):
        self.states = states
        state_index = {state: i for i, state in enumerate(states)}
        n = len(states)

        # Initialize transition matrix
        matrix = np.zeros((n, n))

        # Count transitions
        for i in range(len(sequence) - 1):
            current_state = sequence[i]
            next_state = sequence[i + 1]
            matrix[state_index[current_state], state_index[next_state]] += 1

        # Normalize rows to get probabilities
        for i in range(n):
            row_sum = np.sum(matrix[i])
            if row_sum > 0:
                # Normalize to get probabilities
                matrix[i] /= row_sum  

        self.transition_matrix = matrix
        return matrix

    def generate_samples(self, start_state, seed, length):
        
        random.seed(seed)
        state_index = {state: i for i, state in enumerate(self.states)}
        index_state = {i: state for state, i in state_index.items()}

        current_state = start_state
        sequence = [current_state]

        for _ in range(length):
            current_index = state_index[current_state]
            probabilities = self.transition_matrix[current_index]

            # Generate next state based on transition probabilities
            r = random.random()
            cumulative_prob = 0.0
            for i, prob in enumerate(probabilities):
                cumulative_prob += prob
                if r < cumulative_prob:
                    current_state = index_state[i]
                    break

            sequence.append(current_state)

        return sequence

    def stationary_distribution(self):
        
        # Solve P^T π = π
        P_transposed = np.transpose(self.transition_matrix)
        eigenvalues, eigenvectors = np.linalg.eig(P_transposed)

        # Find the eigenvector corresponding to eigenvalue 1
        stationary_vector = eigenvectors[:, np.isclose(eigenvalues, 1)]

        # Normalize to sum to 1
        stationary_vector = stationary_vector[:, 0].real
        stationary_vector /= np.sum(stationary_vector)

        return stationary_vector

