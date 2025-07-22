import numpy as np

class HopfieldNetwork:
    def __init__(self, size):
        """
        Initialize Hopfield Network
        size: number of neurons (64 for 8x8 grid)
        """
        self.size = size
        self.weights = np.zeros((size, size))
        self.patterns = []
        
    def train(self, patterns):
        """
        Train the network using Hebbian learning rule
        patterns: list of binary patterns (each pattern is a flattened array)
        """
        self.patterns = patterns
        self.weights = np.zeros((self.size, self.size))
        
        for pattern in patterns:
            # Convert 0s to -1s for bipolar representation
            bipolar_pattern = 2 * pattern - 1
            # Hebbian learning: W = sum(xi * xj) for all patterns
            self.weights += np.outer(bipolar_pattern, bipolar_pattern)
        
        # Set diagonal to zero (no self-connections)
        np.fill_diagonal(self.weights, 0)
        
        # Normalize by number of patterns
        if len(patterns) > 0:
            self.weights /= len(patterns)
    
    def recall(self, pattern, max_iterations=100):
        """
        Recall a pattern using asynchronous updates
        Returns the recalled pattern and convergence history
        """
        # Convert to bipolar
        state = 2 * pattern - 1
        history = [pattern.copy()]
        
        for iteration in range(max_iterations):
            prev_state = state.copy()
            
            # Asynchronous update (update one neuron at a time)
            for i in range(self.size):
                net_input = np.dot(self.weights[i], state)
                state[i] = 1 if net_input >= 0 else -1
            
            # Convert back to binary for history
            binary_state = (state + 1) // 2
            history.append(binary_state.copy())
            
            # Check for convergence
            if np.array_equal(state, prev_state):
                break
        
        # Return final binary state and history
        return (state + 1) // 2, history
    
    def energy(self, pattern):
        """Calculate the energy of a given pattern"""
        bipolar = 2 * pattern - 1
        return -0.5 * np.dot(bipolar, np.dot(self.weights, bipolar))
