import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict

class HiddenMarkovModel:
    def __init__(self, n_states: int = 5):
        self.n_states = n_states
        self.transition_probs = np.ones((n_states, n_states)) / n_states
        self.emission_probs = np.ones((n_states, 5)) / 5
        self.initial_probs = np.ones(n_states) / n_states
        
    def _forward(self, observations: List[int]) -> Tuple[np.ndarray, float]:
        T = len(observations)
        alpha = np.zeros((T, self.n_states))
        
        for i in range(self.n_states):
            alpha[0, i] = self.initial_probs[i] * self.emission_probs[i, observations[0]]
        
        for t in range(1, T):
            for j in range(self.n_states):
                alpha[t, j] = np.sum(alpha[t-1] * self.transition_probs[:, j]) * self.emission_probs[j, observations[t]]
        
        likelihood = np.sum(alpha[-1])
        return alpha, likelihood
    
    def _backward(self, observations: List[int]) -> np.ndarray:
        T = len(observations)
        beta = np.zeros((T, self.n_states))
        beta[-1] = 1.0
        
        for t in range(T-2, -1, -1):
            for i in range(self.n_states):
                beta[t, i] = np.sum(self.transition_probs[i] * self.emission_probs[:, observations[t+1]] * beta[t+1])
        
        return beta
    
    def _viterbi(self, observations: List[int]) -> List[int]:
        T = len(observations)
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)
        
        delta[0] = self.initial_probs * self.emission_probs[:, observations[0]]
        
        for t in range(1, T):
            for j in range(self.n_states):
                delta[t, j] = np.max(delta[t-1] * self.transition_probs[:, j]) * self.emission_probs[j, observations[t]]
                psi[t, j] = np.argmax(delta[t-1] * self.transition_probs[:, j])
        
        states = np.zeros(T, dtype=int)
        states[-1] = np.argmax(delta[-1])
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
        
        return states.tolist()
    
    def fit(self, observations: List[int], max_iter: int = 100, tol: float = 1e-4):
        T = len(observations)
        prev_log_likelihood = float('-inf')
        
        for iteration in range(max_iter):
            alpha, likelihood = self._forward(observations)
            beta = self._backward(observations)
            
            gamma = alpha * beta / likelihood
            xi = np.zeros((T-1, self.n_states, self.n_states))
            
            for t in range(T-1):
                for i in range(self.n_states):
                    for j in range(self.n_states):
                        xi[t, i, j] = (alpha[t, i] * self.transition_probs[i, j] * 
                                     self.emission_probs[j, observations[t+1]] * beta[t+1, j]) / likelihood
            
            self.initial_probs = gamma[0]
            self.transition_probs = np.sum(xi, axis=0) / np.sum(gamma[:-1], axis=0)[:, np.newaxis]
            
            for j in range(self.n_states):
                for k in range(5):
                    mask = np.array(observations) == k
                    self.emission_probs[j, k] = np.sum(gamma[mask, j]) / np.sum(gamma[:, j])
            
            log_likelihood = np.log(likelihood)
            if abs(log_likelihood - prev_log_likelihood) < tol:
                break
            prev_log_likelihood = log_likelihood
    
    def predict(self, observations: List[int]) -> List[int]:
        return self._viterbi(observations)

def train_hmm(entries: List[Dict], n_states: int = 5) -> Tuple[HiddenMarkovModel, List[int]]:
    observations = []
    for entry in entries:
        text = entry["entry"].lower()
        features = {
            'exclamation_ratio': text.count('!') / max(1, len(text.split('.'))),
            'self_ref_ratio': text.count(' i ') / max(1, len(text.split())),
            'group_ref_ratio': (text.count(' we ') + text.count(' us ')) / max(1, len(text.split())),
            'contrast_ratio': (text.count(' but ') + text.count(' however ')) / max(1, len(text.split())),
            'question_ratio': text.count('?') / max(1, len(text.split('.'))),
            'emotion_words': sum(text.count(word) for word in ['happy', 'sad', 'excited', 'tired', 'amazing', 'fun']) / max(1, len(text.split()))
        }
        
        if features['exclamation_ratio'] > 0.05 or features['emotion_words'] > 0.01:
            obs = 0
        elif features['self_ref_ratio'] > 0.03:
            obs = 1
        elif features['group_ref_ratio'] > 0.01:
            obs = 2
        elif features['contrast_ratio'] > 0.01 or features['question_ratio'] > 0.05:
            obs = 3
        else:
            obs = 4
        observations.append(obs)
    
    model = HiddenMarkovModel(n_states=n_states)
    
    model.transition_probs = np.ones((n_states, n_states)) * 0.1
    np.fill_diagonal(model.transition_probs, 0.3)
    for i in range(n_states):
        if i > 0:
            model.transition_probs[i, i-1] = 0.2
        if i < n_states-1:
            model.transition_probs[i, i+1] = 0.2
    model.transition_probs = model.transition_probs / model.transition_probs.sum(axis=1, keepdims=True)
    
    model.emission_probs = np.ones((n_states, 5)) * 0.1
    np.fill_diagonal(model.emission_probs, 0.6)
    model.emission_probs = model.emission_probs / model.emission_probs.sum(axis=1, keepdims=True)
    
    model.fit(observations, max_iter=100)
    
    hidden_states = model.predict(observations)
    
    print("\nFeature Analysis:")
    for i, entry in enumerate(entries):
        print(f"\nEntry {i+1} ({entry['date']}):")
        text = entry['entry'].lower()
        curr_features = {
            'exclamation_ratio': text.count('!') / max(1, len(text.split('.'))),
            'self_ref_ratio': text.count(' i ') / max(1, len(text.split())),
            'group_ref_ratio': (text.count(' we ') + text.count(' us ')) / max(1, len(text.split())),
            'contrast_ratio': (text.count(' but ') + text.count(' however ')) / max(1, len(text.split())),
            'question_ratio': text.count('?') / max(1, len(text.split('.'))),
            'emotion_words': sum(text.count(word) for word in ['happy', 'sad', 'excited', 'tired', 'amazing', 'fun']) / max(1, len(text.split()))
        }
        for feature, value in curr_features.items():
            print(f"{feature}: {value:.4f}")
        print(f"Assigned State: {hidden_states[i]}")
    
    print("\nTransition Matrix:")
    print(model.transition_probs)
    print("\nEmission Matrix:")
    print(model.emission_probs)
    
    return model, hidden_states 