import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import multiprocessing as mp
from dataclasses import dataclass

# ==================== MDP COMPONENTS ====================

@dataclass
class MDPComponents:
    """
    Markov Decision Process Components for AoII Scheduling
    
    State Space S: (Δ, X_l, X_r)
        - Δ: Age of Incorrect Information (unbounded)
        - X_l: Local source state (finite)
        - X_r: Remote estimate (finite)
    
    Action Space A: {0, 1}^N
        - Binary scheduling decision for each sensor
    
    Transition P(s'|s,a): Governed by
        - Markovian source dynamics: P(X_l'|X_l)
        - Transmission success probability: p_s
        - AoII update rule: Δ' = (t - S_c + 1) * I{X_l ≠ X_r}
    
    Cost Function c(s,a):
        - Information cost: ||Δ||_1
        - Transmission cost: w^T a
    """
    
    def __init__(self, transition_matrix: np.ndarray, p_success: float, 
                 transmission_cost: float):
        self.P_transition = transition_matrix  # Source transition matrix
        self.p_s = p_success                   # Transmission success probability
        self.w = transmission_cost             # Transmission cost
        self.num_states = len(transition_matrix)
        
    def state_transition(self, state: Tuple, action: int) -> Tuple:
        """
        Compute next state given current state and action
        
        Args:
            state: (delta, x_l, x_r)
            action: 0 (no transmission) or 1 (transmit)
            
        Returns:
            next_state: (delta', x_l', x_r')
        """
        delta, x_l, x_r = state
        
        # Sample next source state from transition matrix
        x_l_next = np.random.choice(
            self.num_states, 
            p=self.P_transition[x_l]
        )
        
        # Determine remote estimate based on action and transmission success
        if action == 1 and np.random.random() < self.p_s:
            x_r_next = x_l  # Successful update
        else:
            x_r_next = x_r  # No update
        
        # Compute next AoII
        if x_l_next == x_r_next:
            delta_next = 0  # Reset when correct
        else:
            delta_next = delta + 1  # Increment when incorrect
            
        return (delta_next, x_l_next, x_r_next)
    
    def instantaneous_cost(self, delta: int, action: int) -> float:
        """Compute instantaneous cost"""
        return delta + self.w * action


# ==================== OPTIMIZATION FRAMEWORK ====================

class ParametrizedPolicy:
    """
    Parametrized Scheduling Policy
    
    Decision rule: a_t ~ Bernoulli(σ(θ^T φ(s_t)))
    where φ(s) is a feature vector and σ is sigmoid function
    """
    
    def __init__(self, feature_dim: int):
        self.theta = np.zeros(feature_dim)  # Optimization variable
        
    def extract_features(self, state: Tuple) -> np.ndarray:
        """
        Extract features from state for policy parametrization
        
        Features: [1, Δ, I{X_l ≠ X_r}, Δ², log(1+Δ)]
        """
        delta, x_l, x_r = state
        incorrect = float(x_l != x_r)
        
        features = np.array([
            1.0,                    # Bias
            delta,                  # Linear AoII
            incorrect,              # Mismatch indicator
            delta ** 2,             # Quadratic AoII
            np.log(1 + delta)       # Logarithmic AoII
        ])
        return features
    
    def sigmoid(self, x: float) -> float:
        """Sigmoid activation function"""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def get_action_probability(self, state: Tuple) -> float:
        """
        Compute P(a=1|s;θ) = σ(θ^T φ(s))
        """
        features = self.extract_features(state)
        logit = np.dot(self.theta, features)
        return self.sigmoid(logit)
    
    def sample_action(self, state: Tuple) -> int:
        """Sample binary action from Bernoulli distribution"""
        prob = self.get_action_probability(state)
        return int(np.random.random() < prob)


class StochasticGradientOptimizer:
    """
    Asynchronous Stochastic Gradient Descent for Policy Optimization
    
    Update rule: θ_{k+1} = θ_k - α_k ∇̂J(θ_k)
    where ∇̂J is estimated from sampled trajectories
    """
    
    def __init__(self, mdp: MDPComponents, policy: ParametrizedPolicy,
                 learning_rate: float = 0.0005, gamma: float = 0.9):
        self.mdp = mdp
        self.policy = policy
        self.alpha = learning_rate  # Step size
        self.gamma = gamma          # Discount factor
        
    def simulate_trajectory(self, T: int, initial_state: Tuple) -> List[Dict]:
        """
        Generate a single trajectory by simulating the MDP
        
        Returns:
            trajectory: List of {state, action, cost, features, prob}
        """
        trajectory = []
        state = initial_state
        
        for t in range(T):
            # Sample action from current policy
            action = self.policy.sample_action(state)
            action_prob = self.policy.get_action_probability(state)
            
            # Compute cost
            delta = state[0]
            cost = self.mdp.instantaneous_cost(delta, action)
            
            # Record transition
            trajectory.append({
                'state': state,
                'action': action,
                'cost': cost,
                'features': self.policy.extract_features(state),
                'prob': action_prob
            })
            
            # Transition to next state
            state = self.mdp.state_transition(state, action)
            
        return trajectory
    
    def estimate_gradient(self, trajectory: List[Dict]) -> np.ndarray:
        """
        Estimate policy gradient from a single trajectory
        
        Uses policy gradient theorem:
        ∇J(θ) ≈ Σ_t (a_t - π(a_t|s_t)) * R_t * ∇log π(a_t|s_t)
        """
        gradient = np.zeros_like(self.policy.theta)
        T = len(trajectory)
        
        # Compute discounted returns
        returns = np.zeros(T)
        G = 0
        for t in reversed(range(T)):
            G = trajectory[t]['cost'] + self.gamma * G
            returns[t] = G
        
        # Compute gradient estimate
        for t, step in enumerate(trajectory):
            a_t = step['action']
            pi_t = step['prob']
            features = step['features']
            R_t = returns[t]
            
            # Policy gradient: (a_t - π(a_t|s_t)) * R_t * φ(s_t)
            gradient += (a_t - pi_t) * R_t * features
            
        return gradient / T
    
    def parallel_gradient_estimation(self, num_workers: int, T: int, 
                                     initial_state: Tuple) -> np.ndarray:
        """
        Estimate gradient using multiple parallel trajectories (A3C style)
        """
        gradients = []
        
        for _ in range(num_workers):
            trajectory = self.simulate_trajectory(T, initial_state)
            grad = self.estimate_gradient(trajectory)
            gradients.append(grad)
            
        # Average gradients from all workers
        return np.mean(gradients, axis=0)
    
    def update_parameters(self, gradient: np.ndarray):
        """
        Perform gradient descent update: θ ← θ - α∇̂J(θ)
        """
        self.policy.theta -= self.alpha * gradient
        
    def evaluate_policy(self, num_episodes: int, T: int, 
                       initial_state: Tuple) -> float:
        """
        Evaluate current policy by computing average cost
        """
        total_cost = 0
        
        for _ in range(num_episodes):
            trajectory = self.simulate_trajectory(T, initial_state)
            episode_cost = sum(step['cost'] for step in trajectory) / T
            total_cost += episode_cost
            
        return total_cost / num_episodes


# ==================== MAIN OPTIMIZATION PROCEDURE ====================

def run_optimization(transition_matrix: np.ndarray, 
                    p_success: float = 0.8,
                    transmission_cost: float = 0.0,
                    num_episodes: int = 5000,
                    trajectory_length: int = 1000,
                    num_workers: int = 4,
                    learning_rate: float = 0.0005) -> Dict:
    """
    Main optimization loop for A3C-based AoII scheduling
    
    Returns:
        results: Dictionary containing convergence history and final policy
    """
    
    print("=" * 70)
    print("OPTIMIZATION SETUP")
    print("=" * 70)
    print(f"Transition Matrix:\n{transition_matrix}")
    print(f"Success Probability: {p_success}")
    print(f"Transmission Cost: {transmission_cost}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Number of Episodes: {num_episodes}")
    print(f"Trajectory Length: {trajectory_length}")
    print(f"Parallel Workers: {num_workers}")
    print()
    
    # Initialize MDP components
    mdp = MDPComponents(transition_matrix, p_success, transmission_cost)
    
    # Initialize parametrized policy
    feature_dim = 5  # [1, Δ, I{X_l≠X_r}, Δ², log(1+Δ)]
    policy = ParametrizedPolicy(feature_dim)
    
    # Initialize optimizer
    optimizer = StochasticGradientOptimizer(mdp, policy, learning_rate)
    
    # Initial state: synchronized (delta=0, both at state 0)
    initial_state = (0, 0, 0)
    
    # Storage for convergence history
    cost_history = []
    theta_history = []
    
    print("=" * 70)
    print("OPTIMIZATION PROGRESS")
    print("=" * 70)
    
    # Main optimization loop
    for episode in range(num_episodes):
        # Estimate gradient using parallel trajectories
        gradient = optimizer.parallel_gradient_estimation(
            num_workers, trajectory_length, initial_state
        )
        
        # Update policy parameters
        optimizer.update_parameters(gradient)
        
        # Evaluate current policy
        avg_cost = optimizer.evaluate_policy(10, trajectory_length, initial_state)
        
        cost_history.append(avg_cost)
        theta_history.append(policy.theta.copy())
        
        if episode % 20 == 0:
            print(f"Episode {episode:3d} | Avg Cost: {avg_cost:.4f} | "
                  f"θ norm: {np.linalg.norm(policy.theta):.4f}")
    
    print()
    print("=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)
    
    # Final evaluation
    final_cost = optimizer.evaluate_policy(50, trajectory_length, initial_state)
    final_theta = policy.theta
    
    print(f"\nFinal Average Cost: {final_cost:.4f}")
    print(f"Final Parameters θ: {final_theta}")
    
    # Compute scheduling probabilities for different AoII values
    print("\n" + "=" * 70)
    print("FINAL POLICY DECISION RULE")
    print("=" * 70)
    print("\nScheduling Probability P(a=1|Δ) for X_l ≠ X_r:")
    print("-" * 50)
    
    for delta in [0, 1, 2, 3, 5, 10]:
        state = (delta, 0, 1)  # delta, x_l=0, x_r=1 (mismatch)
        prob = policy.get_action_probability(state)
        print(f"  Δ = {delta:2d}  →  P(schedule) = {prob:.4f}")
    
    return {
        'cost_history': cost_history,
        'theta_history': theta_history,
        'final_cost': final_cost,
        'final_theta': final_theta,
        'policy': policy,
        'mdp': mdp
    }


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    
    R1 = np.array([
        [0.7, 0.1, 0.1, 0.1],
        [0.1, 0.7, 0.1, 0.1],
        [0.1, 0.1, 0.7, 0.1],
        [0.1, 0.1, 0.1, 0.7]
    ])
    
    R2 = np.array([
        [0.1, 0.3, 0.3, 0.3],
        [0.3, 0.1, 0.3, 0.3],
        [0.3, 0.3, 0.1, 0.3],
        [0.3, 0.3, 0.3, 0.1]
    ])
    
    theoretical_R1 = 0.58777
    theoretical_R2 = 2.5
    
    print("\n" + "=" * 70)
    print("SCENARIO 1: High-persistence transitions (R1)")
    print("=" * 70)
    
    results_R1 = run_optimization(
        transition_matrix=R1,
        p_success=0.8,
        transmission_cost=0.0,
        num_episodes=5000,
        trajectory_length=1000,
        num_workers=4,
        learning_rate=0.0005
    )
    
    print(f"\nTheoretical Expected Cost: {theoretical_R1:.4f}")
    print(f"Optimized Cost: {results_R1['final_cost']:.4f}")
    print(f"Absolute Error: {abs(results_R1['final_cost'] - theoretical_R1):.4f}")
    
    print("\n" + "=" * 70)
    print("SCENARIO 2: High-variability transitions (R2)")
    print("=" * 70)
    
    results_R2 = run_optimization(
        transition_matrix=R2,
        p_success=0.8,
        transmission_cost=0.0,
        num_episodes=5000,
        trajectory_length=1000,
        num_workers=4,
        learning_rate=0.0005
    )
    
    print(f"\nTheoretical Expected Cost: {theoretical_R2:.4f}")
    print(f"Optimized Cost: {results_R2['final_cost']:.4f}")
    print(f"Absolute Error: {abs(results_R2['final_cost'] - theoretical_R2):.4f}")
    
    # Visualization
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # R1 cost convergence
    axes[0, 0].plot(results_R1['cost_history'], label='Empirical Cost', 
                    linewidth=2, alpha=0.8)
    axes[0, 0].axhline(theoretical_R1, color='r', linestyle='--', 
                       label='Theoretical Expectation', linewidth=2)
    axes[0, 0].set_xlabel('Episode', fontsize=12)
    axes[0, 0].set_ylabel('Average Cost J(θ)', fontsize=12)
    axes[0, 0].set_title('R1: Cost Convergence', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # R2 cost convergence
    axes[0, 1].plot(results_R2['cost_history'], label='Empirical Cost',
                    linewidth=2, alpha=0.8)
    axes[0, 1].axhline(theoretical_R2, color='r', linestyle='--',
                       label='Theoretical Expectation', linewidth=2)
    axes[0, 1].set_xlabel('Episode', fontsize=12)
    axes[0, 1].set_ylabel('Average Cost J(θ)', fontsize=12)
    axes[0, 1].set_title('R2: Cost Convergence', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # R1 parameter evolution
    theta_array_R1 = np.array(results_R1['theta_history'])
    for i in range(theta_array_R1.shape[1]):
        axes[1, 0].plot(theta_array_R1[:, i], label=f'θ[{i}]', linewidth=1.5)
    axes[1, 0].set_xlabel('Episode', fontsize=12)
    axes[1, 0].set_ylabel('Parameter Value', fontsize=12)
    axes[1, 0].set_title('R1: Parameter Evolution', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # R2 parameter evolution
    theta_array_R2 = np.array(results_R2['theta_history'])
    for i in range(theta_array_R2.shape[1]):
        axes[1, 1].plot(theta_array_R2[:, i], label=f'θ[{i}]', linewidth=1.5)
    axes[1, 1].set_xlabel('Episode', fontsize=12)
    axes[1, 1].set_ylabel('Parameter Value', fontsize=12)
    axes[1, 1].set_title('R2: Parameter Evolution', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('a3c_optimization_results.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'a3c_optimization_results.png'")
    plt.show()
    
    print("\n" + "=" * 70)
    print("ALL OPTIMIZATIONS COMPLETE")
    print("=" * 70)
