import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
import gym
import cv2
import math
import os

# ----------------------------
# Model Definitions (As Provided)
# ----------------------------

# Surprise Minimization Layer
class SurpriseMinimizationLayer(nn.Module):
    def __init__(self, input_dim):
        super(SurpriseMinimizationLayer, self).__init__()
        self.fc = nn.Linear(input_dim, input_dim)

    def forward(self, x, target=None):
        if target is not None:
            # Prediction error (surprise)
            error = torch.abs(x - target)
            surprise = error.mean()
            return self.fc(x), surprise
        return self.fc(x), torch.tensor(0.0, device=x.device)

# Liquid Time Constant Network (LTCN) with adaptive neurons
class LiquidTimeConstantNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LiquidTimeConstantNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.fc_in = nn.Linear(input_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.time_constants = nn.Parameter(torch.rand(hidden_size) * 0.1)

    def forward(self, x, prev_state=None):
        if prev_state is None:
            prev_state = torch.zeros(x.size(0), self.hidden_size, device=x.device)

        # Adjust neuron activations over time
        adjusted_state = prev_state + self.time_constants * (self.fc_in(x) - prev_state)
        output = self.fc_out(adjusted_state)
        return output, adjusted_state

# Spiking Neural Network (SNN) Module with diverse STDP learning rules
class SpikingNNModule(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SpikingNNModule, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.spike_threshold = 1.0  # Example threshold

    def forward(self, x):
        membrane_potential = self.fc(x)
        spikes = (membrane_potential >= self.spike_threshold).float()
        return spikes

# Combined model with LTCN and SNN for routine formation
class AdaptiveRoutineModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AdaptiveRoutineModel, self).__init__()
        self.sml = SurpriseMinimizationLayer(input_size)
        self.ltc = LiquidTimeConstantNetwork(input_size, hidden_size, hidden_size)
        self.snn = SpikingNNModule(hidden_size, output_size)

    def forward(self, x, target=None, prev_state=None):
        prediction, surprise = self.sml(x, target)
        ltc_output, ltc_state = self.ltc(prediction, prev_state)
        snn_output = self.snn(ltc_output)
        return snn_output, surprise, ltc_state

# ----------------------------
# Replay Buffer
# ----------------------------

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# ----------------------------
# DQN Agent
# ----------------------------

class DQNAgent:
    def __init__(self, state_dim, action_dim, hidden_size=256, lr=1e-4,
                 gamma=0.99, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=1000000,
                 buffer_capacity=100000, batch_size=32, target_update=1000, device='cuda'):
        self.action_dim = action_dim
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.policy_net = AdaptiveRoutineModel(state_dim, hidden_size, action_dim).to(self.device)
        self.target_net = AdaptiveRoutineModel(state_dim, hidden_size, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criteria = nn.MSELoss()

        self.memory = ReplayBuffer(buffer_capacity)
        self.batch_size = batch_size
        self.gamma = gamma

        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

        self.target_update = target_update

        # For handling recurrent states
        self.prev_state = None

    def select_action(self, state):
        sample = random.random()
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1

        if sample < self.epsilon:
            return random.randrange(self.action_dim)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values, _, _ = self.policy_net(state)
                return q_values.argmax(1).item()

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Convert to tensors
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done_batch = torch.FloatTensor(batch.done).to(self.device)

        # Compute Q(s_t, a)
        q_values, _, _ = self.policy_net(state_batch)
        state_action_values = q_values.gather(1, action_batch).squeeze(1)

        # Compute V(s_{t+1}) for all next states.
        with torch.no_grad():
            next_q_values, _, _ = self.target_net(next_state_batch)
            next_state_values = next_q_values.max(1)[0]
            next_state_values = next_state_values * (1 - done_batch)

        # Compute the expected Q values
        expected_state_action_values = reward_batch + (self.gamma * next_state_values)

        # Compute loss
        loss = self.criteria(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent explosion
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# ----------------------------
# Preprocessing Utilities
# ----------------------------

class AtariPreprocessing:
    def __init__(self, env, resize=84, grayscale=True, frame_stack=4):
        self.env = env
        self.resize = resize
        self.grayscale = grayscale
        self.frame_stack = frame_stack
        self.frames = deque([], maxlen=frame_stack)

    def reset(self):
        state = self.env.reset()
        processed_frame = self.preprocess(state)
        for _ in range(self.frame_stack):
            self.frames.append(processed_frame)
        return self.get_observation()

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        processed_frame = self.preprocess(next_state)
        self.frames.append(processed_frame)
        return self.get_observation(), reward, done, info

    def preprocess(self, frame):
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.resize, self.resize), interpolation=cv2.INTER_AREA)
        frame = frame / 255.0  # Normalize
        return frame

    def get_observation(self):
        return np.stack(self.frames, axis=0)

# ----------------------------
# Training Loop
# ----------------------------

def train_dqn(env_name='PongNoFrameskip-v4', num_episodes=1000, device='cuda'):
    env = gym.make(env_name)
    env = gym.wrappers.NoopResetEnv(env, noop_max=30)
    env = gym.wrappers.MaxAndSkipEnv(env, skip=4)
    env = gym.wrappers.EpisodicLifeEnv(env)
    env = gym.wrappers.FireResetEnv(env)
    env = AtariPreprocessing(env, resize=84, grayscale=True, frame_stack=4)

    action_dim = env.env.action_space.n
    state_dim = env.get_observation().shape  # e.g., (4, 84, 84)

    agent = DQNAgent(state_dim=np.prod(state_dim), action_dim=action_dim,
                     device=device)

    os.makedirs('checkpoints', exist_ok=True)

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        state = state.flatten()  # Flatten the state if necessary
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state_flat = next_state.flatten()
            agent.memory.push(state, action, reward, next_state_flat, done)
            state = next_state_flat
            total_reward += reward

            agent.optimize_model()

            if agent.steps_done % agent.target_update == 0:
                agent.update_target_network()

        print(f"Episode {episode} - Total Reward: {total_reward} - Epsilon: {agent.epsilon:.4f}")

        # Save the model periodically
        if episode % 50 == 0:
            torch.save(agent.policy_net.state_dict(), f'checkpoints/dqn_{env_name}_{episode}.pth')

    env.close()

# ----------------------------
# Main Execution
# ----------------------------

if __name__ == "__main__":
    train_dqn(env_name='PongNoFrameskip-v4', num_episodes=1000, device='cuda')