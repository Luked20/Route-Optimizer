import numpy as np
import matplotlib.pyplot as plt
from environment import RouteOptimizationEnv
from dqn_agent import DQNAgent

def train_agent(episodes=1000, target_update=10):
    env = RouteOptimizationEnv(num_delivery_points=8)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQNAgent(state_size, action_size)
    
    rewards_history = []
    distances_history = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            
            state = next_state
            total_reward += reward
        
        if episode % target_update == 0:
            agent.update_target_network()
        
        rewards_history.append(total_reward)
        distances_history.append(env.total_distance)
        
        if episode % 10 == 0:
            print(f"Episódio: {episode}, Recompensa: {total_reward:.2f}, "
                  f"Distância: {env.total_distance:.2f}, Epsilon: {agent.epsilon:.2f}")
    
    # Salva o modelo treinado
    agent.save('route_optimizer.pth')
    
    # Plota os resultados
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(rewards_history)
    plt.title('Histórico de Recompensas')
    plt.xlabel('Episódio')
    plt.ylabel('Recompensa Total')
    
    plt.subplot(1, 2, 2)
    plt.plot(distances_history)
    plt.title('Histórico de Distâncias')
    plt.xlabel('Episódio')
    plt.ylabel('Distância Total')
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.close()

if __name__ == "__main__":
    train_agent() 