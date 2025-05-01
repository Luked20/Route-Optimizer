import numpy as np
import matplotlib.pyplot as plt
from environment import RouteOptimizationEnv
from dqn_agent import DQNAgent
import networkx as nx

def test_model(num_tests=5):
    # Carrega o ambiente
    env = RouteOptimizationEnv(num_delivery_points=8)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Cria o agente
    agent = DQNAgent(state_size, action_size)
    
    # Carrega o modelo treinado
    try:
        agent.load('route_optimizer.pth')
        print("Modelo carregado com sucesso!")
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        return
    
    # Lista para armazenar resultados
    results = []
    
    # Executa os testes
    for test in range(num_tests):
        print(f"\nTeste {test + 1}/{num_tests}")
        
        # Reseta o ambiente
        state = env.reset()
        done = False
        total_reward = 0
        route = [0]  # Começa no depósito
        
        # Executa o episódio
        while not done:
            # Obtém a ação do modelo
            action = agent.act(state)
            
            # Executa a ação
            next_state, reward, done, _ = env.step(action)
            
            # Atualiza o estado e acumula a recompensa
            state = next_state
            total_reward += reward
            route.append(action)
            
            # Mostra informações do passo atual
            print(f"Posição atual: {env.current_position}, "
                  f"Ação: {action}, "
                  f"Recompensa: {reward:.2f}, "
                  f"Distância total: {env.total_distance:.2f}")
        
        # Armazena os resultados
        results.append({
            'route': route,
            'total_reward': total_reward,
            'total_distance': env.total_distance,
            'delivery_times': env.delivery_times
        })
        
        # Mostra resumo do teste
        print(f"\nResumo do Teste {test + 1}:")
        print(f"Rota: {route}")
        print(f"Recompensa Total: {total_reward:.2f}")
        print(f"Distância Total: {env.total_distance:.2f}")
        print(f"Tempos de Entrega: {env.delivery_times}")
    
    # Analisa os resultados
    analyze_results(results)
    
    # Visualiza a melhor rota
    best_test = max(results, key=lambda x: x['total_reward'])
    visualize_route(env.graph, best_test['route'])

def analyze_results(results):
    """Analisa e mostra estatísticas dos resultados."""
    print("\nAnálise dos Resultados:")
    
    # Calcula estatísticas
    rewards = [r['total_reward'] for r in results]
    distances = [r['total_distance'] for r in results]
    
    print(f"Recompensa Média: {np.mean(rewards):.2f}")
    print(f"Recompensa Máxima: {np.max(rewards):.2f}")
    print(f"Recompensa Mínima: {np.min(rewards):.2f}")
    print(f"Distância Média: {np.mean(distances):.2f}")
    print(f"Distância Máxima: {np.max(distances):.2f}")
    print(f"Distância Mínima: {np.min(distances):.2f}")
    
    # Plota os resultados
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(range(len(rewards)), rewards)
    plt.title('Recompensas por Teste')
    plt.xlabel('Teste')
    plt.ylabel('Recompensa Total')
    
    plt.subplot(1, 2, 2)
    plt.bar(range(len(distances)), distances)
    plt.title('Distâncias por Teste')
    plt.xlabel('Teste')
    plt.ylabel('Distância Total')
    
    plt.tight_layout()
    plt.savefig('test_results.png')
    plt.close()

def visualize_route(graph: nx.DiGraph, route: list):
    """Visualiza a rota no grafo."""
    plt.figure(figsize=(10, 8))
    
    # Desenha o grafo
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', 
            node_size=500, font_size=10, font_weight='bold')
    
    # Destaca a rota
    route_edges = [(route[i], route[i+1]) for i in range(len(route)-1)]
    nx.draw_networkx_edges(graph, pos, edgelist=route_edges, 
                          edge_color='red', width=2)
    
    # Adiciona pesos das arestas
    edge_labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
    
    plt.title('Melhor Rota Encontrada')
    plt.savefig('best_route.png')
    plt.close()

if __name__ == "__main__":
    test_model() 