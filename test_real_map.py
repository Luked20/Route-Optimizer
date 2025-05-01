import numpy as np
import osmnx as ox
import folium
from environment import RouteOptimizationEnv
from dqn_agent import DQNAgent
from typing import List, Tuple
import random

def get_real_coordinates(city_name: str = "São Paulo, Brasil", num_points: int = 8) -> List[Tuple[float, float]]:
    """Obtém coordenadas reais de pontos na cidade."""
    try:
        # Obtém o grafo da cidade
        G = ox.graph_from_place(city_name, network_type='drive')
        
        # Seleciona pontos aleatórios
        nodes = list(G.nodes())
        points = []
        
        for _ in range(num_points):
            node = random.choice(nodes)
            points.append((G.nodes[node]['y'], G.nodes[node]['x']))  # Note: folium usa (lat, lon)
        
        return points
    except Exception as e:
        print(f"Erro ao obter coordenadas: {e}")
        return [(0, 0) for _ in range(num_points)]

def test_model(num_tests=5, city_name="São Paulo, Brasil"):
    # Obtém coordenadas reais
    real_points = get_real_coordinates(city_name)
    
    # Carrega o ambiente e o modelo
    env = RouteOptimizationEnv(num_delivery_points=8)
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
    
    try:
        agent.load('route_optimizer.pth')
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        return
    
    # Lista para armazenar resultados
    results = []
    
    # Executa os testes
    for test in range(num_tests):
        state = env.reset()
        done = False
        route = [0]  # Começa no depósito
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            route.append(action)
        
        results.append({
            'route': route,
            'total_reward': reward
        })
        
        # Gera mapa para cada teste
        generate_map(real_points, route, f"teste_{test+1}")
    
    # Gera mapa da melhor rota
    best_route = max(results, key=lambda x: x['total_reward'])['route']
    generate_map(real_points, best_route, "melhor_rota")

def generate_map(points: List[Tuple[float, float]], route: List[int], filename: str):
    """Gera um mapa interativo com a rota."""
    try:
        # Cria o mapa
        m = folium.Map(location=points[0], zoom_start=12)
        
        # Adiciona pontos
        for i, point in enumerate(points):
            color = 'red' if i == 0 else 'blue'
            folium.Marker(
                location=point,
                popup=f'Ponto {i}',
                icon=folium.Icon(color=color)
            ).add_to(m)
        
        # Adiciona a rota
        route_points = [points[i] for i in route]
        folium.PolyLine(
            locations=route_points,
            color='green',
            weight=2,
            opacity=0.8
        ).add_to(m)
        
        # Salva o mapa
        m.save(f'{filename}.html')
        print(f"Mapa salvo: {filename}.html")
    except Exception as e:
        print(f"Erro ao gerar mapa {filename}: {e}")

if __name__ == "__main__":
    test_model(city_name="São Paulo, Brasil") 