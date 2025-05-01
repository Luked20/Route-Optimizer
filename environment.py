import gym
import numpy as np
import networkx as nx
from gym import spaces
from typing import List, Tuple, Dict

class RouteOptimizationEnv(gym.Env):
    def __init__(self, num_delivery_points: int = 8):
        super(RouteOptimizationEnv, self).__init__()
        
        self.num_delivery_points = num_delivery_points
        self.graph = self._create_city_graph()
        
        # Espaço de observação: [posição_atual, entregas_pendentes, tempo_atual]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0]),
            high=np.array([self.num_delivery_points, 1, 24]),
            dtype=np.float32
        )
        
        # Espaço de ação: escolher próximo ponto de entrega
        self.action_space = spaces.Discrete(self.num_delivery_points)
        
        self.reset()
    
    def _create_city_graph(self) -> nx.DiGraph:
        """Cria um grafo representando a cidade com pontos de entrega."""
        G = nx.DiGraph()
        
        # Adiciona nós (0 é o depósito, 1-n são pontos de entrega)
        for i in range(self.num_delivery_points + 1):
            G.add_node(i)
        
        # Adiciona arestas com pesos aleatórios
        for i in range(self.num_delivery_points + 1):
            for j in range(self.num_delivery_points + 1):
                if i != j:
                    # Peso baseado em distância e tráfego
                    distance = np.random.uniform(1, 10)
                    traffic = np.random.uniform(0.8, 1.2)
                    G.add_edge(i, j, weight=distance * traffic)
        
        # Garante que o grafo seja fortemente conectado
        if not nx.is_strongly_connected(G):
            # Adiciona arestas adicionais para garantir conectividade
            for i in range(self.num_delivery_points + 1):
                for j in range(self.num_delivery_points + 1):
                    if i != j and not G.has_edge(i, j):
                        distance = np.random.uniform(1, 10)
                        traffic = np.random.uniform(0.8, 1.2)
                        G.add_edge(i, j, weight=distance * traffic)
        
        return G
    
    def reset(self):
        """Reseta o ambiente para o estado inicial."""
        self.current_position = 0  # Depósito
        self.deliveries_pending = np.ones(self.num_delivery_points)
        self.current_time = 0
        self.total_distance = 0
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Retorna o estado atual do ambiente."""
        return np.array([
            self.current_position,
            np.sum(self.deliveries_pending),
            self.current_time
        ], dtype=np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Executa uma ação no ambiente."""
        if action < 0 or action >= self.num_delivery_points:
            raise ValueError(f"Ação inválida: {action}")
        
        # Verifica se a aresta existe
        if not self.graph.has_edge(self.current_position, action):
            # Se não existir, retorna uma penalidade alta
            return self._get_state(), -100.0, False, {}
        
        # Calcula distância até o próximo ponto
        distance = self.graph[self.current_position][action]['weight']
        self.total_distance += distance
        
        # Atualiza tempo (assumindo velocidade constante)
        time_spent = distance
        self.current_time += time_spent
        
        # Marca entrega como concluída
        self.deliveries_pending[action] = 0
        
        # Calcula recompensa
        reward = self._calculate_reward(distance, time_spent)
        
        # Verifica se o episódio terminou
        done = np.sum(self.deliveries_pending) == 0 and self.current_position == 0
        
        # Atualiza posição atual
        self.current_position = action
        
        return self._get_state(), reward, done, {}
    
    def _calculate_reward(self, distance: float, time_spent: float) -> float:
        """Calcula a recompensa baseada na distância e tempo gasto."""
        # Penalidade por distância
        distance_penalty = -distance * 0.1
        
        # Penalidade por tempo
        time_penalty = -time_spent * 0.05
        
        # Bônus por completar entregas
        delivery_bonus = 10.0 if np.sum(self.deliveries_pending) == 0 else 0.0
        
        return distance_penalty + time_penalty + delivery_bonus 