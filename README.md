# Otimização de Rotas com Reinforcement Learning

Este projeto implementa um sistema de otimização de rotas de entrega usando Deep Q-Learning (DQN). O sistema é projetado para otimizar a sequência de visitas a pontos de entrega em um ambiente urbano simplificado, considerando fatores como distância, tempo e restrições.

## Requisitos

- Python 3.8+
- PyTorch
- OpenAI Gym
- NetworkX
- Matplotlib
- NumPy

Instale as dependências usando:
```bash
pip install -r requirements.txt
```

## Estrutura do Projeto

- `environment.py`: Implementa o ambiente de simulação usando OpenAI Gym
- `dqn_agent.py`: Implementa o agente DQN para otimização de rotas
- `train.py`: Script principal para treinar o agente
- `requirements.txt`: Lista de dependências do projeto

## Funcionalidades

- Otimização de rotas para 6-8 pontos de entrega
- Consideração de diferentes pesos de rota (simulando tráfego)
- Respeito a janelas de tempo para entregas
- Minimização de distância total e tempo de viagem
- Retorno ao depósito após completar todas as entregas

## Como Usar

1. Instale as dependências:
```bash
pip install -r requirements.txt
```

2. Treine o agente:
```bash
python train.py
```

3. O treinamento irá gerar:
   - `route_optimizer.pth`: Modelo treinado
   - `training_results.png`: Gráficos de desempenho

## Parâmetros do Modelo

- Taxa de aprendizado: 0.001
- Fator de desconto (gamma): 0.95
- Tamanho do batch: 32
- Taxa de exploração inicial (epsilon): 1.0
- Decaimento do epsilon: 0.995
- Epsilon mínimo: 0.01

## Contribuição

Sinta-se à vontade para contribuir com melhorias ou reportar problemas através de pull requests e issues. 