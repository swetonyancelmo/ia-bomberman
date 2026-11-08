"""IA treinada por PPO, no formato esperado pelo main.py (decidir_acao).

Traduz o estado do jogo (objetos do pygame) para o BombermanEnv, so lendo,
nunca alterando, para nao disparar a deteccao de trapaca do main.py.
"""
import os
from types import SimpleNamespace

import numpy as np
import torch

from bomberman_env import ACOES, DT, PASSOS_PARTIDA, PASSOS_EXPLOSAO
from bomberman_obs import observar, mascara_acoes, acao_real

ARQUIVO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelo_bomberman.pt")
_rede = None


def _carregar():
    global _rede
    if _rede is None:
        from treino_ppo import Rede
        _rede = Rede()
        _rede.load_state_dict(torch.load(ARQUIVO, weights_only=True))
        _rede.eval()
    return _rede


def _snapshot(players, mapa, bombas, tempo_restante):
    """Copia do estado do main.py com a mesma interface que observar() usa."""
    env = SimpleNamespace()
    env.mapa = [linha[:] for linha in mapa]
    env.passo = int(round((PASSOS_PARTIDA * DT - tempo_restante) / DT))
    env.jogadores = [
        SimpleNamespace(x=p.grid_x, y=p.grid_y, time=p.time, vivo=p.ativo,
                        max_bombas=p.max_bombas, nivel=p.bomba_nivel, bombas=len(p.bombas))
        for p in players
    ]
    env.bombas = [
        SimpleNamespace(x=b.x, y=b.y, explodida=b.explodida, fogo=list(b.fogo),
                        timer=max(1, int(round(b.tempo_explosao / DT))), nivel=b.nivel)
        for b in bombas
    ]
    env._bomba_em = lambda x, y: any(b.x == x and b.y == y and not b.explodida for b in env.bombas)

    def zona_explosao(b):
        from bomberman_env import BombermanEnv
        return BombermanEnv.zona_explosao(env, b)
    env.zona_explosao = zona_explosao
    return env


def decidir_acao(player, mapa, jogadores, bombas, tempo_restante, pontos, hud_info, self_state):
    rede = _carregar()
    env = _snapshot(jogadores, mapa, bombas, tempo_restante)
    i = jogadores.index(player)
    obs = torch.from_numpy(observar(env, i)).unsqueeze(0)
    mask = torch.from_numpy(mascara_acoes(env, i)).unsqueeze(0)
    with torch.no_grad():
        logits, _ = rede(obs, mask)
    # Sorteia como no treino. argmax travava: 'parado' era a mais provavel em todo quadro.
    return acao_real(i, int(torch.distributions.Categorical(logits=logits).sample()))
