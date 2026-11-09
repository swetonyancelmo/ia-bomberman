"""Carrega um especialista treinado (coletor, assassino ou fugitivo) no formato do main.py.

Igual ao ia_treinada.py, mas com o arquivo de pesos do papel. So le o estado do
jogo (nunca altera), para nao disparar a deteccao de trapaca.
"""
import os

import torch

from ia_treinada import _snapshot
from bomberman_obs import observar, mascara_acoes, acao_real

PASTA = os.path.dirname(os.path.abspath(__file__))


def fazer(arquivo):
    """Devolve uma funcao decidir_acao que usa os pesos de `arquivo`."""
    rede = []  # carregada na primeira chamada

    def decidir_acao(player, mapa, jogadores, bombas, tempo_restante, pontos, hud_info, self_state):
        if not rede:
            from treino_ppo import Rede
            caminho = os.path.join(PASTA, arquivo)
            if not os.path.exists(caminho):  # papel ainda nao treinado: usa a rede geral
                caminho = os.path.join(PASTA, "modelo_bomberman.pt")
            r = Rede()
            r.load_state_dict(torch.load(caminho, weights_only=True))
            r.eval()
            rede.append(r)
        env = _snapshot(jogadores, mapa, bombas, tempo_restante)
        i = jogadores.index(player)
        obs = torch.from_numpy(observar(env, i)).unsqueeze(0)
        mask = torch.from_numpy(mascara_acoes(env, i)).unsqueeze(0)
        with torch.no_grad():
            logits, _ = rede[0](obs, mask)
        return acao_real(i, int(torch.distributions.Categorical(logits=logits).sample()))

    return decidir_acao
