"""Jogador 4: aprende por reforco (PPO) ENQUANTO joga.

Parte da rede geral (ou dos pesos que ele mesmo salvou em partidas anteriores) e
a cada TAM_LOTE decisoes faz uma atualizacao PPO. A recompensa vem so do que o
jogo mostra: variacao dos pontos, perigo, ficar parado e morrer.

O main.py nao chama a IA depois que o jogador morre nem depois do fim da partida,
entao o resultado final (morte, pontos, vitoria) e lido ao fechar o programa
(atexit), e os pesos sao gravados em modelo_aprendiz.pt para a proxima partida.
"""
import atexit
import os

import numpy as np
import torch
from torch.distributions import Categorical

from ia_treinada import _snapshot
from bomberman_obs import observar, mascara_acoes, acao_real
from papeis import em_perigo, CONFIG, PENALIDADE_PARADO
from treino_ppo import Rede, GAMMA, LAMBDA, CLIP, COEF_VALOR, COEF_ENTROPIA

PASTA = os.path.dirname(os.path.abspath(__file__))
SAIDA = os.path.join(PASTA, "modelo_aprendiz.pt")
TAM_LOTE = 256        # decisoes entre atualizacoes
MIN_FINAL = 32        # atualizacao final so se tiver amostras suficientes
LR_ONLINE = 1e-4      # menor que no treino: aprender com poucos dados e ruidoso
EPOCAS = 2
MINIBATCH = 64
INTERVALO = 0.09      # s de jogo entre decisoes (o ambiente de treino decide a cada 0,1 s)
MORTE = CONFIG["aprendiz"]["morte"]

_e = {}  # estado do aprendiz (so existe um por processo)


def _iniciar():
    rede = Rede()
    origem = SAIDA if os.path.exists(SAIDA) else os.path.join(PASTA, "modelo_bomberman.pt")
    rede.load_state_dict(torch.load(origem, weights_only=True))
    rede.eval()
    _e.update(rede=rede, otim=torch.optim.Adam(rede.parameters(), lr=LR_ONLINE),
              lote=[], pend=None, ult_t=None, ult_acao=None, ult_pontos=0,
              player=None, pontos=None, indice=0, atualizacoes=0)
    atexit.register(_fim)


def _recompensa(env, i, dpontos, acao_ant):
    r = min(dpontos / 1000.0, 3.0)  # vitoria (+10000) vira no maximo +3
    if acao_ant == "parado":
        r -= PENALIDADE_PARADO
    if env is not None and em_perigo(env, i):
        r -= CONFIG["aprendiz"]["perigo"]
    return r


def _atualizar(ultimo_valor):
    """Uma atualizacao PPO com o lote atual. ultimo_valor: valor estimado apos a ultima amostra."""
    lote = _e["lote"]
    if not lote:
        return
    r = np.array([t["r"] for t in lote], dtype=np.float32)
    v = np.array([t["v"] for t in lote], dtype=np.float32)
    adv = np.zeros_like(r)
    ultima = 0.0
    for k in reversed(range(len(r))):
        prox = v[k + 1] if k + 1 < len(r) else ultimo_valor
        delta = r[k] + GAMMA * prox - v[k]
        ultima = delta + GAMMA * LAMBDA * ultima
        adv[k] = ultima
    ret = torch.from_numpy(adv + v)
    adv_t = torch.from_numpy(adv)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
    obs = torch.stack([t["obs"] for t in lote])
    mask = torch.stack([t["mask"] for t in lote])
    a = torch.tensor([t["a"] for t in lote])
    logp = torch.tensor([t["logp"] for t in lote], dtype=torch.float32)

    rede, otim = _e["rede"], _e["otim"]
    rede.train()
    for _ in range(EPOCAS):
        for idx in torch.randperm(len(a)).split(MINIBATCH):
            logits, valor = rede(obs[idx], mask[idx])
            dist = Categorical(logits=logits)
            razao = (dist.log_prob(a[idx]) - logp[idx]).exp()
            perda_pol = -torch.min(razao * adv_t[idx],
                                   razao.clamp(1 - CLIP, 1 + CLIP) * adv_t[idx]).mean()
            perda = (perda_pol + COEF_VALOR * (valor - ret[idx]).pow(2).mean()
                     - COEF_ENTROPIA * dist.entropy().mean())
            otim.zero_grad()
            perda.backward()
            torch.nn.utils.clip_grad_norm_(rede.parameters(), 0.5)
            otim.step()
    rede.eval()
    _e["lote"] = []
    _e["atualizacoes"] += 1
    torch.save(rede.state_dict(), SAIDA)


def _fim():
    """Ao fechar o programa: fecha a ultima decisao com o resultado final e salva os pesos."""
    if not _e or _e.get("pend") is None:
        return
    try:
        p = _e["pend"]
        d = _e["pontos"][_e["indice"]] - _e["ult_pontos"]
        p["r"] = _recompensa(None, 0, d, _e["ult_acao"])
        if not _e["player"].ativo:
            p["r"] -= MORTE  # morreu (o jogo nao chama a IA depois disso)
        _e["lote"].append(p)
        _e["pend"] = None
        if len(_e["lote"]) >= MIN_FINAL:
            _atualizar(0.0)  # a partida acabou para ele: valor futuro 0
        else:
            torch.save(_e["rede"].state_dict(), SAIDA)
    except Exception as erro:  # nunca atrapalhar o fechamento do jogo
        print("aprendiz: falha ao salvar:", erro)


def decidir_acao(player, mapa, jogadores, bombas, tempo_restante, pontos, hud_info, self_state):
    if not _e:
        _iniciar()
    # sem decisao nova em menos de 0,09 s se ele estava parado (senao seriam 60 decisoes/s)
    if (_e["ult_acao"] == "parado" and _e["ult_t"] is not None
            and _e["ult_t"] - tempo_restante < INTERVALO):
        return "parado"

    env = _snapshot(jogadores, mapa, bombas, tempo_restante)
    i = jogadores.index(player)
    _e.update(player=player, pontos=pontos, indice=i)

    # 1) fecha a decisao anterior com o que aconteceu desde entao
    if _e["pend"] is not None:
        p = _e["pend"]
        p["r"] = _recompensa(env, i, pontos[i] - _e["ult_pontos"], _e["ult_acao"])
        _e["lote"].append(p)
        _e["pend"] = None

    # 2) decide
    obs = torch.from_numpy(observar(env, i)).unsqueeze(0)
    mask = torch.from_numpy(mascara_acoes(env, i)).unsqueeze(0)
    with torch.no_grad():
        logits, valor = _e["rede"](obs, mask)
        dist = Categorical(logits=logits)
        a = dist.sample()
        logp = dist.log_prob(a)

    # 3) atualiza se o lote encheu (usa o valor do estado atual como continuacao)
    if len(_e["lote"]) >= TAM_LOTE:
        _atualizar(float(valor))

    acao = acao_real(i, int(a))
    _e["pend"] = dict(obs=obs[0], mask=mask[0], a=int(a), logp=float(logp), v=float(valor), r=0.0)
    _e.update(ult_t=tempo_restante, ult_acao=acao, ult_pontos=pontos[i])
    return acao
