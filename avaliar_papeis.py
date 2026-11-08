"""Avalia cada especialista: ele + 3 redes base vs. controle (4 redes base).

Uso: py avaliar_papeis.py [sufixo] [partidas]     ex.: py avaliar_papeis.py _v1 64
Mede por jogador: power-ups, kills, blocos, taxa de morte. Se o especialista nao
for melhor que a base na metrica do papel, o treino nao funcionou.
"""
import sys
import numpy as np
import torch
from torch.distributions import Categorical

from bomberman_env import BombermanEnv
from bomberman_obs import observar, mascara_acoes, acao_real
from treino_ppo import Rede, PASSOS_TREINO

PAPEIS = ("coletor", "assassino", "fugitivo")


def carregar(caminho):
    r = Rede(); r.load_state_dict(torch.load(caminho, weights_only=True)); r.eval(); return r


def jogar(espec, base, n, seed):
    """espec=None -> controle (4 bases). Devolve metricas do jogador 'alvo' (ou media dos 4 no controle)."""
    envs = [BombermanEnv(seed=seed + k, max_passos=PASSOS_TREINO) for k in range(n)]
    alvo = [k % 4 for k in range(n)]  # canto do especialista, rodando entre partidas
    while not all(e.fim for e in envs):
        ativos = [(e, i) for e, env in enumerate(envs) if not env.fim
                  for i, p in enumerate(env.jogadores) if p.vivo]
        obs = torch.from_numpy(np.stack([observar(envs[e], i) for e, i in ativos]))
        mask = torch.from_numpy(np.stack([mascara_acoes(envs[e], i) for e, i in ativos]))
        acao = torch.zeros(len(ativos), dtype=torch.long)
        with torch.no_grad():
            for modelo, sel in ((espec, [n_ for n_, (e, i) in enumerate(ativos) if alvo[e] == i]),
                                (base, [n_ for n_, (e, i) in enumerate(ativos) if espec is None or alvo[e] != i])):
                if modelo is not None and sel:
                    acao[sel] = Categorical(logits=modelo(obs[sel], mask[sel])[0]).sample()
        esc = {}
        for k, (e, i) in enumerate(ativos):
            esc.setdefault(e, {})[i] = acao_real(i, acao[k].item())
        for e, a in esc.items():
            envs[e].step([a.get(i, "parado") for i in range(4)])
    linhas = []
    for e, env in enumerate(envs):
        for i in ([alvo[e]] if espec is not None else range(4)):
            linhas.append((env.pu_total[i], env.kills_total[i], env.blocos_total[i], 0 if env.jogadores[i].vivo else 1))
    return np.mean(linhas, axis=0)


def main():
    suf = sys.argv[1] if len(sys.argv) > 1 else ""
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    torch.manual_seed(1)
    base = carregar("modelo_bomberman.pt")
    c = jogar(None, base, n, 5000)
    print(f"{'':10s} powerups  kills  blocos  morreu")
    print(f"{'base':10s} {c[0]:8.2f} {c[1]:6.2f} {c[2]:7.1f} {c[3]:6.0%}")
    for p in PAPEIS:
        m = jogar(carregar(f"modelo_{p}{suf}.pt"), base, n, 5000)
        print(f"{p:10s} {m[0]:8.2f} {m[1]:6.2f} {m[2]:7.1f} {m[3]:6.0%}")


if __name__ == "__main__":
    main()
