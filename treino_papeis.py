"""Treina 3 especialistas por PPO: coletor, assassino e fugitivo.

Cada partida tem 4 jogadores: os 3 especialistas (que aprendem, cada um com sua
rede e sua recompensa) e 1 "base" (a rede geral ja treinada, congelada), que faz
o papel do jogador 4. Os papeis sorteiam de canto a cada partida.

Uso:  py treino_papeis.py [iteracoes]          (padrao: 100)
      py treino_papeis.py 100 --ver              abre a janela para assistir o treino
      py treino_papeis.py 100 --retomar         continua dos modelo_<papel>.pt
Salva modelo_coletor.pt, modelo_assassino.pt e modelo_fugitivo.pt.
"""
import gc
import os
import random
import sys
import time

import numpy as np
import psutil
import torch
from torch.distributions import Categorical

from bomberman_env import BombermanEnv, ACOES
from bomberman_obs import observar, mascara_acoes, acao_real
from papeis import recompensa_papel, distancias
from treino_ppo import Rede, atualizar, N_ENVS, LR, PASSOS_TREINO, MEMORIA_MINIMA_MB

ESPECIALISTAS = ("coletor", "assassino", "fugitivo")
BASE = "modelo_bomberman.pt"
MAX_ITER = 150


def arquivo(papel):
    return f"modelo_{papel}.pt"


def carregar(caminho):
    r = Rede()
    r.load_state_dict(torch.load(caminho, weights_only=True))
    return r


def jogar_iteracao(redes, base, envs, viz=None):
    """Joga uma partida completa em cada env. Devolve as trajetorias de cada especialista."""
    for env in envs:
        env.reset()
    # quem joga em cada canto, sorteado por partida
    papeis = []
    for _ in envs:
        p = list(ESPECIALISTAS) + ["base"]
        random.shuffle(p)
        papeis.append(p)
    traj = {(e, i): dict(obs=[], mask=[], a=[], logp=[], v=[], r=[])
            for e in range(len(envs)) for i in range(4)}
    dist = {(e, i): distancias(envs[e], i) for e in range(len(envs)) for i in range(4)}

    while not all(env.fim for env in envs):
        ativos = [(e, i) for e, env in enumerate(envs) if not env.fim
                  for i, p in enumerate(env.jogadores) if p.vivo]
        obs = torch.from_numpy(np.stack([observar(envs[e], i) for e, i in ativos]))
        mask = torch.from_numpy(np.stack([mascara_acoes(envs[e], i) for e, i in ativos]))
        quem = [papeis[e][i] for e, i in ativos]

        acao = torch.zeros(len(ativos), dtype=torch.long)
        logp = torch.zeros(len(ativos))
        valor = torch.zeros(len(ativos))
        with torch.no_grad():
            for nome in ESPECIALISTAS + ("base",):
                modelo = base if nome == "base" else redes[nome]
                idx = [n for n, q in enumerate(quem) if q == nome]
                if idx:
                    logits, v = modelo(obs[idx], mask[idx])
                    dist_a = Categorical(logits=logits)
                    a = dist_a.sample()
                    acao[idx], logp[idx], valor[idx] = a, dist_a.log_prob(a), v

        escolhas = {}
        for n, (e, i) in enumerate(ativos):
            escolhas.setdefault(e, {})[i] = acao_real(i, acao[n].item())
            if quem[n] != "base":
                t = traj[(e, i)]
                t["obs"].append(obs[n].half()); t["mask"].append(mask[n].clone())
                t["a"].append(int(acao[n])); t["logp"].append(float(logp[n])); t["v"].append(float(valor[n]))

        for e, acoes in escolhas.items():
            env = envs[e]
            vivos_antes = [p.vivo for p in env.jogadores]
            env.step([acoes.get(i, "parado") for i in range(4)])
            for i, nome_acao in acoes.items():
                if papeis[e][i] != "base":
                    r, dist[(e, i)] = recompensa_papel(env, i, papeis[e][i], vivos_antes[i],
                                                       nome_acao, dist[(e, i)])
                    traj[(e, i)]["r"].append(r)
        if viz is not None and viz.ativo:
            # mostra a primeira partida que ainda nao acabou
            e_show = next((e for e, env in enumerate(envs) if not env.fim), len(envs) - 1)
            viz.desenhar(envs[e_show], papeis[e_show], e_show)
    return traj, papeis


def metricas(envs, papeis):
    """Por papel: media de blocos, power-ups, kills por partida e fracao que morreu."""
    acc = {n: dict(blocos=[], pu=[], kills=[], morreu=[]) for n in ESPECIALISTAS}
    for env, pap in zip(envs, papeis):
        for i, nome in enumerate(pap):
            if nome == "base":
                continue
            acc[nome]["blocos"].append(env.blocos_total[i])
            acc[nome]["pu"].append(env.pu_total[i])
            acc[nome]["kills"].append(env.kills_total[i])
            acc[nome]["morreu"].append(0 if env.jogadores[i].vivo else 1)
    return {n: {k: float(np.mean(v)) for k, v in d.items()} for n, d in acc.items()}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    retomar = "--retomar" in sys.argv
    viz = None
    if "--ver" in sys.argv:
        from visualizador import Visualizador
        viz = Visualizador()
    iteracoes = int(args[0]) if args else MAX_ITER
    torch.manual_seed(0)
    random.seed(0)

    base = carregar(BASE)
    base.eval()
    redes, otim = {}, {}
    for nome in ESPECIALISTAS:
        # comeca da rede geral (ja sabe por bomba e fugir) e so se especializa
        redes[nome] = carregar(arquivo(nome) if retomar and os.path.exists(arquivo(nome)) else BASE)
        otim[nome] = torch.optim.Adam(redes[nome].parameters(), lr=LR)
    envs = [BombermanEnv(seed=k, max_passos=PASSOS_TREINO) for k in range(N_ENVS)]

    t0 = time.time()
    for it in range(1, iteracoes + 1):
        if viz is not None:
            viz.contexto = dict(fase="papeis", it=it)
        traj, papeis = jogar_iteracao(redes, base, envs, viz)
        m = metricas(envs, papeis)
        n_amostras = {}
        for nome in ESPECIALISTAS:
            sub = {k: t for k, t in traj.items() if papeis[k[0]][k[1]] == nome and t["r"]}
            retorno = np.mean([sum(t["r"]) for t in sub.values()]) if sub else 0.0
            m[nome]["retorno"] = float(retorno)
            n_amostras[nome] = atualizar(redes[nome], otim[nome], sub) if sub else 0
        if viz is not None:
            viz.registrar_iteracao(np.mean([m[n]["blocos"] for n in ESPECIALISTAS]),
                                  np.mean([m[n]["retorno"] for n in ESPECIALISTAS]),
                                  np.mean([env.passo for env in envs]))
        del traj
        gc.collect()

        livre_mb = psutil.virtual_memory().available / 2**20
        print(f"it {it:3d} | {time.time() - t0:5.0f}s | PC livre {livre_mb:5.0f}MB", flush=True)
        for nome in ESPECIALISTAS:
            d = m[nome]
            print(f"   {nome:9s} retorno {d['retorno']:+6.2f} | blocos {d['blocos']:4.1f} | "
                  f"powerups {d['pu']:4.2f} | kills {d['kills']:4.2f} | morreu {d['morreu']:4.0%}", flush=True)
        if it % 5 == 0 or it == iteracoes or livre_mb < MEMORIA_MINIMA_MB:
            for nome in ESPECIALISTAS:
                torch.save(redes[nome].state_dict(), arquivo(nome))
        if livre_mb < MEMORIA_MINIMA_MB:
            print("RAM livre baixa: modelos salvos, treino interrompido com seguranca.", flush=True)
            return


if __name__ == "__main__":
    main()
