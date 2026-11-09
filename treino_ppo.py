"""Treino PPO com self-play: uma unica rede controla os 4 jogadores.

Uso:  py treino_ppo.py [iteracoes]
Salva os pesos em modelo_bomberman.pt (e um checkpoint a cada 10 iteracoes).
"""
import copy
import gc
import os
import random
import sys
import time

import numpy as np
import psutil
import torch
import torch.nn as nn
from torch.distributions import Categorical

from bomberman_env import BombermanEnv, ACOES, SPAWNS
from bomberman_obs import N_CANAIS, observar, mascara_acoes, recompensa, acao_real

N_ENVS = 16          # partidas simultaneas por iteracao
GAMMA = 0.99
LAMBDA = 0.95
CLIP = 0.2
EPOCAS = 4
MINIBATCH = 512
LR = 3e-4
COEF_VALOR = 0.5
COEF_ENTROPIA = 0.01
ARQUIVO = "modelo_bomberman.pt"
MEMORIA_MINIMA_MB = 1500   # se a RAM livre do PC cair abaixo disso, salva e para (em vez de o sistema matar o treino)
PASSOS_TREINO = 600     # partidas de treino mais curtas que as de 1800 passos
ATUALIZA_ANTIGA = 20    # a cada N iteracoes a "rede antiga" vira uma copia da atual


class Rede(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(N_CANAIS, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 11 * 13, 256), nn.ReLU(),
        )
        self.politica = nn.Linear(256, len(ACOES))
        self.valor = nn.Linear(256, 1)

    def forward(self, obs, mask):
        h = self.conv(obs)
        logits = self.politica(h).masked_fill(~mask, -1e9)
        return logits, self.valor(h).squeeze(-1)


def sortear_controle(env=None):
    """Quem controla cada jogador numa partida. 'rede' aprende; os outros so atuam.

    50% self-play puro; 25% times 1 contra aleatorios; 25% contra a rede antiga."""
    r = random.random()
    if r < 0.5:
        return ["rede"] * 4
    adversario = "aleatorio" if r < 0.75 else "antiga"
    return ["rede" if time_ == 0 else adversario for _, _, time_ in SPAWNS]


def jogar_iteracao(rede, antiga, envs, sortear=None, viz=None):
    """Joga N_ENVS partidas inteiras e devolve as trajetorias dos jogadores da rede."""
    for env in envs:
        env.reset()
    controle = [(sortear or sortear_controle)(env) for env in envs]
    traj = {(e, i): dict(obs=[], mask=[], a=[], logp=[], v=[], r=[])
            for e in range(len(envs)) for i in range(4)}
    contagem = np.zeros(len(ACOES))

    while not all(env.fim for env in envs):
        ativos = [(e, i) for e, env in enumerate(envs) if not env.fim
                  for i, p in enumerate(env.jogadores) if p.vivo]
        obs = torch.from_numpy(np.stack([observar(envs[e], i) for e, i in ativos]))
        mask = torch.from_numpy(np.stack([mascara_acoes(envs[e], i) for e, i in ativos]))
        quem = [controle[e][i] for e, i in ativos]

        acao = torch.zeros(len(ativos), dtype=torch.long)
        logp = torch.zeros(len(ativos))
        valor = torch.zeros(len(ativos))
        with torch.no_grad():
            for nome, modelo in (("rede", rede), ("antiga", antiga)):
                idx = [n for n, q in enumerate(quem) if q == nome]
                if idx:
                    logits, v = modelo(obs[idx], mask[idx])
                    dist = Categorical(logits=logits)
                    a = dist.sample()
                    acao[idx], logp[idx], valor[idx] = a, dist.log_prob(a), v
            for n, q in enumerate(quem):
                if q == "aleatorio":
                    acao[n] = int(torch.multinomial(mask[n].float(), 1))

        escolhas = {}
        for n, (e, i) in enumerate(ativos):
            nome_acao = acao_real(i, acao[n].item())
            escolhas.setdefault(e, {})[i] = nome_acao
            if quem[n] == "rede":
                t = traj[(e, i)]
                t["obs"].append(obs[n].half()); t["mask"].append(mask[n].clone()); t["a"].append(int(acao[n]))
                t["logp"].append(float(logp[n])); t["v"].append(float(valor[n]))
                contagem[acao[n].item()] += 1

        for e, acoes in escolhas.items():
            env = envs[e]
            vivos_antes = [p.vivo for p in env.jogadores]
            ganhos, _ = env.step([acoes.get(i, "parado") for i in range(4)])
            for i, nome_acao in acoes.items():
                if controle[e][i] == "rede":
                    traj[(e, i)]["r"].append(
                        recompensa(env, i, ganhos[i], vivos_antes[i], nome_acao))
        if viz is not None and viz.ativo:
            # mostra a primeira partida que ainda nao acabou
            e_show = next((e for e, env in enumerate(envs) if not env.fim), len(envs) - 1)
            viz.desenhar(envs[e_show], controle[e_show], e_show)
    return traj, contagem / max(1, contagem.sum())


def vantagens(t):
    """GAE de uma trajetoria. O valor apos o ultimo passo e 0 (morreu ou acabou)."""
    v = np.array(t["v"], dtype=np.float32)
    r = np.array(t["r"], dtype=np.float32)
    adv = np.zeros_like(r)
    ultima = 0.0
    for k in reversed(range(len(r))):
        prox = v[k + 1] if k + 1 < len(r) else 0.0
        delta = r[k] + GAMMA * prox - v[k]
        ultima = delta + GAMMA * LAMBDA * ultima
        adv[k] = ultima
    return adv, adv + v


def atualizar(rede, otimizador, traj):
    obs, mask, a, logp, adv, ret = [], [], [], [], [], []
    for t in traj.values():
        if not t["r"]:
            continue
        ad, rt = vantagens(t)
        obs += t["obs"]; mask += t["mask"]; a += t["a"]; logp += t["logp"]
        adv.append(ad); ret.append(rt)
    obs = torch.stack(obs); mask = torch.stack(mask)  # obs fica em float16 (metade da memoria)
    a = torch.tensor(a); logp = torch.tensor(logp, dtype=torch.float32)
    adv = torch.from_numpy(np.concatenate(adv)); ret = torch.from_numpy(np.concatenate(ret))
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    n = len(a)
    for _ in range(EPOCAS):
        for idx in torch.randperm(n).split(MINIBATCH):
            logits, valor = rede(obs[idx].float(), mask[idx])
            dist = Categorical(logits=logits)
            razao = (dist.log_prob(a[idx]) - logp[idx]).exp()
            perda_pol = -torch.min(razao * adv[idx],
                                   razao.clamp(1 - CLIP, 1 + CLIP) * adv[idx]).mean()
            perda_val = (valor - ret[idx]).pow(2).mean()
            perda = perda_pol + COEF_VALOR * perda_val - COEF_ENTROPIA * dist.entropy().mean()
            otimizador.zero_grad()
            perda.backward()
            nn.utils.clip_grad_norm_(rede.parameters(), 0.5)
            otimizador.step()
    return n


def controle_vs_aleatorios(env=None):
    return ["rede" if time_ == 0 else "aleatorio" for _, _, time_ in SPAWNS]


def controle_solo(env):
    return ["rede" if i == env.solo_idx else "aleatorio" for i in range(4)]  # os outros nem existem


# (nome, n_envs, solo, funcao de sorteio, iteracoes maximas)
def fases(max1, max2, max3):
    return [
        ("1-solo", 24, True, controle_solo, max1),
        ("2-vs-aleatorios", N_ENVS, False, controle_vs_aleatorios, max2),
        ("3-misto", N_ENVS, False, None, max3),
    ]


META_BLOCOS = 20  # fase 1 termina quando a media de blocos por partida passa disso (3x seguidas)


def main():
    args = sys.argv[1:]
    retomar = None
    if "--retomar" in args:
        k = args.index("--retomar")
        retomar = args[k + 1]
        del args[k:k + 2]
    viz = None
    if "--ver" in args:
        args.remove("--ver")
        from visualizador import Visualizador
        viz = Visualizador()
    max1, max2, max3 = (int(a) for a in args[:3]) if len(args) >= 3 else (150, 80, 200)
    torch.manual_seed(0)
    rede = Rede()
    if retomar:
        rede.load_state_dict(torch.load(retomar, weights_only=True))
        print(f"retomando de {retomar}", flush=True)
    otimizador = torch.optim.Adam(rede.parameters(), lr=LR)
    antiga = copy.deepcopy(rede)
    t0 = time.time()
    it = 0
    for nome, n_envs, solo, sortear, maximo in fases(max1, max2, max3):
        print(f"=== fase {nome} (ate {maximo} iteracoes) ===", flush=True)
        envs = [BombermanEnv(seed=1000 * it + k, max_passos=PASSOS_TREINO, solo=solo)
                for k in range(n_envs)]
        seguidas = 0
        for _ in range(maximo):
            it += 1
            if viz is not None:
                viz.contexto = dict(fase=nome, it=it)
            traj, freq = jogar_iteracao(rede, antiga, envs, sortear, viz)
            duracao = np.mean([env.passo for env in envs])
            retorno = np.mean([sum(t["r"]) for t in traj.values() if t["r"]])
            blocos = np.mean([env.blocos_total[i] for e, env in enumerate(envs)
                              for i in range(4) if traj[(e, i)]["r"]])
            suic = np.mean([env.suicidios for env in envs])
            n = atualizar(rede, otimizador, traj)
            del traj
            gc.collect()
            if viz is not None:
                viz.registrar_iteracao(blocos, retorno, duracao)
            livre_mb = psutil.virtual_memory().available / 2**20
            usado_mb = psutil.Process(os.getpid()).memory_info().rss / 2**20
            print(f"it {it:4d} | amostras {n:6d} | duracao {duracao:5.1f} | retorno {retorno:+.3f} | "
                  f"blocos {blocos:5.1f} | suicidios/partida {suic:4.2f} | "
                  f"parado {freq[5]:4.0%} bomba {freq[4]:4.0%} | mem {usado_mb:4.0f}MB (PC livre {livre_mb:5.0f}MB) | "
                  f"{time.time() - t0:5.0f}s", flush=True)
            if livre_mb < MEMORIA_MINIMA_MB:
                torch.save(rede.state_dict(), ARQUIVO)
                print(f"RAM livre baixa ({livre_mb:.0f}MB): modelo salvo, treino interrompido com seguranca.", flush=True)
                return
            if it % ATUALIZA_ANTIGA == 0:
                antiga = copy.deepcopy(rede)
            if it % 10 == 0:
                torch.save(rede.state_dict(), ARQUIVO)
            if solo:
                seguidas = seguidas + 1 if blocos >= META_BLOCOS else 0
                if seguidas >= 3:
                    print(f"meta da fase {nome} atingida", flush=True)
                    break
        torch.save(rede.state_dict(), ARQUIVO)
        torch.save(rede.state_dict(), f"modelo_fase_{nome}.pt")


if __name__ == "__main__":
    main()
