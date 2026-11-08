"""Observacao, mascara de acoes e recompensa para o treino da IA."""
import numpy as np

from bomberman_env import (
    ACOES, DIRECOES, SPAWNS, ROWS, COLS, BLOCO, PAREDE, PU_BOMBA, PU_FOGO, CHAO,
    PASSOS_EXPLOSAO, PASSOS_PARTIDA, MAX_BOMBAS, NIVEL_MAX,
)

N_CANAIS = 13

# --- visao canonica -------------------------------------------------------
# Cada jogador nasce num canto. Para a rede aprender uma vez so, espelhamos o mapa
# de modo que TODOS se vejam no canto superior esquerdo (e as acoes sao traduzidas de volta).
TROCA_Y = [1, 0, 2, 3, 4, 5]   # cima <-> baixo
TROCA_X = [0, 1, 3, 2, 4, 5]   # esquerda <-> direita


def _espelhos(i):
    x, y, _ = SPAWNS[i]
    return x > COLS // 2, y > ROWS // 2   # (espelhar em x?, espelhar em y?)


def acao_real(i, idx):
    """Indice de acao na visao canonica do jogador i -> nome da acao de verdade."""
    fx, fy = _espelhos(i)
    if fx:
        idx = TROCA_X[idx]
    if fy:
        idx = TROCA_Y[idx]
    return ACOES[idx]


def observar(env, i):
    """Estado visto pelo jogador i -> array (N_CANAIS, ROWS, COLS).

    0 parede | 1 bloco | 2 power-up bomba | 3 power-up fogo
    4 bomba (urgencia) | 5 perigo de explosao (urgencia) | 6 eu | 7 aliado | 8 inimigo
    9-12: planos constantes com max_bombas, bombas livres, nivel e tempo restante
    """
    obs = np.zeros((N_CANAIS, ROWS, COLS), dtype=np.float32)
    mapa = np.asarray(env.mapa)
    obs[0] = mapa == PAREDE
    obs[1] = mapa == BLOCO
    obs[2] = mapa == PU_BOMBA
    obs[3] = mapa == PU_FOGO

    for b in env.bombas:
        if b.explodida:
            for x, y in b.fogo:
                obs[5, y, x] = 1.0
        else:
            urgencia = 1.0 - b.timer / PASSOS_EXPLOSAO  # 0 = acabou de por, 1 = vai explodir
            obs[4, b.y, b.x] = urgencia
            for x, y in env.zona_explosao(b):
                obs[5, y, x] = max(obs[5, y, x], urgencia)

    eu = env.jogadores[i]
    for j, p in enumerate(env.jogadores):
        if not p.vivo:
            continue
        if j == i:
            obs[6, p.y, p.x] = 1.0
        elif p.time == eu.time:
            obs[7, p.y, p.x] = 1.0
        else:
            obs[8, p.y, p.x] = 1.0

    obs[9] = eu.max_bombas / MAX_BOMBAS
    obs[10] = (eu.max_bombas - eu.bombas) / MAX_BOMBAS
    obs[11] = eu.nivel / NIVEL_MAX
    obs[12] = 1.0 - env.passo / PASSOS_PARTIDA
    fx, fy = _espelhos(i)
    if fy:
        obs = obs[:, ::-1, :]
    if fx:
        obs = obs[:, :, ::-1]
    return np.ascontiguousarray(obs)


def mascara_acoes(env, i):
    """Quais das 6 acoes fazem algo valido, na visao canonica (ver acao_real). 'parado' sempre vale."""
    p = env.jogadores[i]
    mask = np.ones(len(ACOES), dtype=bool)
    for k, nome in enumerate(ACOES):
        if nome in DIRECOES:
            dx, dy = DIRECOES[nome]
            nx, ny = p.x + dx, p.y + dy
            mask[k] = (
                0 <= nx < COLS and 0 <= ny < ROWS
                and env.mapa[ny][nx] in (CHAO, PU_BOMBA, PU_FOGO)
                and not env._bomba_em(nx, ny)
            )
        elif nome == "bomba":
            mask[k] = p.bombas < p.max_bombas and not env._bomba_em(p.x, p.y)
    fx, fy = _espelhos(i)
    if fy:
        mask = mask[TROCA_Y]
    if fx:
        mask = mask[TROCA_X]
    return mask


# ---------------------------------------------------------------- recompensa
ESCALA_PONTOS = 1000.0
BONUS_VITORIA = 3.0        # no lugar dos 10000 pontos do jogo (senao dominaria tudo)
PENALIDADE_MORTE = 0.5
PENALIDADE_PERIGO = 0.02   # por passo em tile que vai explodir em breve
BONUS_BLOCO = 0.2          # somado aos 0.1 dos pontos: bloco destruido vale 0.3
BONUS_POR_BOMBA = 0.05     # empurra a rede a experimentar bombas no comeco
PENALIDADE_PARADO = 0.005  # impede que "nao fazer nada" seja a estrategia otima
LIMIAR_PERIGO = 0.8       # urgencia a partir da qual estar no tile e arriscado


def recompensa(env, i, ganho_pontos, estava_vivo, acao="mover"):
    """Recompensa do jogador i no passo que acabou de ser executado.

    Usa os proprios pontos do jogo (escalados) mais um pouco de shaping para
    ensinar a fugir de bombas, que o placar sozinho demora muito a ensinar."""
    p = env.jogadores[i]
    r = ganho_pontos / ESCALA_PONTOS
    if acao == "parado" and p.vivo:
        r -= PENALIDADE_PARADO
    r += BONUS_BLOCO * env.blocos_passo[i]
    if acao == "bomba" and any(b.dono == i and (b.x, b.y) == (p.x, p.y)
                               and b.timer == PASSOS_EXPLOSAO - 1 for b in env.bombas):
        r += BONUS_POR_BOMBA  # so conta se a bomba foi realmente posta neste passo
    if ganho_pontos >= 10000:  # bonus de vitoria vem junto no delta de pontos
        r += BONUS_VITORIA - 10000 / ESCALA_PONTOS
    if estava_vivo and not p.vivo:
        r -= PENALIDADE_MORTE * env.fator_morte
    elif p.vivo:
        for b in env.bombas:
            if b.explodida:
                if (p.x, p.y) in b.fogo:
                    r -= PENALIDADE_PERIGO
                    break
            else:
                urgencia = 1.0 - b.timer / PASSOS_EXPLOSAO
                if urgencia >= LIMIAR_PERIGO and (p.x, p.y) in env.zona_explosao(b):
                    r -= PENALIDADE_PERIGO
                    break
    return r
