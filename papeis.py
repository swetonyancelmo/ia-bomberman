"""Papeis dos jogadores e a recompensa de cada um.

Cada papel usa a MESMA rede (mesma observacao e mesmas acoes). O que muda e so
a recompensa, que empurra a rede para um estilo de jogo:

    coletor   - corre atras de power-ups
    assassino - quer matar qualquer jogador (inclusive o parceiro de time)
    fugitivo  - quer sobreviver; foge de bombas e de inimigos
    aprendiz  - recompensa equilibrada, usada pelo jogador 4 que aprende jogando
"""
from bomberman_env import PU_BOMBA, PU_FOGO, PASSOS_EXPLOSAO
from bomberman_obs import LIMIAR_PERIGO

PAPEIS = ("coletor", "assassino", "fugitivo", "aprendiz")

# Pesos de cada papel. Todos os valores estao na mesma escala do treino original
# (1.0 = 1000 pontos do jogo).
CONFIG = {
    # bloco/bomba quase zerados nos especialistas: o que define o papel e o sinal proprio.
    # bomba: bonus por soltar bomba; bomba_inimigo: bonus por inimigo dentro da area da bomba solta.
    "coletor": dict(
        bloco=0.03, powerup=2.0, matar=0.1, morte=0.5, vitoria=1.0, perigo=0.02, vivo=0.0,
        aprox_pu=0.08, aprox_inim=0.0, afasta_inim=0.0, bomba=0.01, bomba_inimigo=0.0,
    ),
    "assassino": dict(
        bloco=0.0, powerup=0.05, matar=4.0, morte=0.3, vitoria=3.0, perigo=0.015, vivo=0.0,
        aprox_pu=0.0, aprox_inim=0.08, afasta_inim=0.0, bomba=0.02, bomba_inimigo=0.4,
    ),
    "fugitivo": dict(
        bloco=0.0, powerup=0.05, matar=0.1, morte=3.0, vitoria=3.0, perigo=0.15, vivo=0.01,
        aprox_pu=0.0, aprox_inim=0.0, afasta_inim=0.03, bomba=0.0, bomba_inimigo=0.0,
    ),
    "aprendiz": dict(
        bloco=0.30, powerup=0.30, matar=1.0, morte=0.5, vitoria=3.0, perigo=0.02, vivo=0.0,
        aprox_pu=0.01, aprox_inim=0.005, afasta_inim=0.0, bomba=0.05, bomba_inimigo=0.0,
    ),
}

PENALIDADE_PARADO = 0.005


def _dist_mais_proximo(env, i, alvos):
    p = env.jogadores[i]
    melhor = None
    for (x, y) in alvos:
        d = abs(x - p.x) + abs(y - p.y)
        if melhor is None or d < melhor:
            melhor = d
    return melhor


def distancias(env, i):
    """(distancia ao power-up mais proximo, distancia ao inimigo mais proximo); None se nao houver."""
    pus = [(x, y) for y, linha in enumerate(env.mapa) for x, t in enumerate(linha)
           if t in (PU_BOMBA, PU_FOGO)]
    inimigos = [(q.x, q.y) for j, q in enumerate(env.jogadores)
                if j != i and q.vivo and q.time != env.jogadores[i].time]
    return _dist_mais_proximo(env, i, pus), _dist_mais_proximo(env, i, inimigos)


def em_perigo(env, i):
    p = env.jogadores[i]
    for b in env.bombas:
        if b.explodida:
            if (p.x, p.y) in b.fogo:
                return True
        elif 1.0 - b.timer / PASSOS_EXPLOSAO >= LIMIAR_PERIGO and (p.x, p.y) in env.zona_explosao(b):
            return True
    return False


def _delta_dist(antes, agora):
    """Quanto a distancia diminuiu (positivo = chegou mais perto), limitado a +-1."""
    if antes is None or agora is None:
        return 0.0
    return max(-1.0, min(1.0, antes - agora))


def recompensa_papel(env, i, papel, estava_vivo, acao, dist_antes):
    """Recompensa do jogador i neste passo, segundo o papel.

    dist_antes: resultado de distancias(env, i) ANTES do passo (para o shaping).
    Devolve (recompensa, dist_agora); guarde dist_agora para o proximo passo."""
    c = CONFIG[papel]
    p = env.jogadores[i]
    agora = distancias(env, i)
    r = 0.0
    r += c["bloco"] * env.blocos_passo[i]
    r += c["powerup"] * env.pu_passo[i]
    r += c["matar"] * env.kills_passo[i]
    if acao == "parado" and p.vivo:
        r -= PENALIDADE_PARADO
    nova = [b for b in env.bombas if acao == "bomba" and b.dono == i and (b.x, b.y) == (p.x, p.y)
            and b.timer == PASSOS_EXPLOSAO - 1]  # bomba solta neste passo
    if nova:
        r += c["bomba"]
        zona = set(env.zona_explosao(nova[0]))
        r += c["bomba_inimigo"] * sum(1 for j, q in enumerate(env.jogadores)
                                     if j != i and q.vivo and (q.x, q.y) in zona)
    if env.fim and env.vencedor == i:
        r += c["vitoria"]
    if estava_vivo and not p.vivo:
        r -= c["morte"] * env.fator_morte
    elif p.vivo:
        r += c["vivo"]
        if em_perigo(env, i):
            r -= c["perigo"]
        r += c["aprox_pu"] * _delta_dist(dist_antes[0], agora[0])
        r += c["aprox_inim"] * _delta_dist(dist_antes[1], agora[1])
        r -= c["afasta_inim"] * _delta_dist(dist_antes[1], agora[1])
    return r, agora
