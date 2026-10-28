"""
IA do Bomberman - protótipo (Acompanhamento 2)

Arquitetura: agente reativo baseado em PRIORIDADES + busca em largura (BFS)
com noção de TEMPO (sabe quando cada casa vai pegar fogo).

A cada chamada a IA percorre a lista de prioridades e executa a primeira
regra que se aplica:
    1. FUGIR      - estou numa casa que vai pegar fogo -> ir para casa segura
    2. POWER-UP   - existe power-up alcançável com segurança -> ir buscar
    3. BOMBARDEAR - daqui a bomba destrói blocos/acerta inimigo E tenho rota de fuga -> "bomba"
    4. POSICIONAR - ir até a melhor casa para soltar a próxima bomba
    5. CAÇAR      - não há blocos úteis -> aproximar do inimigo mais próximo
    6. ANTI-OCIOSO - se ficou parado tempo demais, dá um passo seguro aleatório

Importante: a IA nunca altera os objetos recebidos (o main.py confere com
checksum e pune quem modificar mapa, jogadores, bombas ou pontos).
"""
import random
from collections import deque

VAZIO, BLOCO, PAREDE, PU_BOMBA, PU_FOGO = 0, 1, 2, 3, 4
ANDAVEL = (VAZIO, PU_BOMBA, PU_FOGO)
DIRECOES = {"cima": (0, -1), "baixo": (0, 1), "esquerda": (-1, 0), "direita": (1, 0)}

MARGEM = 0.15        # folga de segurança (segundos) nas janelas de fogo
LIMITE_OCIOSO = 30   # chamadas seguidas em "parado" antes de forçar um passo

# Pesos da função de ganho de uma bomba
GANHO_BLOCO = 100
GANHO_INIMIGO = 300
PERDA_POWERUP = 150  # perde o power-up (+200 não ganhos) e ainda leva -50

_ocioso = {}  # id(player) -> quantas vezes seguidas ficou parado


def alcance(nivel):
    # mesma fórmula do main.py (Bomba.explodir)
    return 1 + (nivel - 1) * 2


def area_explosao(mapa, x, y, nivel, rows, cols):
    """Casas atingidas por uma bomba em (x, y). Reproduz Bomba.explodir()."""
    casas = [(x, y)]
    for dx, dy in DIRECOES.values():
        for i in range(1, alcance(nivel) + 1):
            nx, ny = x + dx * i, y + dy * i
            if not (0 <= nx < cols and 0 <= ny < rows) or mapa[ny][nx] == PAREDE:
                break
            casas.append((nx, ny))
            if mapa[ny][nx] == BLOCO:
                break
    return casas


def mapa_perigo(mapa, bombas, rows, cols, tempo_fogo, extra=None):
    """
    Retorna {(x, y): [(inicio, fim), ...]}: intervalos de tempo (a partir de
    agora) em que cada casa estará pegando fogo. Considera reação em cadeia
    (uma bomba atingida pela explosão de outra explode junto).
    `extra` = (x, y, nivel, tempo) simula uma bomba que ainda não existe.
    """
    pendentes = [[b.x, b.y, b.nivel, b.tempo_explosao] for b in bombas if not b.explodida]
    if extra:
        pendentes.append(list(extra))
    areas = [set(area_explosao(mapa, bx, by, nv, rows, cols)) for bx, by, nv, _ in pendentes]

    # Reação em cadeia: se A atinge B e A explode antes, B explode junto com A
    mudou = True
    while mudou:
        mudou = False
        for i, a in enumerate(pendentes):
            for b in pendentes:
                if b[3] > a[3] and (b[0], b[1]) in areas[i]:
                    b[3] = a[3]
                    mudou = True

    perigo = {}
    for (_, _, _, t), area in zip(pendentes, areas):
        for casa in area:
            perigo.setdefault(casa, []).append((t, t + tempo_fogo))
    # Fogo que já está queimando agora
    for b in bombas:
        if b.explodida and b.tempo_fogo > 0:
            for casa in b.fogo:
                perigo.setdefault(casa, []).append((0, b.tempo_fogo))
    return perigo


def livre(perigo, casa, t0, t1):
    """A casa está sem fogo durante todo o intervalo [t0, t1]?"""
    return all(fim + MARGEM < t0 or ini - MARGEM > t1 for ini, fim in perigo.get(casa, ()))


def explorar(origem, mapa, bloqueio, perigo, passo, rows, cols):
    """
    BFS a partir de `origem`. Só entra numa casa se ela estiver sem fogo no
    intervalo em que o personagem estaria nela (passo d -> d+1).
    Retorna {casa: (distancia, primeira_acao_para_chegar_la)}.
    """
    info = {origem: (0, None)}
    fila = deque([origem])
    while fila:
        casa = fila.popleft()
        d, primeira = info[casa]
        for nome, (dx, dy) in DIRECOES.items():
            n = (casa[0] + dx, casa[1] + dy)
            if n in info or not (0 <= n[0] < cols and 0 <= n[1] < rows):
                continue
            if mapa[n[1]][n[0]] not in ANDAVEL or n in bloqueio:
                continue
            if not livre(perigo, n, d * passo, (d + 1) * passo):
                continue
            info[n] = (d + 1, primeira or nome)
            fila.append(n)
    return info


def ganho_bomba(mapa, casa, nivel, inimigos, perigo, rows, cols):
    """Quanto vale soltar uma bomba em `casa`."""
    ganho = 0
    for x, y in area_explosao(mapa, casa[0], casa[1], nivel, rows, cols):
        if mapa[y][x] == BLOCO and (x, y) not in perigo:  # bloco que outra bomba já vai quebrar não conta
            ganho += GANHO_BLOCO
        elif mapa[y][x] in (PU_BOMBA, PU_FOGO):
            ganho -= PERDA_POWERUP
        if (x, y) in inimigos:
            ganho += GANHO_INIMIGO
    return ganho


def tem_fuga(casa, mapa, bombas, bloqueio, nivel, ctx):
    """Se eu soltar uma bomba em `casa`, ainda consigo chegar a uma casa segura?"""
    rows, cols, passo, tempo_fogo, tempo_explosao = ctx
    perigo = mapa_perigo(mapa, bombas, rows, cols, tempo_fogo,
                         extra=(casa[0], casa[1], nivel, tempo_explosao))
    alcancaveis = explorar(casa, mapa, bloqueio | {casa}, perigo, passo, rows, cols)
    return any(c not in perigo for c in alcancaveis)


def escolher_acao(player, mapa, jogadores, bombas, hud_info, self_state):
    rows, cols = hud_info['rows'], hud_info['cols']
    passo = hud_info['tempo_movimento'] + 0.02  # um passo leva ~6 frames a 60 FPS
    tempo_fogo = hud_info['tempo_fogo']
    ctx = (rows, cols, passo, tempo_fogo, hud_info['tempo_explosao'])

    pos = (self_state['grid_x'], self_state['grid_y'])
    nivel = self_state['bomba_nivel']
    pode_bombardear = self_state['bombas_ativas'] < self_state['max_bombas']

    perigo = mapa_perigo(mapa, bombas, rows, cols, tempo_fogo)
    bloqueio = {(b.x, b.y) for b in bombas if not b.explodida}
    inimigos = {(j.grid_x, j.grid_y) for j in jogadores
                if j is not player and j.ativo and j.time != player.time}
    alcancaveis = explorar(pos, mapa, bloqueio, perigo, passo, rows, cols)
    seguras = {c: v for c, v in alcancaveis.items() if c not in perigo}

    # 1. FUGIR
    if pos in perigo:
        if seguras:
            alvo = min(seguras, key=lambda c: seguras[c][0])
            return seguras[alvo][1] or "parado"
        # Sem saída garantida: vai para a casa cujo fogo começa mais tarde
        alvo = max(alcancaveis, key=lambda c: min(ini for ini, _ in perigo.get(c, [(99, 99)])))
        return alcancaveis[alvo][1] or "parado"

    # 2. POWER-UP mais próximo
    powerups = [c for c in seguras if mapa[c[1]][c[0]] in (PU_BOMBA, PU_FOGO)]
    if powerups:
        alvo = min(powerups, key=lambda c: seguras[c][0])
        return seguras[alvo][1]

    # 3. BOMBARDEAR aqui
    ja_tem_bomba = pos in bloqueio
    if (pode_bombardear and not ja_tem_bomba
            and ganho_bomba(mapa, pos, nivel, inimigos, perigo, rows, cols) > 0
            and tem_fuga(pos, mapa, bombas, bloqueio, nivel, ctx)):
        return "bomba"

    # 4. POSICIONAR: melhor casa para a próxima bomba (ganho / distância)
    candidatos = []
    for casa, (dist, _) in seguras.items():
        g = ganho_bomba(mapa, casa, nivel, inimigos, perigo, rows, cols)
        if g > 0:
            candidatos.append((g / (dist + 1), casa))
    candidatos.sort(reverse=True)
    for _, casa in candidatos[:5]:  # só testa fuga nos 5 melhores (custo)
        if tem_fuga(casa, mapa, bombas, bloqueio, nivel, ctx):
            if casa == pos:
                return "parado"  # já estou no lugar, esperando bomba ficar disponível
            return seguras[casa][1]

    # 5. CAÇAR o inimigo mais próximo
    if inimigos and len(seguras) > 1:
        def dist_inimigo(c):
            return min(abs(c[0] - ix) + abs(c[1] - iy) for ix, iy in inimigos)
        alvo = min(seguras, key=lambda c: (dist_inimigo(c), seguras[c][0]))
        if alvo != pos:
            return seguras[alvo][1]

    return "parado"


def decidir_acao(player, mapa, jogadores, bombas, tempo_restante, pontos, hud_info, self_state):
    acao = escolher_acao(player, mapa, jogadores, bombas, hud_info, self_state)

    # 6. ANTI-OCIOSIDADE: muitas chamadas seguidas paradas -> passo seguro aleatório
    chave = id(player)
    if acao == "parado":
        _ocioso[chave] = _ocioso.get(chave, 0) + 1
        if _ocioso[chave] > LIMITE_OCIOSO:
            rows, cols = hud_info['rows'], hud_info['cols']
            perigo = mapa_perigo(mapa, bombas, rows, cols, hud_info['tempo_fogo'])
            bloqueio = {(b.x, b.y) for b in bombas if not b.explodida}
            opcoes = []
            for nome, (dx, dy) in DIRECOES.items():
                nx, ny = player.grid_x + dx, player.grid_y + dy
                if (0 <= nx < cols and 0 <= ny < rows and mapa[ny][nx] in ANDAVEL
                        and (nx, ny) not in bloqueio and (nx, ny) not in perigo):
                    opcoes.append(nome)
            if opcoes:
                acao = random.choice(opcoes)
    if acao != "parado":
        _ocioso[chave] = 0
    return acao
