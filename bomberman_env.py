"""Motor headless do Bomberman Arena (sem pygame), para treino de IA.

Reproduz as regras do main.py em tempo discreto: 1 passo = 0.1s = 1 decisao
de cada jogador (o tempo que um movimento leva no jogo original).
"""
import random

ACOES = ["cima", "baixo", "esquerda", "direita", "bomba", "parado"]
DIRECOES = {"cima": (0, -1), "baixo": (0, 1), "esquerda": (-1, 0), "direita": (1, 0)}

# Tiles
CHAO, BLOCO, PAREDE, PU_BOMBA, PU_FOGO = 0, 1, 2, 3, 4

ROWS, COLS = 11, 13
DT = 0.1
PASSOS_EXPLOSAO = 40      # TEMPO_EXPLOSAO (4s) / DT
PASSOS_FOGO = 5           # TEMPO_FOGO (0.5s) / DT
PASSOS_PARTIDA = 1800     # TEMPO_PARTIDA (180s) / DT
MAX_BOMBAS = 5
NIVEL_MAX = 4

PONTOS_BLOCO = 100
PONTOS_POWERUP_COLETADO = 200
PONTOS_POWERUP_DESTRUIDO = -50
PONTOS_MATAR_JOGADOR = 1000
PONTOS_VITORIA = 10000

PROB_BOMBA = 0.12
PROB_FOGO = 0.10

MAPA_INICIAL = [
    [0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0],
    [0, 2, 1, 2, 0, 2, 0, 2, 1, 2, 1, 2, 0],
    [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1],
    [0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1],
    [1, 2, 0, 2, 1, 2, 0, 2, 1, 2, 1, 2, 1],
    [1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1],
    [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1],
    [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 2, 1, 2, 0, 2, 1, 2, 0, 2, 0, 2, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
]

# (x, y, time) de cada jogador, igual ao main.py
SPAWNS = [(0, 0, 0), (12, 0, 1), (0, 10, 1), (12, 10, 0)]


class Jogador:
    __slots__ = ("x", "y", "time", "vivo", "max_bombas", "nivel", "bombas")

    def __init__(self, x, y, time):
        self.x, self.y, self.time = x, y, time
        self.vivo = True
        self.max_bombas = 1
        self.nivel = 1
        self.bombas = 0  # quantas bombas suas estao ativas


class Bomba:
    __slots__ = ("x", "y", "timer", "nivel", "dono", "explodida", "fogo", "timer_fogo")

    def __init__(self, x, y, nivel, dono):
        self.x, self.y, self.nivel, self.dono = x, y, nivel, dono
        self.timer = PASSOS_EXPLOSAO
        self.explodida = False
        self.fogo = []
        self.timer_fogo = 0


class BombermanEnv:
    def __init__(self, seed=None, max_passos=PASSOS_PARTIDA, solo=False):
        self.rng = random.Random(seed)
        self.max_passos = max_passos  # no treino usamos partidas mais curtas
        # solo: so o jogador 0 existe e morrer na propria bomba e barato.
        # Usado na primeira fase do treino para a rede aprender "bomba, fugir, bloco".
        self.solo = solo
        self.custo_suicidio = 0 if solo else PONTOS_MATAR_JOGADOR
        self.fator_morte = 0.3 if solo else 1.0
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            self.rng = random.Random(seed)
        self.mapa = [linha[:] for linha in MAPA_INICIAL]
        self.jogadores = [Jogador(*s) for s in SPAWNS]
        if self.solo:
            # o unico jogador vivo e sorteado: a rede precisa aprender em todos os cantos
            self.solo_idx = self.rng.randrange(len(self.jogadores))
            for k, p in enumerate(self.jogadores):
                p.vivo = k == self.solo_idx
        self.bombas = []
        self.blocos_total = [0] * len(self.jogadores)
        self.suicidios = 0
        self.pontos = [0] * len(self.jogadores)
        self.passo = 0
        self.blocos_passo = [0] * len(self.jogadores)
        # contadores por jogador (usados so pelas recompensas por papel)
        self.pu_total = [0] * len(self.jogadores)      # power-ups coletados
        self.pu_passo = [0] * len(self.jogadores)
        self.kills_total = [0] * len(self.jogadores)   # jogadores mortos pela bomba dele (nao conta suicidio)
        self.kills_passo = [0] * len(self.jogadores)
        self.fim = False
        self.vencedor = None  # indice do vencedor, ou None (empate/ninguem vivo)
        return self

    # ---------------------------------------------------------------- regras
    def _bomba_em(self, x, y):
        return any(b.x == x and b.y == y and not b.explodida for b in self.bombas)

    def zona_explosao(self, b):
        """Tiles que a bomba b atingiria se explodisse agora."""
        zona = [(b.x, b.y)]
        for dx, dy in DIRECOES.values():
            for i in range(1 + (b.nivel - 1) * 2):
                nx, ny = b.x + dx * (i + 1), b.y + dy * (i + 1)
                if not (0 <= nx < COLS and 0 <= ny < ROWS):
                    break
                if self.mapa[ny][nx] == PAREDE:
                    break
                zona.append((nx, ny))
                if self.mapa[ny][nx] == BLOCO:
                    break
        return zona

    def _explodir(self, b):
        b.explodida = True
        b.timer_fogo = PASSOS_FOGO
        b.fogo = self.zona_explosao(b)
        # reacao em cadeia
        for fx, fy in b.fogo:
            for outra in self.bombas:
                if not outra.explodida and (outra.x, outra.y) == (fx, fy):
                    self._explodir(outra)

    def _powerup_aleatorio(self):
        r = self.rng.random()
        if r < PROB_BOMBA:
            return PU_BOMBA
        if r < PROB_BOMBA + PROB_FOGO:
            return PU_FOGO
        return CHAO

    def _agir(self, i, acao):
        p = self.jogadores[i]
        if acao == "bomba":
            if p.bombas < p.max_bombas and not self._bomba_em(p.x, p.y):
                b = Bomba(p.x, p.y, p.nivel, i)
                self.bombas.append(b)
                p.bombas += 1
            return
        if acao not in DIRECOES:
            return
        dx, dy = DIRECOES[acao]
        nx, ny = p.x + dx, p.y + dy
        if not (0 <= nx < COLS and 0 <= ny < ROWS):
            return
        if self.mapa[ny][nx] not in (CHAO, PU_BOMBA, PU_FOGO) or self._bomba_em(nx, ny):
            return
        p.x, p.y = nx, ny
        tile = self.mapa[ny][nx]
        if tile == PU_BOMBA:
            p.max_bombas = min(p.max_bombas + 1, MAX_BOMBAS)
        elif tile == PU_FOGO:
            p.nivel = min(p.nivel + 1, NIVEL_MAX)
        if tile in (PU_BOMBA, PU_FOGO):
            self.mapa[ny][nx] = CHAO
            self.pontos[i] += PONTOS_POWERUP_COLETADO
            self.pu_total[i] += 1
            self.pu_passo[i] += 1

    # ------------------------------------------------------------------ passo
    def step(self, acoes):
        """acoes: lista com uma acao (str de ACOES) por jogador.

        Retorna (pontos_ganhos_neste_passo, fim)."""
        if self.fim:
            return [0] * len(self.jogadores), True
        antes = self.pontos[:]
        self.passo += 1
        self.blocos_passo = [0] * len(self.jogadores)  # blocos destruidos por jogador neste passo
        self.pu_passo = [0] * len(self.jogadores)
        self.kills_passo = [0] * len(self.jogadores)

        # 1) jogadores agem
        for i, p in enumerate(self.jogadores):
            if p.vivo:
                self._agir(i, acoes[i])

        # 2) bombas avancam; fogo que acabou destroi blocos/power-ups
        for b in self.bombas[:]:
            if not b.explodida:
                b.timer -= 1
                if b.timer <= 0:
                    self._explodir(b)
            else:
                b.timer_fogo -= 1
            if b.explodida and b.timer_fogo <= 0:
                for fx, fy in b.fogo:
                    t = self.mapa[fy][fx]
                    if t == BLOCO:
                        self.mapa[fy][fx] = self._powerup_aleatorio()
                        self.pontos[b.dono] += PONTOS_BLOCO
                        self.blocos_passo[b.dono] += 1
                        self.blocos_total[b.dono] += 1
                    elif t in (PU_BOMBA, PU_FOGO):
                        self.mapa[fy][fx] = CHAO
                        self.pontos[b.dono] += PONTOS_POWERUP_DESTRUIDO
                self.bombas.remove(b)
                self.jogadores[b.dono].bombas -= 1

        # 3) fogo ativo mata
        for b in self.bombas:
            if b.explodida and b.timer_fogo > 0:
                for i, p in enumerate(self.jogadores):
                    if p.vivo and (p.x, p.y) in b.fogo:
                        p.vivo = False
                        if b.dono == i:
                            self.pontos[i] = max(0, self.pontos[i] - self.custo_suicidio)
                            self.suicidios += 1
                        else:
                            self.pontos[b.dono] += PONTOS_MATAR_JOGADOR
                            self.kills_total[b.dono] += 1
                            self.kills_passo[b.dono] += 1

        # 4) fim de jogo
        vivos = [i for i, p in enumerate(self.jogadores) if p.vivo]
        if self.solo:
            if not vivos or self.passo >= self.max_passos:
                self.fim = True
        elif len(vivos) == 1:
            self.vencedor = vivos[0]
            self.pontos[self.vencedor] += PONTOS_VITORIA
            self.fim = True
        elif len(vivos) == 0:
            self.fim = True
        elif self.passo >= self.max_passos:
            self.vencedor = self.pontos.index(max(self.pontos))
            self.fim = True

        return [d - a for d, a in zip(self.pontos, antes)], self.fim
