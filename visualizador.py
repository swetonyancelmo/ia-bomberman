"""Janela para assistir as partidas do treino ao vivo (opcional: treino_ppo.py --ver).

Teclas: 1/2/3 = velocidade 1x/2x/4x | 4 = turbo (sem limite) | ESPACO = pausa | ESC = fecha a janela
(fechar a janela nao interrompe o treino, so deixa de desenhar).
"""
import time

import pygame

from bomberman_env import ROWS, COLS, MAX_BOMBAS

TILE = 48
HUD = 40
PAINEL = 310
LARGURA = COLS * TILE + PAINEL
ALTURA = ROWS * TILE + HUD
PASSOS_POR_SEG = {pygame.K_1: 10, pygame.K_2: 20, pygame.K_3: 40, pygame.K_4: None}  # 10/s = tempo real

COR_FUNDO = (20, 20, 20)
COR_PAINEL = (30, 30, 34)
COR_TEXTO = (230, 230, 230)
COR_FRACA = (140, 140, 150)
COR_CHAO = (160, 82, 45)
COR_GRAMA = (80, 150, 80)
COR_PAREDE = (60, 60, 60)
COR_JOGADORES = [(0, 200, 255), (255, 100, 100), (200, 255, 100), (255, 255, 0)]
NOME_CONTROLE = {"rede": "rede (aprende)", "antiga": "rede antiga", "aleatorio": "aleatorio",
                 "coletor": "COLETOR (power-ups)", "assassino": "ASSASSINO (matar)",
                 "fugitivo": "FUGITIVO (sobreviver)", "base": "base (congelada)",
                 "aprendiz": "APRENDIZ (online)"}


class Visualizador:
    def __init__(self, passos_por_seg=20):
        pygame.init()
        self.tela = pygame.display.set_mode((LARGURA, ALTURA))
        pygame.display.set_caption("Bomberman Arena - treino ao vivo")
        self.fonte = pygame.font.SysFont("Arial", 15)
        self.fonte_b = pygame.font.SysFont("Arial", 17, bold=True)
        self.relogio = pygame.time.Clock()
        self.passos_por_seg = passos_por_seg
        self.ativo = True
        self.pausado = False
        self.contexto = {}           # preenchido pelo treino: fase, iteracao
        self.historico = {"blocos": [], "retorno": [], "duracao": []}
        self._carregar_imagens()

    def _carregar_imagens(self):
        def img(nome):
            return pygame.transform.scale(pygame.image.load(nome).convert(), (TILE, TILE))
        self.bloco = img("bloco.jpg")
        self.pu_bomba = img("pu_bomb.jpg")
        self.pu_fogo = img("pu_fire.jpg")
        bomba = pygame.image.load("sprite_bomb_no_bg.png").convert_alpha()
        self.bomba = [pygame.transform.scale(bomba.subsurface(pygame.Rect(i * 48, 0, 48, 48)), (TILE, TILE))
                      for i in range(3)]
        explosao = pygame.image.load("sprite_explosao.png").convert_alpha()
        self.fogo = pygame.transform.scale(explosao.subsurface(pygame.Rect(TILE, 0, TILE, TILE)), (TILE, TILE))
        # sprite do jogador de frente (frame 6 da folha): player1.png = time 0, player2.png = time 1
        self.sprites = []
        for arq in ("player1.png", "player2.png"):
            folha = pygame.image.load(arq).convert_alpha()
            self.sprites.append(pygame.transform.scale(folha.subsurface(pygame.Rect(6 * 48, 0, 48, 48)), (TILE, TILE)))

    # ------------------------------------------------------------- entrada
    def _eventos(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                self.ativo = False
                pygame.display.quit()
                return
            if ev.type == pygame.KEYDOWN:
                if ev.key in PASSOS_POR_SEG:
                    self.passos_por_seg = PASSOS_POR_SEG[ev.key]
                elif ev.key == pygame.K_SPACE:
                    self.pausado = not self.pausado

    # ------------------------------------------------------------- desenho
    def desenhar(self, env, controle, partida):
        """Desenha um passo da partida `partida` (indice do env) e respeita a velocidade escolhida."""
        if not self.ativo:
            return
        self._eventos()
        while self.pausado and self.ativo:
            time.sleep(0.05)
            self._eventos()
        if not self.ativo:
            return

        self.tela.fill(COR_FUNDO)
        self._hud(env, partida)
        self._mapa(env)
        self._painel(env, controle)
        pygame.display.flip()
        if self.passos_por_seg:
            self.relogio.tick(self.passos_por_seg)

    def _texto(self, txt, pos, cor=COR_TEXTO, fonte=None):
        self.tela.blit((fonte or self.fonte).render(txt, True, cor), pos)

    def _hud(self, env, partida):
        c = self.contexto
        vel = "turbo" if not self.passos_por_seg else f"{self.passos_por_seg / 10:g}x"
        pausa = "  [PAUSADO]" if self.pausado else ""
        self._texto(f"Fase {c.get('fase', '?')}   iteracao {c.get('it', '?')}   partida {partida + 1}   "
                    f"passo {env.passo}/{env.max_passos}   velocidade {vel}{pausa}", (8, 10), fonte=self.fonte_b)

    def _mapa(self, env):
        for y in range(ROWS):
            for x in range(COLS):
                px, py = x * TILE, y * TILE + HUD
                v = env.mapa[y][x]
                pygame.draw.rect(self.tela, COR_CHAO if v != 2 else COR_PAREDE, (px, py, TILE, TILE))
                if v == 1:
                    self.tela.blit(self.bloco, (px, py))
                elif v == 3:
                    self.tela.blit(self.pu_bomba, (px, py))
                elif v == 4:
                    self.tela.blit(self.pu_fogo, (px, py))
                pygame.draw.rect(self.tela, COR_FUNDO, (px, py, TILE, TILE), 1)
        for b in env.bombas:
            if b.explodida:
                for fx, fy in b.fogo:
                    self.tela.blit(self.fogo, (fx * TILE, fy * TILE + HUD))
            else:
                self.tela.blit(self.bomba[(env.passo // 2) % 3], (b.x * TILE, b.y * TILE + HUD))
        for i, p in enumerate(env.jogadores):
            if p.vivo:
                self.tela.blit(self.sprites[p.time], (p.x * TILE, p.y * TILE + HUD))
                self._texto(str(i + 1), (p.x * TILE + 3, p.y * TILE + HUD + 1), COR_JOGADORES[i], self.fonte_b)

    def _painel(self, env, controle):
        x0 = COLS * TILE
        pygame.draw.rect(self.tela, COR_PAINEL, (x0, 0, PAINEL, ALTURA))
        x = x0 + 12
        y = 10
        self._texto("Jogadores", (x, y), fonte=self.fonte_b); y += 26
        for i, p in enumerate(env.jogadores):
            if env.solo and i != env.solo_idx:
                continue
            estado = "vivo" if p.vivo else "morto"
            cor = COR_JOGADORES[i] if p.vivo else COR_FRACA
            self._texto(f"J{i + 1}  {NOME_CONTROLE[controle[i]]}", (x, y), cor)
            self._texto(f"{estado}  {env.pontos[i]} pts  bombas {p.max_bombas}  alc. {p.nivel}", (x + 14, y + 17), COR_FRACA)
            y += 42
        y += 6
        self._texto("Aprendizado (por iteracao)", (x, y), fonte=self.fonte_b); y += 26
        for chave, titulo in (("blocos", "blocos destruidos / partida"), ("retorno", "retorno medio"),
                              ("duracao", "duracao media (passos)")):
            serie = self.historico[chave]
            ultimo = f"{serie[-1]:.1f}" if serie else "-"
            self._texto(f"{titulo}: {ultimo}", (x, y), COR_FRACA); y += 18
            self._grafico(serie, pygame.Rect(x, y, PAINEL - 24, 52)); y += 62
        self._texto("1/2/3 velocidade   4 turbo   espaco pausa", (x, ALTURA - 40), COR_FRACA)
        self._texto("esc fecha a janela (treino continua)", (x, ALTURA - 22), COR_FRACA)

    def _grafico(self, serie, rect):
        pygame.draw.rect(self.tela, (22, 22, 26), rect)
        pygame.draw.rect(self.tela, (60, 60, 66), rect, 1)
        if len(serie) < 2:
            return
        serie = serie[-120:]
        lo, hi = min(serie), max(serie)
        if hi - lo < 1e-9:
            hi = lo + 1
        pts = [(rect.x + 3 + k * (rect.w - 6) / (len(serie) - 1),
                rect.bottom - 4 - (v - lo) / (hi - lo) * (rect.h - 8)) for k, v in enumerate(serie)]
        pygame.draw.lines(self.tela, (120, 220, 140), False, pts, 2)

    # ------------------------------------------------------------- treino
    def registrar_iteracao(self, blocos, retorno, duracao):
        self.historico["blocos"].append(float(blocos))
        self.historico["retorno"].append(float(retorno))
        self.historico["duracao"].append(float(duracao))

    def salvar_captura(self, caminho):
        pygame.image.save(self.tela, caminho)
