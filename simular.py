"""
Roda partidas do jogo sem janela e em velocidade acelerada para testar a IA.

Uso (de dentro desta pasta):
    ../.venv/bin/python simular.py                 # 5 partidas contra as IAs aleatórias
    ../.venv/bin/python simular.py -n 3 --espelho  # 3 partidas com 4 cópias da nossa IA

Não modifica o ../main.py: lê o arquivo, aplica ajustes em memória e executa a
cópia ajustada em um subprocesso (um por partida, para zerar o estado).
"""
import argparse
import os
import subprocess
import sys
import tempfile

PASTA = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(PASTA)  # pasta do jogo (main.py, imagens, .venv)

# (trecho original do main.py, substituição). Se o professor mudar o main.py e
# algum trecho sumir, o simulador avisa em vez de rodar algo errado.
AJUSTES = [
    ('pygame.init()', 'pygame.init()\nSTATS = {}'),
    # tempo fixo de 1/60 s por frame, sem esperar o relógio real
    ('delta = clock.tick(60) / 1000', 'delta = 1 / 60'),
    ('pygame.display.flip()', 'pass'),
    # conta chamadas e pausas da IA de cada jogador
    ('            return acao\n',
     '            s = STATS.setdefault(id(self), [0, 0, 0])  # [paradas, sequência, maior sequência]\n'
     '            s[0] += acao == "parado"\n'
     '            s[1] = s[1] + 1 if acao == "parado" else 0\n'
     '            s[2] = max(s[2], s[1])\n'
     '            return acao\n'),
    ('''    if vencedor_final is not None:
        # Apenas''', '''    if vencedor_final is not None:
        dur = max(1e-9, TEMPO_PARTIDA - tempo_restante)
        def _st(p):
            s = STATS.get(id(p), [0, 0, 0])
            return round(100 * s[0] / 60 / dur), round(s[2] / 60, 1)
        print("RESULTADO", "pontos=" + str(pontos[:len(players)]),
              "vivos=" + str([p.ativo for p in players]),
              "blocos_restantes=" + str(sum(r.count(1) for r in mapa)),
              "duracao_s=" + str(round(dur, 1)),
              "parado_%=" + str([_st(p)[0] for p in players]),
              "maior_pausa_s=" + str([_st(p)[1] for p in players]),
              "bombas_max_p1=" + str(players[0].max_bombas),
              "nivel_fogo_p1=" + str(players[0].bomba_nivel))
        sys.exit()
        # Apenas'''),
]


def montar_fonte(espelho):
    with open(os.path.join(BASE, "main.py"), encoding="utf-8") as f:
        fonte = f.read()
    jogadores = {"p1": "ia_equipe"}
    if espelho:
        jogadores.update(p2="ia_equipe", p3="ia_equipe", p4="ia_equipe")
    for var, modulo in jogadores.items():
        AJUSTES.append((f'{var} = "ia_jogador{var[1]}"', f'{var} = "{modulo}"'))
    for original, novo in AJUSTES:
        if original not in fonte:
            # p1 já pode estar apontando para ia_equipe no main.py
            if original.startswith('p1 = '):
                continue
            sys.exit(f"Trecho não encontrado no main.py (ele mudou?): {original[:60]!r}")
        fonte = fonte.replace(original, novo)
    return fonte


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=5, help="número de partidas")
    ap.add_argument("--espelho", action="store_true", help="4 cópias da nossa IA")
    args = ap.parse_args()

    python = os.path.join(BASE, ".venv", "bin", "python")
    if not os.path.exists(python):
        python = sys.executable
    env = dict(os.environ, SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy",
               PYTHONPATH=os.pathsep.join([PASTA, BASE]))

    with tempfile.TemporaryDirectory() as tmp:
        script = os.path.join(tmp, "sim_main.py")
        with open(script, "w", encoding="utf-8") as f:
            f.write(montar_fonte(args.espelho))
        for i in range(1, args.n + 1):
            r = subprocess.run([python, script], cwd=BASE, env=env,
                               capture_output=True, text=True, timeout=900)
            linhas = [l for l in (r.stdout + r.stderr).splitlines()
                      if l.startswith(("RESULTADO", "[ALERTA]", "Traceback"))
                      or "Error" in l or "morreu" in l]
            print(f"Partida {i}:")
            for l in linhas:
                print("   ", l)


if __name__ == "__main__":
    main()
