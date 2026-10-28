# IA da Equipe — Bomberman Arena

Script de inteligência artificial que controla um personagem no **Bomberman Arena**, jogo em Python + Pygame fornecido pelo professor. A IA decide sozinha, a cada momento, se o personagem anda, solta uma bomba ou espera.

## O que a IA faz

| Capacidade | Como |
|---|---|
| Destruir blocos | Procura a casa onde uma bomba quebra mais blocos, vai até lá, solta a bomba e foge. |
| Capturar power-ups | Vai buscar qualquer power-up que dê para alcançar sem perigo. |
| Não ficar ociosa | Sempre tem um próximo objetivo; se ficar parada mais de ~0,5 s sem motivo, dá um passo seguro. |

## Como funciona

A IA é um **agente reativo baseado em prioridades**. Toda vez que o jogo pede uma ação, ela segue estas regras em ordem e executa a primeira que se aplica:

1. **Fugir**: se a casa atual vai pegar fogo, vai para a casa segura mais próxima.
2. **Pegar power-up**: se existe um power-up alcançável com segurança, vai buscá-lo.
3. **Soltar bomba**: se a bomba ali quebra blocos ou acerta um inimigo, **e** existe rota de fuga.
4. **Se posicionar**: anda até o melhor lugar para a próxima bomba.
5. **Caçar**: quando acabam os blocos, se aproxima do inimigo mais próximo (nunca ataca o próprio time).
6. **Anti-ociosidade**: se ficar parada tempo demais sem motivo, dá um passo seguro.

Para tomar essas decisões, ela usa quatro técnicas:

- **Mapa de perigo com tempo**: calcula *quando* cada casa do mapa vai pegar fogo, levando em conta o alcance das bombas e a reação em cadeia entre elas.
- **Busca em largura (BFS)**: encontra o caminho mais curto até o objetivo. Só passa por uma casa se ela estiver sem fogo no momento em que o personagem chegar.
- **Função de ganho**: dá uma nota a cada lugar possível para soltar uma bomba (+100 por bloco, +300 por inimigo, −150 por power-up que seria destruído) e divide pela distância. Assim, prefere alvos bons e próximos.
- **Simulação antes de bombardear**: antes de soltar a bomba, a IA simula a explosão e confere se ainda consegue fugir. Por isso ela não se mata com a própria bomba.

A IA só lê as informações do jogo e nunca as altera, respeitando a verificação anti-trapaça do `main.py`.

## Estrutura

```
Base Projeto/              ← jogo do professor
├── main.py                ← motor do jogo
├── ia_equipe.py           ← link para ia-equipe/ia_equipe.py (é por ele que o jogo carrega a IA)
├── Explicacao_IA.pdf      ← explicação técnica entregue ao professor
├── .venv/                 ← ambiente Python com o pygame
└── ia-equipe/             ← esta pasta (código da equipe, versionado com git)
    ├── ia_equipe.py       ← a IA (arquivo entregue)
    ├── simular.py         ← simulador de partidas para testes
    ├── README.md
    └── CLAUDE.md          ← instruções para agentes de IA que trabalharem no projeto
```

Funções principais de `ia_equipe.py`:

| Função | O que faz |
|---|---|
| `decidir_acao` | Ponto de entrada chamado pelo jogo; aplica a regra anti-ociosidade. |
| `escolher_acao` | Aplica as regras de prioridade. |
| `mapa_perigo` | Calcula quando cada casa estará em chamas. |
| `explorar` | Busca em largura (BFS) que respeita o tempo do fogo. |
| `ganho_bomba` | Dá a nota de soltar uma bomba numa casa. |
| `tem_fuga` | Simula a bomba e verifica se há rota de fuga. |

## Como rodar

O pygame está instalado só no ambiente virtual `.venv`, na pasta do jogo. Por isso, use sempre o Python de lá.

**Assistir uma partida.** Rode a partir da pasta `Base Projeto`. A nossa IA é o jogador 1 (azul, no canto superior esquerdo).

```bash
.venv/bin/python main.py
```

**Testar sem janela.** Rode a partir desta pasta. As partidas são simuladas em velocidade acelerada.

```bash
../.venv/bin/python simular.py                  # 5 partidas contra as IAs aleatórias
../.venv/bin/python simular.py -n 3 --espelho   # 3 partidas com 4 cópias da nossa IA
```

Para trocar qual jogador usa a nossa IA, mude os nomes nas linhas `p1 = ...` a `p4 = ...` do `main.py`.

## Resultados dos testes

- **Contra as 3 IAs aleatórias**: venceu todas as partidas sem morrer. As aleatórias costumam se explodir sozinhas em poucos segundos.
- **4 cópias da nossa IA jogando entre si**:
  - quase todos os blocos do mapa foram destruídos (sobraram de 0 a 2 em cada partida);
  - as IAs pegaram power-ups até o máximo (5 bombas e fogo nível 4);
  - não houve nenhum alerta de trapaça.

## Limitações e próximos passos

- Com apenas 1 bomba, a IA espera ~4 s até a explosão para soltar a próxima.
- No fim da partida, pode ficar presa atrás das bombas de outros jogadores.
- A caça ao inimigo é simples: ela se aproxima, mas não prevê para onde o adversário vai.
