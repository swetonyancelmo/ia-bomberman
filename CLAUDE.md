# IA da equipe — Bomberman Arena

Script de IA de um personagem para o jogo "Bomberman Arena" (trabalho de faculdade, Python + Pygame).
O jogo é do professor e fica na pasta pai (`../main.py`). Esta pasta contém **só o código da equipe**.

O entregável é **um único arquivo**: `ia_equipe.py`. Fale com o usuário em português, e escreva comentários e nomes de variáveis em português, como no resto do código.

## Arquivos

| Arquivo | O que é |
|---|---|
| `ia_equipe.py` | A IA (o entregável). |
| `README.md` | Apresentação do projeto para pessoas (equipe e professor). Mantenha alinhado com a IA quando o comportamento mudar. |
| `simular.py` | Roda partidas sem janela e em velocidade acelerada, e imprime métricas. Ferramenta de teste; não é entregue. |
| `../ia_equipe.py` | Link simbólico para `ia-equipe/ia_equipe.py`. É por ele que o `../main.py` importa a IA. **Não apague.** |
| `../Explicacao_IA.pdf` | Explicação técnica entregue ao professor. Se o comportamento da IA mudar, avise o usuário que o PDF ficou desatualizado. |

A pasta se chama `ia-equipe` (com hífen) de propósito. Se se chamasse `ia_equipe`, o `import ia_equipe` do jogo carregaria a pasta no lugar do arquivo e quebraria.

## Requisitos do professor

> "A IA deverá ser capaz de destruir blocos, capturar power-ups e não ficar muito tempo ociosa. A equipe deverá explicar o comportamento adotado pela IA de forma técnica."

- Acompanhamento 2 (26/10/2026): protótipo do script. Já foi entregue.
- Não há outra etapa conhecida; pergunte ao usuário.

## Como rodar

O ambiente virtual com o pygame fica em `../.venv`. O `python` do sistema **não** tem pygame.

```bash
# Jogo com janela (o jogador 1, azul, é a nossa IA). Rodar da pasta pai, por causa das imagens:
cd .. && .venv/bin/python main.py

# Teste sem janela (de dentro desta pasta):
../.venv/bin/python simular.py                  # 5 partidas contra as 3 IAs aleatórias
../.venv/bin/python simular.py -n 3 --espelho   # 3 partidas com 4 cópias da nossa IA
```

Cada partida simulada leva de alguns segundos a ~1 minuto. Use `--espelho` para avaliar de verdade, porque as IAs aleatórias se explodem sozinhas em ~5 s.

Na linha `RESULTADO`:
- `blocos_restantes`: deve chegar a ~0;
- `bombas_max_p1` e `nivel_fogo_p1`: mostram se a IA pegou power-ups (os máximos são 5 e 4);
- `parado_%` e `maior_pausa_s`: medem ociosidade;
- `[ALERTA]`: indica que a IA modificou o estado do jogo (ver abaixo). Nunca deve aparecer.

Valores de referência no modo espelho: 0 a 2 blocos restantes, power-ups no máximo, parado entre 15% e 55% e maior pausa de ~4,3 s (esperando uma bomba explodir em local seguro).

## Contrato com o jogo (`../main.py`)

O jogo chama, a cada frame em que o personagem não está andando:

```python
decidir_acao(player, mapa, jogadores, bombas, tempo_restante, pontos, hud_info, self_state) -> str
```

O retorno deve ser `"cima"`, `"baixo"`, `"esquerda"`, `"direita"`, `"bomba"` ou `"parado"`.

- `mapa[y][x]`: 11 linhas × 13 colunas. Valores: `0` vazio, `1` bloco destrutível, `2` parede, `3` power-up de bomba (+1 bomba simultânea, máx. 5), `4` power-up de fogo (+1 nível, máx. 4).
- `jogadores`: objetos `Player` com `grid_x`, `grid_y`, `ativo` e `time`. A lista inclui o próprio `player`. Times: jogadores 1 e 4 são o time 0; jogadores 2 e 3 são o time 1.
- `bombas`: objetos `Bomba` com `x`, `y`, `nivel`, `explodida`, `tempo_explosao` (segundos restantes), `tempo_fogo`, `fogo` (casas em chamas) e `dono`.
- `self_state`: `grid_x`, `grid_y`, `max_bombas`, `bombas_ativas`, `bomba_nivel`.
- `hud_info`: `rows`, `cols`, `tempo_movimento` (0,1 s por casa), `tempo_explosao` (4 s), `tempo_fogo` (0,5 s), `max_bombas`.

Regras do motor que a IA reproduz:
- **Alcance da explosão:** `1 + (nivel - 1) * 2` casas em cada direção. O fogo para na parede (sem atingi-la) e no primeiro bloco (atingindo-o).
- **Reação em cadeia:** uma bomba atingida pelo fogo de outra explode na hora.
- **Bombas bloqueiam a passagem.** Só dá para sair de cima da própria bomba.
- **Morte:** estar numa casa de `fogo` enquanto `tempo_fogo > 0` mata.
- **Destruição:** blocos e power-ups atingidos somem só quando o fogo acaba. Um bloco pode virar power-up (12% bomba, 10% fogo).
- **Pontos:** bloco +100, power-up coletado +200, power-up destruído −50, matar +1000, suicídio −1000, vencer +10000.

**Anti-trapaça:** o jogo tira um checksum de `player`, `mapa`, `jogadores`, `bombas` e `pontos` antes e depois da chamada. Se algo mudou, a IA é punida. **Nunca modifique nada recebido**, nem listas internas como `player.bombas`. Copie antes de alterar. Estado próprio entre chamadas deve ficar em variáveis do módulo, indexadas por `id(player)`, porque o mesmo módulo pode controlar vários jogadores.

## Arquitetura de `ia_equipe.py`

Agente reativo por prioridades. `escolher_acao` executa a primeira regra que se aplica:

1. **Fugir:** a casa atual está no mapa de perigo. Vai para a casa segura mais próxima.
2. **Power-up:** vai ao power-up seguro mais próximo.
3. **Bombardear:** solta a bomba se `ganho_bomba > 0` e `tem_fuga`.
4. **Posicionar:** vai à casa com maior `ganho / (distância + 1)` (testa a fuga só nos 5 melhores candidatos, por custo).
5. **Caçar:** vai em direção ao inimigo mais próximo.

Por fim, `decidir_acao` aplica a **anti-ociosidade**: depois de `LIMITE_OCIOSO` (30) chamadas seguidas em `"parado"`, dá um passo aleatório para um vizinho fora de perigo.

Funções principais:
- `mapa_perigo`: retorna `{casa: [(inicio, fim), ...]}`, as janelas de tempo com fogo, incluindo reação em cadeia. O parâmetro `extra` simula uma bomba hipotética.
- `explorar`: BFS que só entra numa casa se ela estiver livre no intervalo `[d*passo, (d+1)*passo]`. Retorna `{casa: (distancia, primeira_acao)}`.
- `ganho_bomba`: bloco +100 (só se nenhuma outra bomba já o atinge), inimigo +300, power-up −150.
- `tem_fuga`: roda `mapa_perigo` com a bomba hipotética e depois `explorar`. Retorna se existe casa segura alcançável.

Parâmetros ajustáveis estão no topo do arquivo: `MARGEM`, `LIMITE_OCIOSO` e os pesos `GANHO_*`/`PERDA_*`.

Limitações conhecidas: a IA espera ~4 s quando só tem 1 bomba, pode ficar presa atrás de bombas alheias no fim da partida e não prevê o movimento dos inimigos.

## Regras de trabalho

- **Não edite `../main.py`**, que é a base do professor. A única exceção é trocar os nomes dos módulos nas linhas `p1 = ...` a `p5 = ...` (por volta da linha 441).
- Use só a biblioteca padrão do Python em `ia_equipe.py`. Não importe `pygame` nem acesse arquivos.
- Mantenha cada decisão rápida (bem abaixo de 1 frame, ~16 ms), porque ela é chamada quase todo frame.
- Depois de qualquer mudança na IA, rode `simular.py --espelho` e compare com os valores de referência acima.
- O usuário precisa conseguir explicar o código ao professor. Prefira mudanças simples e comentadas a otimizações obscuras, e avise quando o `Explicacao_IA.pdf` precisar ser atualizado.
