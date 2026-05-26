# Agente Connect-4: Monte Carlo Tree Search (MCTS)

> **Proyecto:** IA 2026-2 — Connect-4 Tournament Framework  
> **Estrategia:** Búsqueda en Árbol de Montecarlo (MCTS)

---

## Tabla de Contenidos

1. [Requisitos](#1-requisitos)
2. [Instalación](#2-instalación)
3. [Estructura del Proyecto](#3-estructura-del-proyecto)
4. [Configuración del Agente](#4-configuración-del-agente)
5. [Guía de Uso](#5-guía-de-uso)

---

## 1. Requisitos

- Python 3.10+
- `numpy`
- `matplotlib` (para visualizaciones en el notebook)

---

## 2. Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/juanfope/connect-4-Proyecto-IA-2026-2.git

# 2. Entrar a la carpeta del agente
cd connect-4-Proyecto-IA-2026-2/Samuel

# 3. (Opcional) Crear entorno virtual
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

# 4. Instalar dependencias
pip install numpy matplotlib
```

---

## 3. Estructura del Proyecto

```
Samuel/
├── main.py                    # Punto de entrada: carga participantes y lanza el torneo
├── tournament.py              # Lógica del torneo eliminatorio (best-of-N, BYEs, seed)
├── entrega.ipynb              # Notebook de análisis, experimentos y visualizaciones
├── connect4/
│   ├── connect_state.py       # Estado del tablero (6×7), transiciones y detección de victoria
│   ├── policy.py              # Clase base abstracta Policy (mount + act)
│   ├── dtos.py                # Tipos de datos: Game, Match, Participant, Versus
│   ├── environment_state.py   # Interfaz base EnvironmentState
│   └── utils.py               # find_importable_classes: descubrimiento automático de agentes
├── groups/
│   ├── MCTS/
│   │   └── policy.py          # ← Implementación principal del agente MCTS
|   |   └── fig_mirror_match.png
|   |   └── fig_vs_greedy.png
|   |   └── fig_vs_greedy_colors.png
|   |   └── fig_vs_random.png
│   ├── Group A/
│   │   └── policy.py          # Agente de referencia
│   ├── Group B/
│   │   └── policy.py          # Agente de referencia
│   └── Group C/
│       └── policy.py          # Agente de referencia
└── versus/
    └── match_Group A_vs_Group B.json           # Resultados de partidas, guardados automáticamente
    └── match_Group C_vs_Group A.json           # Resultados de partidas, guardados automáticamente
```

> `find_importable_classes` recorre todas las subcarpetas de `groups/` buscando clases que hereden de `Policy`. El **nombre de la subcarpeta** es el nombre del participante en el torneo.

---

## 4. Configuración del Agente

Esta sección se centra en la inicialización, calibración de parámetros internos y ejecución del agente MCTS dentro del framework de torneos.

El comportamiento y la eficiencia del agente no son fijos, sino que dependen de la manipulación de dos hiperparámetros críticos en su constructor, ubicado en `groups/MCTS/policy.py`:

```python
agente = MCTS(n_simulations=800, c_exploration=1.414)
```

### `n_simulations` — Presupuesto de Cómputo

Determina cuántas partidas simuladas ejecutará el agente **por turno** para expandir el árbol.

| Valor | Uso recomendado |
|---|---|
| `300` | Mínimo garantizado; se aplica automáticamente cuando el framework limita el tiempo por turno |
| `400` | **Equilibrio óptimo** tiempo/rendimiento |
| `800` | **Mayor calidad de decisión** — resultados notablemente mejores en las evaluaciones |

> Si el framework provee un límite de tiempo via `mount(action_timeout)`, el agente ajusta `n_simulations` automáticamente con `max(300, int(action_timeout * 150))`.

### `c_exploration` — Constante de Exploración UCB1

Controla el balance exploración/explotación dentro de la fórmula UCB1.

| Valor | Comportamiento | Cuándo usar |
|---|---|---|
| `1.414` (√2) | Promueve una exploración horizontal amplia. Ideal para descubrir líneas complejas a largo plazo si se cuenta con suficiente presupuesto de simulaciones. | Con `n_simulations` alto |
| `1.0` | Fuerza una explotación más agresiva. Concentra el cómputo en las columnas centralmente ganadoras, ideal para mitigar descuidos tácticos tempranos si el presupuesto es ajustado. | Con `n_simulations` bajo |

---

## 5. Guía de Uso

### Torneo completo entre todos los grupos

```bash
python main.py
```

El torneo es eliminatorio (*single-elimination*) con partidos **best-of-7** por defecto. Los resultados de cada enfrentamiento se guardan automáticamente en `versus/match_<A>_vs_<B>.json`.

### Partida individual entre dos agentes

Desde el notebook o un script propio:

```python
from connect4.policy import Policy
from connect4.utils import find_importable_classes
from tournament import play

participants = find_importable_classes("groups", Policy)
players = list(participants.items())

# Ejemplo: primer agente vs segundo agente
winner = play(players[0], players[1], best_of=7, first_player_distribution=0.5)
print("Ganador:", winner[0])
```

### Notebook de análisis

```bash
jupyter notebook entrega.ipynb
```

Contiene la configuración completa de la arena (`play_game`, `run_series`), los experimentos de mirror-match, comparaciones contra agentes aleatorios y codiciosos, y visualizaciones de rendimiento por color.


