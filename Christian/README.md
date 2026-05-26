# MyAgent — Connect 4 · Híbrido Minimax

Agente para Connect 4 que combina detección inmediata, fork detection, tabla de aperturas y Minimax Alpha-Beta con Transposition Table persistente.

---

## Instalación

Clona el repositorio y asegúrate de tener las dependencias necesarias:

```bash
pip install numpy
```

El archivo del agente se encuentra en:

```
groups/V_final/policy.py
```

---

## Uso rápido

```python
from groups.V_final.policy import MyAgent

agent = MyAgent()
agent.mount()          # reinicia el estado interno al inicio de cada partida

action = agent.act(board)   # board: np.ndarray de shape (6, 7)
```

`act()` recibe el estado actual del tablero y devuelve un entero `0–6` indicando la columna donde colocar la ficha.

---

## Convención del tablero

| Valor | Significado        |
|-------|--------------------|
| `0`   | Celda vacía        |
| `-1`  | Ficha del jugador 1 (Rojo)   |
| `+1`  | Ficha del jugador 2 (Amarillo) |

El agente detecta automáticamente su color en cada llamada a `act()` comparando el conteo de fichas en el tablero, por lo que **no es necesario indicarle su color manualmente**.

---

## Ciclo de vida por partida

```python
agent = MyAgent()

# Al inicio de cada partida nueva — limpia la TT y el historial de jugadas
agent.mount()

# En cada turno
while not game_over:
    col = agent.act(board)
    board = apply_move(board, col)
```

> **Importante:** llama a `mount()` antes de cada partida nueva. Sin esto, la Transposition Table y el MoveTracker acumularán estado de partidas anteriores.

---

## Parámetros configurables

Los siguientes valores se pueden ajustar directamente en `policy.py`:

| Parámetro | Ubicación | Descripción |
|-----------|-----------|-------------|
| `DEPTH_SCHEDULE` | clase `MyAgent` | Profundidad Minimax según piezas en el tablero. Por defecto: depth 6 si hay < 28 piezas, depth 7 si hay ≥ 28. |
| `OPENING_BOOK` | módulo | Diccionario de aperturas. Se pueden agregar entradas como `(seq): col`. |

Ejemplo para aumentar profundidad en endgame:

```python
DEPTH_SCHEDULE = [
    (28, 6),   # < 28 piezas → depth 6
    (42, 8),   # ≥ 28 piezas → depth 8 (más lento, más fuerte)
]
```

---

## Árbol de decisión

En cada turno `act()` evalúa en este orden:

```
1. ¿Puedo ganar ahora?            → jugar ahí
2. ¿El rival gana en su turno?    → bloquear
3. ¿Puedo crear doble amenaza?    → fork propio
4. ¿El rival puede crear fork?    → bloquear fork
5. ¿Hay jugada en la tabla de aperturas? (≤ 8 piezas)
6. Minimax Alpha-Beta + TT
```

---

## Estructura de archivos

```
groups/
└── V_final/
    └── policy.py          # Agente principal (MyAgent)
```

---

## Notas

- La Transposition Table se mantiene **entre turnos de la misma partida**, lo que equivale en la práctica a 1–2 niveles extra de profundidad sin costo adicional de tiempo.
- La tabla de aperturas admite **simetría horizontal** automáticamente: no es necesario codificar la versión espejada de cada entrada.
- En self-play (V2 vs. V2) el win rate esperado es ~50% por diseño — cualquier desviación grande indica un bug de perspectiva.
