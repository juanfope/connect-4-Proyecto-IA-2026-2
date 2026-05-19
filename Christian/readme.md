# Agente Connect 4 — Tabla de Aperturas + Minimax Híbrido

## Idea principal

El agente combina dos enfoques según la fase de la partida:

|     Fase      |        Condición         |            Estrategia                  |
|---------------|--------------------------|----------------------------------------|
| **Apertura**  | ≤ 8 piezas en el tablero | Tabla de aperturas teóricamente óptima |
| **Medio/Fin** |       > 8 piezas         | Minimax con Alpha-Beta pruning         |

Esto lo diferencia del Minimax puro del grupo: durante la apertura no hace ninguna búsqueda sino que consulta movimientos probadamente óptimos derivados de la solución completa del juego (Allis, 1988).

---

## Fuentes de la tabla de aperturas

Connect 4 es un **juego completamente resuelto** desde 1988. Las aperturas codificadas provienen directamente de la literatura académica y bases de datos computacionales:

### 1. Victor Allis (1988)
**"A Knowledge-Based Approach of Connect-Four"**
Master's Thesis, Vrije Universiteit Amsterdam.
- Demostró que el **primer jugador siempre gana** empezando en la columna central (col 3, índice 0-based).
- Columnas 0 y 6 (bordes): **pierde** el primer jugador con juego perfecto.
- Columnas 2 y 4 (adyacentes al centro): **tablas** con juego perfecto.
- Método: combinación de reglas basadas en conocimiento + búsqueda exhaustiva.
- PDF de la tesis: https://tromp.github.io/c4/allis.pdf

### 2. James D. Allen (1988)
**Anuncios en rec.games.programmer (1 Oct 1988)**; luego publicado en:
*"The Complete Book of CONNECT 4: History, Strategy, Puzzles"*
- Resolvió Connect 4 independientemente, 15 días antes que Allis.
- Confirmó los mismos resultados por un enfoque más computacional.

### 3. John Tromp (1995) — Base de datos de 8-ply
**Sitio web:** https://tromp.github.io/c4.html
- Construyó la primera base de datos que da el resultado óptimo (ganada/perdida/tablas) para **todas las posiciones a 8 jugadas**.
- Esta base de datos es la fuente de las respuestas específicas de movimientos 3-5 en la tabla de aperturas del agente.
- La base de datos también fue base del benchmark de rendimiento entero **Fhourstones**.

### 4. James D. Allen — "Expert Play in Connect-Four"
**URL:** http://www.pomakis.com/c4/expert_play.html
- Tabla completa de resultados (X gana / O gana / Tablas) para todas las aperturas de 2 jugadas.
- Incluye las secuencias Joseki (jugadas perfectas para ambos lados) para las variantes principales.
- Fuente directa de las entradas de 3-5 movimientos en la tabla del agente.

### 5. Pascal Pons — "Solving Connect Four: How to Build a Perfect AI"
**URL:** http://blog.gamesolver.org/solving-connect-four/01-introduction/
- Tutorial moderno con implementación de referencia (alpha-beta + tablas de transposición).
- Confirma los resultados de apertura de Allis/Allen/Tromp con código verificable.

### Resultado clave de la teoría (para el PDF)

```
Primera jugada del Jugador 1:
  Columna 3 (centro)   → GANA P1 con juego perfecto ✓
  Columna 2 o 4        → TABLAS con juego perfecto
  Columna 1 o 5        → GANA P2 con juego perfecto ✗
  Columna 0 o 6        → GANA P2 con juego perfecto ✗
```

---

## Profundidad dinámica del Minimax

Conforme se llenan columnas, el árbol de búsqueda se reduce, permitiendo mayor profundidad:

| Piezas en el tablero | Profundidad Minimax |
|----------------------|---------------------|
|        < 10          |         5           |
|        < 20          |         6           |
|        < 30          |         7           |
|       ≥ 30 (endgame) |         9           |

---

## Uso

Colocar `policy.py` en `groups/christian/` con un `__init__.py` vacío. No requiere entrenamiento previo ni archivos adicionales.

```
groups/
  christian/
    __init__.py
    policy.py     ← este archivo
```
