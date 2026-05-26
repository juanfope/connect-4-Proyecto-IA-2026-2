# Agente D - Conecta 4 (Minimaxer)

Este agente utiliza una estrategia de búsqueda basada en el algoritmo **Minimax con Poda Alpha-Beta** para tomar decisiones óptimas en el juego Conecta 4. Está diseñado para equilibrar la profundidad de análisis con la velocidad de ejecución.

## Características Técnicas

### 1. Motor de Búsqueda
*   **Algoritmo:** Minimax con Poda Alpha-Beta para reducir el número de nodos explorados.
*   **Profundidad:** Configurada por defecto en un nivel de **5**, lo que permite anticipar jugadas críticas manteniendo un tiempo de respuesta rápido.
*   **Ordenamiento de Movimientos:** El agente evalúa primero las columnas centrales. Esto maximiza la eficiencia de la poda Alpha-Beta al encontrar valores altos más pronto en la búsqueda.

### 2. Heurística de Evaluación
El agente evalúa el estado del tablero mediante una función heurística que analiza "ventanas" de 4 celdas (horizontales, verticales y diagonales). Los pesos asignados son:

| Condición | Puntuación |
| :--- | :--- |
| **4 en línea (Victoria)** | +100.0 |
| **4 en línea (Oponente)** | -100.0 |
| **3 en línea + 1 vacío** | +5.0 |
| **3 oponente + 1 vacío** | -4.0 |
| **2 en línea + 2 vacíos** | +2.0 |
| **Pieza en columna central**| +3.0 por pieza |

### 3. Optimizaciones de Actuación
Antes de iniciar la búsqueda Minimax, el método `act` realiza comprobaciones de prioridad inmediata:
1.  **Victoria Inmediata:** Si existe un movimiento que gana la partida en ese turno, lo toma sin buscar más.
2.  **Bloqueo Crítico:** Si el oponente tiene una jugada ganadora inmediata, el agente la bloquea prioritariamente.

## Estructura del Código

*   `Minimaxer`: Clase principal que hereda de `Policy`.
*   `minimax()`: Implementación recursiva del algoritmo de búsqueda.
*   `heuristic()`: Evalúa la calidad del tablero para un jugador dado.
*   `drop_piece()`: Simula la caída de una pieza para explorar estados futuros sin modificar el tablero real.

## Requisitos

*   `numpy`
*   Entorno de Conecta 4 del torneo (clase `Policy`).

## Uso

Para integrar este agente en una partida:

```python
from groups.group_d.policy import Minimaxer

agente = Minimaxer(depth=5)
# En el bucle de juego:
# columna = agente.act(tablero)
```
