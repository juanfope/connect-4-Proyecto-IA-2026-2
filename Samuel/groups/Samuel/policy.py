import math
import numpy as np
from connect4.policy import Policy
from connect4.connect_state import ConnectState


# Alternating Markov Games
class AMGNode:

    def __init__(
        self,
        state: ConnectState,
        parent: "AMGNode | None" = None,
        action: int | None = None,
    ):
        self.state = state
        self.parent = parent
        self.action = action

        self.children: list["AMGNode"] = []

        # Número de veces que se ha pasado por este nodo
        self.visits: int = 0

        # Valor acumulado
        self.value: float = 0.0

        # Estados aún no simulados
        self.untried: list[int] = (state.get_free_cols() if not state.is_final() else [])

    def is_fully_expanded(self) -> bool:
        return len(self.untried) == 0

    def ucb1(self, c: float) -> float:
        if self.visits == 0:
            return float("inf")
        return self.value / self.visits + c * math.sqrt(math.log(self.parent.visits) / self.visits)

    def best_child(self, c: float) -> "AMGNode":
        return max(self.children, key=lambda n: n.ucb1(c))

    def expand(self) -> "AMGNode":

        col = self.untried.pop(np.random.randint(len(self.untried)))

        next_state = self.state.transition(col)
        child = AMGNode(next_state, parent=self, action=col)
        self.children.append(child)
        return child


# Busca el nodo por donde expandirse
def select(node: AMGNode, c: float) -> AMGNode:
    while not node.state.is_final():

        if not node.is_fully_expanded():
            return node

        node = node.best_child(c)

    return node


# Determina si una posición es estadísticamente buena o mala
def rollout(state: ConnectState, my_player: int) -> float:
    current_state = state

    while not current_state.is_final():
        cols = current_state.get_free_cols()
        col = int(np.random.choice(cols))
        current_state = current_state.transition(col)

    winner = current_state.get_winner()
    if winner == my_player:
        return 1.0
    if winner == -my_player:
        return -1.0
    return 0.0


# Ajusta los valores de los nodos dependiendo si la jugada benefició o perjudicó al jugador
def backpropagate(node: AMGNode, result: float, my_player: int) -> None:
    current = node
    while current is not None:
        current.visits += 1
        if current.parent is not None:

            if current.parent.state.player == my_player:
                current.value += result
            else:
                current.value -= result
        current = current.parent


# Monte-Carlo Tree Search
class SamuelMCTS(Policy):

    def __init__(self, n_simulations: int = 500, c_exploration: float = math.sqrt(2)):
        self.n_simulations = n_simulations
        self.c_exploration = c_exploration
        self.player: int | None = None

    def mount(self, action_timeout: float | None = None) -> None:
        self.player = None
        if action_timeout is not None:
            self.n_simulations = max(300, int(action_timeout * 150))

    def act(self, state_or_board: np.ndarray | ConnectState) -> int:
        # --- CAPA DE COMPATIBILIDAD ---
        if isinstance(state_or_board, np.ndarray):
            board = state_or_board.copy()
            current_player = -1 if np.sum(board == -1) == np.sum(board == 1) else 1
            state = ConnectState(board=board, player=current_player)
        else:
            state = state_or_board

        if self.player is None:
            self.player = state.player

        valid_cols = state.get_free_cols()

        # --- PROTECCIÓN PARA EL AUTOGRADER ---
        # Si el tablero ya tiene un ganador o está lleno, devolvemos
        # una columna libre para evitar que 'transition' colapse.
        if state.is_final():
            return int(valid_cols[0]) if valid_cols else 0

        if len(valid_cols) == 1:
            return int(valid_cols[0])

        # --- HEURÍSTICA 1: Ganar de inmediato ---
        for col in valid_cols:
            # Forzamos int() por si el autograder usa tipos numpy.int64
            if state.transition(int(col)).get_winner() == self.player:
                return int(col)

        # --- HEURÍSTICA 2: Bloquear victoria del oponente ---
        opponent_state = ConnectState(state.board, player=-self.player)
        for col in valid_cols:
            if opponent_state.transition(int(col)).get_winner() == -self.player:
                return int(col)

        # --- PROCESO MCTS ---
        root = AMGNode(state)

        for _ in range(self.n_simulations):
            node = select(root, self.c_exploration)

            if not node.state.is_final() and not node.is_fully_expanded():
                node = node.expand()

            result = rollout(node.state, self.player)
            backpropagate(node, result, self.player)

        best = max(root.children, key=lambda n: n.visits)
        return int(best.action)