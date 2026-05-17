from __future__ import annotations

from collections import deque
from collections.abc import Callable

from planning.pddl import (
    Action,
    ActionSchema,
    Problem,
    State,
    Objects,
    get_all_groundings,
)
from planning.utils import Queue, PriorityQueue
from planning.heuristics import nullHeuristic


# ---------------------------------------------------------------------------
# Reference implementation – read and understand before coding the rest.
# ---------------------------------------------------------------------------


def tinyBaseSearch(problem: Problem) -> list[Action]:
    """
    Hardcoded plan for the tinyBase layout.
    The robot at (1,4) must: pick up supplies at (1,3), set them up at (1,2),
    pick up the patient at (1,1), bring them to (1,2), and execute Rescue.

    Useful to understand the Action object format and plan structure.
    """
    robot = "robot"
    supplies = "supplies_0"
    patient = "patient_0"

    c14 = (1, 4)  # robot start
    c13 = (1, 3)  # supplies
    c12 = (1, 2)  # medical post
    c11 = (1, 1)  # patient

    plan = [
        Action(
            "Move(robot,(1,4),(1,3))",
            [("At", robot, c14), ("Adjacent", c14, c13), ("Free", c13)],
            [],
            [("At", robot, c13), ("Free", c14)],
            [("At", robot, c14), ("Free", c13)],
        ),
        Action(
            "PickUp(robot,supplies_0,(1,3))",
            [
                ("At", robot, c13),
                ("At", supplies, c13),
                ("HandsFree", robot),
                ("Pickable", supplies),
            ],
            [],
            [("Holding", robot, supplies)],
            [("At", supplies, c13), ("HandsFree", robot)],
        ),
        Action(
            "Move(robot,(1,3),(1,2))",
            [("At", robot, c13), ("Adjacent", c13, c12), ("Free", c12)],
            [],
            [("At", robot, c12), ("Free", c13)],
            [("At", robot, c13), ("Free", c12)],
        ),
        Action(
            "SetupSupplies(robot,supplies_0,(1,2))",
            [("At", robot, c12), ("MedicalPost", c12), ("Holding", robot, supplies)],
            [("SuppliesReady", c12)],
            [("SuppliesReady", c12), ("HandsFree", robot)],
            [("Holding", robot, supplies)],
        ),
        Action(
            "Move(robot,(1,2),(1,1))",
            [("At", robot, c12), ("Adjacent", c12, c11), ("Free", c11)],
            [],
            [("At", robot, c11), ("Free", c12)],
            [("At", robot, c12), ("Free", c11)],
        ),
        Action(
            "PickUp(robot,patient_0,(1,1))",
            [
                ("At", robot, c11),
                ("At", patient, c11),
                ("HandsFree", robot),
                ("Pickable", patient),
            ],
            [],
            [("Holding", robot, patient)],
            [("At", patient, c11), ("HandsFree", robot)],
        ),
        Action(
            "Move(robot,(1,1),(1,2))",
            [("At", robot, c11), ("Adjacent", c11, c12), ("Free", c12)],
            [],
            [("At", robot, c12), ("Free", c11)],
            [("At", robot, c11), ("Free", c12)],
        ),
        Action(
            "PutDown(robot,patient_0,(1,2))",
            [("At", robot, c12), ("Holding", robot, patient)],
            [],
            [("At", patient, c12), ("HandsFree", robot)],
            [("Holding", robot, patient)],
        ),
        Action(
            "Rescue(robot,patient_0,(1,2))",
            [
                ("At", robot, c12),
                ("At", patient, c12),
                ("MedicalPost", c12),
                ("SuppliesReady", c12),
            ],
            [],
            [("Rescued", patient)],
            [("At", patient, c12)],
        ),
    ]
    return plan


# ---------------------------------------------------------------------------
# Punto 2 – Forward Planning
# ---------------------------------------------------------------------------


def forwardBFS(problem: Problem) -> list[Action]:
    """
    Forward BFS in state space.

    Explore states reachable from the initial state by applying actions,
    in breadth-first order, until a goal state is found.

    Returns a list of Action objects forming a valid plan, or [] if no plan exists.

    Tip: The state is a frozenset of fluents. Use problem.getSuccessors(state)
         to get (next_state, action, cost) triples. Track visited states to
         avoid revisiting the same state twice (graph search, not tree search).
    
Código inicial:

    start = problem.getStartState()

    if problem.isGoalState(start):
        return []
        
    frontier = deque()
    frontier.append(start)
    visited = set() 
    parent = {}

    while len(frontier) > 0:
        current = frontier.popleft()
        successors = problem.getSuccessors(current)
        for successor in successors:
            nextState = successor[0]
            action = successor[1]

            if nextState not in visited:
                parent[nextState] = (current, action)

                if problem.isGoalState(nextState):
                    moves = []
                    state = nextState

                    while state != start:
                        previous = parent[state][0]
                        step = parent[state][1]
                        moves.append(step)
                        state = previous

                    moves.reverse()
                    return moves

                visited.add(nextState)
                frontier.append(nextState)

    return []

    Prompt:
    Tu tarea es arreglar este código de búsqueda forwardBFS para que sea más claro, 
    más compacto y más cercano a una implementación formal. Mantén exactamente la misma lógica 
    general: obtener el estado inicial, revisar si ya es meta, usar una cola como frontera, 
    guardar los estados visitados, guardar padres para reconstruir el plan y retornar la lista 
    de acciones cuando se encuentre la meta. También corrige detalles menores de estilo, nombres 
    de variables y estructura, así como posibles errores logicos que puedan existir. Al finalizar 
    entrégame una lista de los cambios realizados.


    Cambios realizados:
    1. Se cambió `start` por `start_state`.
    2. Se cambió `frontier = deque(); frontier.append(start)` por `frontier = deque([start_state])`.
    3. Se cambió `visited = set()` por `visited = {start_state}`.
    4. Se cambió `parent = {}` por `parent: dict[State, tuple[State, Action]] = {}`.
    5. Se cambió `while len(frontier) > 0:` por `while frontier:`.
    6. Se cambió `current` por `current_state`.
    7. Se reemplazó el acceso manual `successor[0]` y `successor[1]` por desempaquetado directo: `for next_state, action, _ in ...`.
    8. Se usó `continue` cuando el estado ya estaba visitado.
    9. Se cambió `moves` por `plan`.
    10. Se simplificó la reconstrucción del camino con `state, step = parent[state]`.  
    """
    start_state = problem.getStartState()
    if problem.isGoalState(start_state):
        return []

    frontier = deque([start_state])
    visited = {start_state}
    parent: dict[State, tuple[State, Action]] = {}

    while frontier:
        current_state = frontier.popleft()

        for next_state, action, _ in problem.getSuccessors(current_state):
            if next_state in visited:
                continue

            parent[next_state] = (current_state, action)
            if problem.isGoalState(next_state):
                plan: list[Action] = []
                state = next_state
                while state != start_state:
                    state, step = parent[state]
                    plan.append(step)
                plan.reverse()
                return plan

            visited.add(next_state)
            frontier.append(next_state)

    return []


# ---------------------------------------------------------------------------
# Punto 3 – Backward Planning
# ---------------------------------------------------------------------------


def regress(goal_set: State, action: Action) -> State | None:
    """
    Compute the regression of goal_set through action.

    Given a goal description (set of fluents that must be true) and an action,
    return the new goal description that, if satisfied, guarantees the original
    goal is satisfied after executing action.

    REGRESS(g, a) = (g − ADD(a)) ∪ PRECOND_pos(a)
        IF:  ADD(a) ∩ g ≠ ∅   (action is relevant: contributes to the goal)
        AND: DEL(a) ∩ g = ∅   (action does not undo any goal fluent)
    Returns None if the action is not relevant or creates a contradiction.

    Tip: Use frozenset operations: intersection (&), difference (-), union (|).
         Check relevance first, then check for contradictions, then compute.
    """
    ### Your code here ###
    if action.add_list.isdisjoint(goal_set):
        return None
    if not action.del_list.isdisjoint(goal_set):
        return None
    return (goal_set - action.add_list) | action.precond_pos
    ### End of your code ###


def backwardSearch(problem: Problem) -> list[Action]:
    """
    Backward search (regression search) from the goal.

    Start from the goal description and apply action regressions until
    the resulting goal is satisfied by the initial state.

    Returns a list of Action objects forming a valid plan (in forward order),
    or [] if no plan exists.

    Tip: The "state" in backward search is a frozenset of fluents that must
         be true (a partial goal description). The initial state is reached
         when all fluents in the current goal are satisfied by problem.initial_state.
         Only consider actions whose add_list has at least one unsatisfied goal fluent
         (relevant actions). Use regress() to compute the new subgoal.
         Skip subgoals that contain static predicates (MedicalPost, Adjacent,
         Pickable) that are false in the initial state — these are dead ends.

Version inicial si IAG:

    initial_state = problem.getStartState()
    goal = problem.goal
    all_actions = get_all_groundings(problem.domain, problem.objects)
    STATIC = {"Adjacent", "MedicalPost", "Pickable"}

    frontier = Queue()
    frontier.push((goal, []))
    visited = {goal}

    while not frontier.isEmpty():
        current_goal, plan_rev = frontier.pop()
        problem._expanded += 1
        if current_goal.issubset(initial_state):
            return list(reversed(plan_rev))
        for action in all_actions:
            new_goal = regress(current_goal, action)
            if new_goal is None:
                continue
            if any(f[0] in STATIC and f not in initial_state for f in new_goal):
                continue
            if new_goal in visited:
                continue
            visited.add(new_goal)
            frontier.push((new_goal, plan_rev + [action]))
    return []

prompt: Tengo esta version de backwardSearch que funciona bien en tinyBase pero
se queda sin terminar en layouts mas grandes. El problema es que explora demasiados
subobjetivos inconsistentes, por ejemplo el robot en dos celdas a la vez, o sosteniendo
un objeto y con HandsFree al mismo tiempo. Como puedo agregar podas que descarten esos
subobjetivos imposibles sin cambiar la correctitud del algoritmo? Tambien, hay alguna
forma de no iterar sobre todas las acciones en cada nodo sino solo las relevantes?
"""
    ### Your code here ###
    initial_state = problem.getStartState()
    goal = problem.goal
    all_actions = get_all_groundings(problem.domain, problem.objects)

    fluent_to_actions: dict = {}
    for a in all_actions:
        for f in a.add_list:
            fluent_to_actions.setdefault(f, []).append(a)

    STATIC = {"Adjacent", "MedicalPost", "Pickable"}

    def is_consistent(sub: State) -> bool:
        at_by_entity: dict = {}
        for f in sub:
            if f[0] == "At":
                entity, loc = f[1], f[2]
                if entity in at_by_entity and at_by_entity[entity] != loc:
                    return False
                at_by_entity[entity] = loc
        if "robot" in at_by_entity:
            if ("Free", at_by_entity["robot"]) in sub:
                return False
        if ("HandsFree", "robot") in sub:
            if any(f[0] == "Holding" for f in sub):
                return False
        held = {f[2] for f in sub if f[0] == "Holding"}
        if held and any(f[0] == "At" and f[1] in held for f in sub):
            return False
        return True

    frontier = Queue()
    frontier.push((goal, []))
    visited: set[State] = {goal}

    while not frontier.isEmpty():
        current_goal, plan_rev = frontier.pop()
        problem._expanded += 1
        if current_goal.issubset(initial_state):
            return list(reversed(plan_rev))
        candidates: set = set()
        for f in current_goal:
            for a in fluent_to_actions.get(f, ()):
                candidates.add(a)
        for action in candidates:
            new_goal = regress(current_goal, action)
            if new_goal is None:
                continue
            if any(f[0] in STATIC and f not in initial_state for f in new_goal):
                continue
            if not is_consistent(new_goal):
                continue
            if new_goal in visited:
                continue
            visited.add(new_goal)
            frontier.push((new_goal, plan_rev + [action]))

    return []
    ### End of your code ###


# ---------------------------------------------------------------------------
# Punto 4 – A* Planner
# ---------------------------------------------------------------------------

# Heuristic signature:  heuristic(state, goal, domain, objects) -> float
Heuristic = Callable[[State, State, list[ActionSchema], Objects], float]


def aStarPlanner(
    problem: Problem,
    heuristic: Heuristic = nullHeuristic,
) -> list[Action]:
    """
    Forward A* search guided by a heuristic.

    Combines the real accumulated cost g(n) with the heuristic estimate h(n)
    to prioritize which state to expand next: f(n) = g(n) + h(n).

    Returns a list of Action objects forming a valid plan, or [] if no plan exists.

    Tip: The heuristic signature is heuristic(state, goal, domain, objects) → float.
         Use PriorityQueue with priority = g + h(next_state).
         Track the best g-cost seen for each state to avoid stale expansions.

Version inicial si IAG: 
    
     frontier = PriorityQueue()
     start_state = problem.getStartState()
     frontier.push((start_state, []), 0)
     
     while not frontier.isEmpty():
         current_state, plan = frontier.pop()
         
         if problem.isGoalState(current_state):
             return plan
             
         for next_state, action, cost in problem.getSuccessors(current_state):
             new_plan = plan + [action]
             h_cost = heuristic(next_state, problem.goal, problem.domain, problem.objects)
             frontier.push((next_state, new_plan), h_cost)
     
     return []

prompt: Ayudame a corregir esta funcion aStarPlanner, trate de hacer la estructura basica de A* usando la PriorityQueue,
pero el programa se queda corriendo infinitamente o expande nodos de forma ineficiente para este problema PDDL.
Me faltan detalles importantes sobre el calculo correcto de f(n) con la heuristica y como manejar estados
repetidos, pero no se como ajustarlo, completalo para que funcione en este caso especifico?
"""
    ### Your code here ###
    frontier = PriorityQueue()
    start_state = problem.getStartState()
    h_start = heuristic(start_state, problem.goal, problem.domain, problem.objects)
    frontier.push((start_state, []), 0 + h_start)
    best_g = {start_state: 0}

    while not frontier.isEmpty():
        current_state, plan = frontier.pop()

        if problem.isGoalState(current_state):
            return plan

        g_cost = len(plan)

        if g_cost > best_g.get(current_state, float('inf')):
            continue

        for next_state, action, cost in problem.getSuccessors(current_state):
            new_g = g_cost + cost
            new_plan = plan + [action]

            if new_g < best_g.get(next_state, float('inf')):
                best_g[next_state] = new_g
                h_cost = heuristic(next_state, problem.goal, problem.domain, problem.objects)
                f_cost = new_g + h_cost
                frontier.push((next_state, new_plan), f_cost)

    return []
    ### End of your code ###


# Aliases used by the command-line argument parser
tinyBaseSearch = tinyBaseSearch
forwardBFS = forwardBFS
backwardSearch = backwardSearch
aStarPlanner = aStarPlanner
