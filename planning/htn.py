from __future__ import annotations
from planning.utils import Queue
from planning.pddl import get_all_groundings
from planning.pddl import Action, Problem, apply_action, is_applicable


# ---------------------------------------------------------------------------
# HTN Infrastructure
# ---------------------------------------------------------------------------


class HLA:
    """
    A High-Level Action (HLA) in HTN planning.

    An HLA is an abstract task that can be refined into sequences of
    more primitive actions (or other HLAs). Each refinement is a list
    of HLA or Action objects.

    name:        Human-readable name for display
    refinements: List of possible refinements, each a list of HLA/Action objects
    """

    def __init__(self, name: str, refinements: list[list] | None = None) -> None:
        self.name = name
        self.refinements = refinements or []

    def __repr__(self) -> str:
        return f"HLA({self.name})"


def is_primitive(action: Action | HLA) -> bool:
    """Return True if action is a primitive (grounded Action), False if it is an HLA."""
    return isinstance(action, Action)


def is_plan_primitive(plan: list[Action | HLA]) -> bool:
    """Return True if every step in the plan is a primitive action."""
    return all(is_primitive(step) for step in plan)


# ---------------------------------------------------------------------------
# Punto 5a – hierarchicalSearch
# ---------------------------------------------------------------------------


def hierarchicalSearch(problem: Problem, hlas: list[HLA]) -> list[Action]:
    """
    HTN planning via BFS over hierarchical plan refinements.

    Start with an initial plan containing a single top-level HLA.
    At each step, find the first non-primitive step in the plan and
    replace it with one of its refinements. Continue until the plan
    is fully primitive and achieves the goal when executed from the
    initial state.

    Returns a list of primitive Action objects, or [] if no plan found.

    Tip: The search space consists of (partial plan, current plan index) pairs.
         Use a Queue (BFS) to explore all refinement choices fairly.
         A plan is a solution when:
           1. It contains only primitive actions (is_plan_primitive), AND
           2. Executing it from the initial state reaches a goal state.
         To simulate execution, apply each action in order using apply_action().
    """
    ### Your code here ###
    frontier = Queue()
    frontier.push(hlas)
    
    while not frontier.isEmpty():
        plan = frontier.pop()

        if is_plan_primitive(plan):
            state = problem.initial_state
            valid = True
            for action in plan:
                if not is_applicable(state, action):
                    valid = False
                    break
                state = apply_action(state, action)
            if valid and problem.isGoalState(state):
                return plan
        else:
            for i, step in enumerate(plan):
                if not is_primitive(step):
                    for refinement in step.refinements:
                        new_plan = plan[:i] + refinement + plan[i+1:]
                        frontier.push(new_plan)
                    break
    return []
    ### End of your code ###


# ---------------------------------------------------------------------------
# Punto 5b – HLA Definitions
# ---------------------------------------------------------------------------


def build_htn_hierarchy(problem: Problem) -> list[HLA]:
    """
    Build HTN HLAs for the rescue domain.

    The hierarchy defines four HLA types:
      - Navigate(from, to):       Move the robot step by step from one cell to another
      - PrepareSupplies(s, m):    Collect supplies and set them up at the medical post
      - ExtractPatient(p, m):     Pick up the patient and bring them to the medical post
      - FullRescueMission(s,p,m): Complete one rescue: prepare supplies + extract + rescue

    Refinements are built from the ground state to generate concrete Action objects.

    Tip: Refinements for Navigate are all single-step Move sequences between
         adjacent cells. PrepareSupplies and ExtractPatient chain Navigate HLAs
         with primitive PickUp, SetupSupplies, PutDown, and Rescue actions.
    """
    ### Your code here ###
    """"
    initial = problem.initial_state
    robot = problem.objects["robots"][0]
    patient = problem.objects["patients"][0]
    supplies = problem.objects["supplies"][0]
    medical_post = problem.objects["medical_posts"][0]
    
    robot_loc = None
    patient_loc = None
    supplies_loc = None
    
    for fluent in initial:
        if fluent[0] == "At":
            if fluent[1] == robot:
                robot_loc = fluent[2]
            elif fluent[1] == patient:
                patient_loc = fluent[2]
            elif fluent[1] == supplies:
                supplies_loc = fluent[2]                                                
    
    navigate_to_supplies = HLA("NavigateToSupplies")
    navigate_to_medical = HLA("NavigateToMedical")
    navigate_to_patient = HLA("NavigateToPatient")
    
    navigate_to_supplies.refinements = [[]]  
    navigate_to_medical.refinements = [[]]
    navigate_to_patient.refinements = [[]]
    
    pickup_supplies = Action(
        "PickUpSupplies",
        [
            ("At", robot, supplies_loc),
            ("At", supplies, supplies_loc),
            ("HandsFree", robot),
            ("Pickable", supplies)
        ],
        [],
        [
            ("Holding", robot, supplies)
        ],
        [
            ("At", supplies, supplies_loc),
            ("HandsFree", robot)
        ]
    )

    setup_supplies = Action(
        "SetupSupplies",
        [
            ("At", robot, medical_post),
            ("Holding", robot, supplies),
            ("MedicalPost", medical_post)
        ],
        [],
        [
            ("SuppliesReady", medical_post),
            ("HandsFree", robot)
        ],
        [
            ("Holding", robot, supplies)
        ]
    )

    pickup_patient = Action(
        "PickUpPatient",
        [
            ("At", robot, patient_loc),
            ("At", patient, patient_loc),
            ("HandsFree", robot),
            ("Pickable", patient)
        ],
        [],
        [
            ("Holding", robot, patient)
        ],
        [
            ("At", patient, patient_loc),
            ("HandsFree", robot)
        ]
    )

    putdown_patient = Action(
        "PutDownPatient",
        [
            ("At", robot, medical_post),
            ("Holding", robot, patient)
        ],
        [],
        [
            ("At", patient, medical_post),
            ("HandsFree", robot)
        ],
        [
            ("Holding", robot, patient)
        ]
    )

    rescue_patient = Action(
        "RescuePatient",
        [
            ("At", robot, medical_post),
            ("At", patient, medical_post),
            ("MedicalPost", medical_post),
            ("SuppliesReady", medical_post)
        ],
        [],
        [
            ("Rescued", patient)
        ],
        [
            ("At", patient, medical_post)
        ]
    )
    
    prepare = HLA("PrepareSupplies")
    prepare.refinements = [[navigate_to_supplies, pickup_supplies, navigate_to_medical, setup_supplies]]
    
    extract = HLA("ExtractPatient")
    extract.refinements = [[navigate_to_patient, pickup_patient, navigate_to_medical, putdown_patient]]
    
    rescue = HLA("RescuePatient")
    rescue.refinements = [[navigate_to_medical, rescue_patient]]

    full_mission = HLA("FullRescueMission")
    full_mission.refinements = [[prepare, extract, rescue]]
    
    return [full_mission]
    
    prompt:Ayúdame a transformar mi implementación de `build_htn_hierarchy(problem)` en una versión funcional para un planner
    HTN en Python. Actualmente tengo HLAs de navegación, acciones primitivas y refinamientos básicos, pero el planificador no 
    encuentra un plan válido o se queda ejecutando indefinidamente. Quiero que revises la lógica de navegación, los 
    refinamientos de los HLAs y la forma en que se generan los movimientos a partir de las acciones groundeadas, y que me 
    expliques cómo modificar el código para que el HTN pueda construir un plan correcto y eficiente usando movimientos válidos
    entre posiciones relevantes del mapa.

    initial = problem.initial_state
    robot = problem.objects["robots"][0]
    patient = problem.objects["patients"][0]
    supplies = problem.objects["supplies"][0]
    medical_post = problem.objects["medical_posts"][0]
    
    robot_loc = None
    patient_loc = None
    supplies_loc = None
    
    for fluent in initial:
        if fluent[0] == "At":
            if fluent[1] == robot:
                robot_loc = fluent[2]
            elif fluent[1] == patient:
                patient_loc = fluent[2]
            elif fluent[1] == supplies:
                supplies_loc = fluent[2] 
                
    all_actions = get_all_groundings(problem.domain,problem.objects)
    
    moves_to_supplies = []
    moves_to_medical_from_supplies = []
    moves_to_patient = []
    moves_to_medical_from_patient = []

    for action in all_actions:

        if not action.name.startswith("Move"):
            continue

        from_cell = None
        to_cell = None

        for fluent in action.precond_pos:
            if fluent[0] == "At" and fluent[1] == robot:
                from_cell = fluent[2]

        for fluent in action.add_list:
            if fluent[0] == "At" and fluent[1] == robot:
                to_cell = fluent[2]

        if from_cell == robot_loc and to_cell == supplies_loc:
            moves_to_supplies.append(action)

        if from_cell == supplies_loc and to_cell == medical_post:
            moves_to_medical_from_supplies.append(action)

        if from_cell == medical_post and to_cell == patient_loc:
            moves_to_patient.append(action)

        if from_cell == patient_loc and to_cell == medical_post:
            moves_to_medical_from_patient.append(action)

    navigate_to_supplies = HLA("NavigateToSupplies")
    navigate_to_supplies.refinements = [
        [a] for a in moves_to_supplies
    ]
    navigate_to_medical_supplies = HLA("NavigateToMedicalSupplies")
    navigate_to_medical_supplies.refinements = [
        [a] for a in moves_to_medical_from_supplies
    ]
    navigate_to_patient = HLA("NavigateToPatient")
    navigate_to_patient.refinements = [
    [a] for a in moves_to_patient
    ]
    
    navigate_to_medical_patient = HLA("NavigateToMedicalPatient")
    navigate_to_medical_patient.refinements = [
        [a] for a in moves_to_medical_from_patient
    ]
    
    pickup_supplies = Action(
        "PickUpSupplies",
        [
            ("At", robot, supplies_loc),
            ("At", supplies, supplies_loc),
            ("HandsFree", robot),
            ("Pickable", supplies)
        ],
        [],
        [
            ("Holding", robot, supplies)
        ],
        [
            ("At", supplies, supplies_loc),
            ("HandsFree", robot)
        ]
    )

    setup_supplies = Action(
        "SetupSupplies",
        [
            ("At", robot, medical_post),
            ("Holding", robot, supplies),
            ("MedicalPost", medical_post)
        ],
        [],
        [
            ("SuppliesReady", medical_post),
            ("HandsFree", robot)
        ],
        [
            ("Holding", robot, supplies)
        ]
    )

    pickup_patient = Action(
        "PickUpPatient",
        [
            ("At", robot, patient_loc),
            ("At", patient, patient_loc),
            ("HandsFree", robot),
            ("Pickable", patient)
        ],
        [],
        [
            ("Holding", robot, patient)
        ],
        [
            ("At", patient, patient_loc),
            ("HandsFree", robot)
        ]
    )

    putdown_patient = Action(
        "PutDownPatient",
        [
            ("At", robot, medical_post),
            ("Holding", robot, patient)
        ],
        [],
        [
            ("At", patient, medical_post),
            ("HandsFree", robot)
        ],
        [
            ("Holding", robot, patient)
        ]
    )

    rescue_patient = Action(
        "RescuePatient",
        [
            ("At", robot, medical_post),
            ("At", patient, medical_post),
            ("MedicalPost", medical_post),
            ("SuppliesReady", medical_post)
        ],
        [],
        [
            ("Rescued", patient)
        ],
        [
            ("At", patient, medical_post)
        ]
    )
    
    prepare = HLA("PrepareSupplies")
    prepare.refinements = [[navigate_to_supplies, pickup_supplies, navigate_to_medical_supplies, setup_supplies]]
    
    extract = HLA("ExtractPatient")
    extract.refinements = [[navigate_to_patient, pickup_patient, navigate_to_medical_patient, putdown_patient]]
    
    rescue = HLA("RescuePatient")
    rescue.refinements = [[rescue_patient]]

    full_mission = HLA("FullRescueMission")
    full_mission.refinements = [[prepare, extract, rescue]]
    
    return [full_mission]
    
    Prompt: Ayúdame a mejorar mi implementación de build_htn_hierarchy(problem) para que el HTN no solo funcione en layouts 
    pequeños con movimientos directos, sino también en mapas más grandes como htnBase, donde el robot necesita varios 
    movimientos intermedios para llegar a un objetivo
    """
    initial = problem.initial_state

    robot = problem.objects["robots"][0]
    patient = problem.objects["patients"][0]
    supplies = problem.objects["supplies"][0]
    medical_post = problem.objects["medical_posts"][0]

    robot_loc = None
    patient_loc = None
    supplies_loc = None

    for fluent in initial:

        if fluent[0] == "At":

            if fluent[1] == robot:
                robot_loc = fluent[2]

            elif fluent[1] == patient:
                patient_loc = fluent[2]

            elif fluent[1] == supplies:
                supplies_loc = fluent[2]

    all_actions = get_all_groundings(
        problem.domain,
        problem.objects
    )

    def build_navigation_plan(start, goal):

        queue = [([], start)]
        visited = set()

        queue = [([], start)]

        while queue:

            path, current = queue.pop(0)

            if current == goal:
                return path

            if current in visited:
                continue

            visited.add(current)
            
            temp_state = set(initial)
            
            for fluent in list(temp_state):

                if fluent[0] == "At" and fluent[1] == robot:
                    temp_state.remove(fluent)

            temp_state.add(("At", robot, current))

            temp_state = frozenset(temp_state)

            for action in all_actions:

                if not action.name.startswith("Move"):
                    continue
                if not is_applicable(temp_state, action):
                    continue
                
                next_pos = None

                for fluent in action.add_list:

                    if fluent[0] == "At" and fluent[1] == robot:
                        next_pos = fluent[2]

                if next_pos is not None:

                    queue.append(
                        (
                            path + [action],
                            next_pos
                        )
                    )
        return []

    path_to_supplies = build_navigation_plan(
        robot_loc,
        supplies_loc
    )

    path_to_medical_supplies = build_navigation_plan(
        supplies_loc,
        medical_post
    )

    path_to_patient = build_navigation_plan(
        medical_post,
        patient_loc
    )

    path_to_medical_patient = build_navigation_plan(
        patient_loc,
        medical_post
    )

    navigate_to_supplies = HLA("NavigateToSupplies")

    navigate_to_supplies.refinements = [
        path_to_supplies
    ]

    navigate_to_medical_supplies = HLA(
        "NavigateToMedicalSupplies"
    )

    navigate_to_medical_supplies.refinements = [
        path_to_medical_supplies
    ]

    navigate_to_patient = HLA("NavigateToPatient")

    navigate_to_patient.refinements = [
        path_to_patient
    ]

    navigate_to_medical_patient = HLA(
        "NavigateToMedicalPatient"
    )

    navigate_to_medical_patient.refinements = [
        path_to_medical_patient
    ]

    pickup_supplies = Action(
        "PickUpSupplies",
        [
            ("At", robot, supplies_loc),
            ("At", supplies, supplies_loc),
            ("HandsFree", robot),
            ("Pickable", supplies)
        ],
        [],
        [
            ("Holding", robot, supplies)
        ],
        [
            ("At", supplies, supplies_loc),
            ("HandsFree", robot)
        ]
    )

    setup_supplies = Action(
        "SetupSupplies",
        [
            ("At", robot, medical_post),
            ("Holding", robot, supplies),
            ("MedicalPost", medical_post)
        ],
        [],
        [
            ("SuppliesReady", medical_post),
            ("HandsFree", robot)
        ],
        [
            ("Holding", robot, supplies)
        ]
    )

    pickup_patient = Action(
        "PickUpPatient",
        [
            ("At", robot, patient_loc),
            ("At", patient, patient_loc),
            ("HandsFree", robot),
            ("Pickable", patient)
        ],
        [],
        [
            ("Holding", robot, patient)
        ],
        [
            ("At", patient, patient_loc),
            ("HandsFree", robot)
        ]
    )

    putdown_patient = Action(
        "PutDownPatient",
        [
            ("At", robot, medical_post),
            ("Holding", robot, patient)
        ],
        [],
        [
            ("At", patient, medical_post),
            ("HandsFree", robot)
        ],
        [
            ("Holding", robot, patient)
        ]
    )

    rescue_patient = Action(
        "RescuePatient",
        [
            ("At", robot, medical_post),
            ("At", patient, medical_post),
            ("MedicalPost", medical_post),
            ("SuppliesReady", medical_post)
        ],
        [],
        [
            ("Rescued", patient)
        ],
        [
            ("At", patient, medical_post)
        ]
    )

    prepare = HLA("PrepareSupplies")

    prepare.refinements = [[
        navigate_to_supplies,
        pickup_supplies,
        navigate_to_medical_supplies,
        setup_supplies
    ]]

    extract = HLA("ExtractPatient")

    extract.refinements = [[
        navigate_to_patient,
        pickup_patient,
        navigate_to_medical_patient,
        putdown_patient
    ]]

    rescue = HLA("RescuePatient")

    rescue.refinements = [[
        rescue_patient
    ]]

    full_mission = HLA("FullRescueMission")

    full_mission.refinements = [[
        prepare,
        extract,
        rescue
    ]]

    return [full_mission]
    ### End of your code ###
