#Imports
from typing import List, Tuple, Dict, Callable
import copy
from copy import deepcopy
import numpy as np
import streamlit as st
import random

#Default full world 
full_world = [
['🌾', '🌾', '🌾', '🌾', '🌾', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾'],
['🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌾', '🌾', '🗻', '🗻', '🗻', '🗻', '🗻', '🗻', '🗻', '🌾', '🌾'],
['🌾', '🌾', '🌾', '🌾', '🗻', '🗻', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🗻', '🗻', '🗻', '🪨', '🪨', '🪨', '🗻', '🗻', '🪨', '🪨'],
['🌾', '🌾', '🌾', '🌾', '🪨', '🗻', '🗻', '🗻', '🌲', '🌲', '🌲', '🌲', '🐊', '🐊', '🌲', '🌲', '🌲', '🌲', '🌲', '🌾', '🌾', '🪨', '🪨', '🗻', '🗻', '🪨', '🌾'],
['🌾', '🌾', '🌾', '🪨', '🪨', '🗻', '🗻', '🌲', '🌲', '🌾', '🌾', '🐊', '🐊', '🐊', '🐊', '🌲', '🌲', '🌲', '🌾', '🌾', '🌾', '🪨', '🗻', '🗻', '🗻', '🪨', '🌾'],
['🌾', '🪨', '🪨', '🪨', '🗻', '🗻', '🪨', '🪨', '🌾', '🌾', '🌾', '🌾', '🐊', '🐊', '🐊', '🐊', '🐊', '🌾', '🌾', '🌾', '🌾', '🌾', '🪨', '🗻', '🪨', '🌾', '🌾'],
['🌾', '🪨', '🪨', '🗻', '🗻', '🪨', '🪨', '🌾', '🌾', '🌾', '🌾', '🪨', '🗻', '🗻', '🗻', '🐊', '🐊', '🐊', '🌾', '🌾', '🌾', '🌾', '🌾', '🪨', '🌾', '🌾', '🌾'],
['🌾', '🌾', '🪨', '🪨', '🪨', '🪨', '🪨', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🪨', '🗻', '🗻', '🗻', '🐊', '🐊', '🐊', '🌾', '🌾', '🪨', '🪨', '🪨', '🌾', '🌾'],
['🌾', '🌾', '🌾', '🪨', '🪨', '🪨', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🪨', '🪨', '🗻', '🗻', '🌾', '🐊', '🐊', '🌾', '🌾', '🪨', '🪨', '🪨', '🌾', '🌾', '🌾'],
['🌾', '🌾', '🌾', '🐊', '🐊', '🐊', '🌾', '🌾', '🪨', '🪨', '🪨', '🗻', '🗻', '🗻', '🗻', '🌾', '🌾', '🌾', '🐊', '🌾', '🪨', '🪨', '🪨', '🌾', '🌾', '🌾', '🌾'],
['🌾', '🌾', '🐊', '🐊', '🐊', '🐊', '🐊', '🌾', '🪨', '🪨', '🗻', '🗻', '🗻', '🪨', '🌾', '🌾', '🌾', '🌾', '🌾', '🪨', '🗻', '🗻', '🗻', '🪨', '🌾', '🌾', '🌾'],
['🌾', '🐊', '🐊', '🐊', '🐊', '🐊', '🌾', '🌾', '🪨', '🗻', '🗻', '🪨', '🌾', '🌾', '🌾', '🌾', '🐊', '🐊', '🌾', '🌾', '🪨', '🗻', '🗻', '🪨', '🌾', '🌾', '🌾'],
['🐊', '🐊', '🐊', '🐊', '🐊', '🌾', '🌾', '🪨', '🪨', '🗻', '🗻', '🪨', '🌾', '🐊', '🐊', '🐊', '🐊', '🌾', '🌾', '🌾', '🪨', '🗻', '🪨', '🌾', '🌾', '🌾', '🌾'],
['🌾', '🐊', '🐊', '🐊', '🐊', '🌾', '🌾', '🪨', '🌲', '🌲', '🪨', '🌾', '🌾', '🌾', '🌾', '🐊', '🐊', '🐊', '🐊', '🌾', '🌾', '🪨', '🌾', '🌾', '🌾', '🌾', '🌾'],
['🌾', '🌾', '🌾', '🌾', '🗻', '🌾', '🌾', '🌲', '🌲', '🌲', '🌲', '🪨', '🪨', '🪨', '🪨', '🌾', '🐊', '🐊', '🐊', '🌾', '🌾', '🪨', '🗻', '🪨', '🌾', '🌾', '🌾'],
['🌾', '🌾', '🌾', '🗻', '🗻', '🗻', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🗻', '🗻', '🗻', '🪨', '🪨', '🌾', '🐊', '🌾', '🪨', '🗻', '🗻', '🪨', '🌾', '🌾', '🌾'],
['🌾', '🌾', '🗻', '🗻', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🗻', '🗻', '🗻', '🌾', '🌾', '🗻', '🗻', '🗻', '🌾', '🌾', '🌾', '🌾', '🌾'],
['🌾', '🌾', '🌾', '🗻', '🗻', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🗻', '🗻', '🗻', '🗻', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾'],
['🌾', '🌾', '🌾', '🗻', '🗻', '🗻', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌾', '🌾', '🌾', '🪨', '🪨', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾'],
['🌾', '🌾', '🌾', '🌾', '🗻', '🗻', '🗻', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🐊', '🐊', '🐊', '🐊'],
['🌾', '🌾', '🪨', '🪨', '🪨', '🪨', '🗻', '🗻', '🌲', '🌲', '🌲', '🌲', '🌲', '🌾', '🗻', '🌾', '🌾', '🌾', '🌾', '🌾', '🐊', '🐊', '🐊', '🐊', '🐊', '🐊', '🐊'],
['🌾', '🌾', '🌾', '🌾', '🪨', '🪨', '🪨', '🗻', '🗻', '🗻', '🌲', '🌲', '🗻', '🗻', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🐊', '🐊', '🐊', '🐊', '🐊', '🐊', '🐊'],
['🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🪨', '🪨', '🪨', '🗻', '🗻', '🗻', '🗻', '🌾', '🌾', '🌾', '🌾', '🪨', '🪨', '🌾', '🌾', '🐊', '🐊', '🐊', '🐊', '🐊', '🐊'],
['🌾', '🪨', '🪨', '🌾', '🌾', '🪨', '🪨', '🪨', '🪨', '🪨', '🌾', '🌾', '🌾', '🌾', '🌾', '🪨', '🪨', '🗻', '🗻', '🪨', '🪨', '🌾', '🐊', '🐊', '🐊', '🐊', '🐊'],
['🪨', '🗻', '🪨', '🪨', '🪨', '🪨', '🌾', '🌾', '🌾', '🌾', '🌾', '🗻', '🗻', '🗻', '🪨', '🪨', '🗻', '🗻', '🌾', '🗻', '🗻', '🪨', '🪨', '🐊', '🐊', '🐊', '🐊'],
['🪨', '🗻', '🗻', '🗻', '🪨', '🌾', '🌾', '🌾', '🌾', '🌾', '🪨', '🪨', '🗻', '🗻', '🗻', '🗻', '🪨', '🪨', '🪨', '🪨', '🗻', '🗻', '🗻', '🐊', '🐊', '🐊', '🐊'],
['🪨', '🪨', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🪨', '🪨', '🪨', '🪨', '🪨', '🌾', '🌾', '🌾', '🌾', '🪨', '🪨', '🪨', '🌾', '🌾', '🌾']
]

#Move set
MOVES = [(0,-1), (1,0), (0,1), (-1,0)]

#Costs
COSTS = { '🌾': 1, '🌲': 3, '🪨': 5, '🐊': 7, '🗻': 10000}

def successors( current_state_coord: Tuple[int, int], moves: List[Tuple[int, int]], world: List[List[str]]) -> List[Tuple[int, int]]: 
    '''
    `successors` takes the current state coordinates and determines all next possible
    children by iteratively adding all moves to said current coordinates. 
    Coordinates outside the range of `world` are not included in the full list of all possible successors. 
    This part is critical as A* search must evaluate the the optimal next state and all options should be considered. 
     
    **current_state_coord** Tuple[int, int]: `Tuple` containing the current state coordinates.
    **moves** List[Tuple[int, int]]: the legal movement model expressed in offsets in **world**.
    **world** List[List[str]]: the actual context for the navigation problem.
     
    **children**: List[Tuple[int, int]] documentation of the returned value and type.
    '''
    
    children = []
    
    for move in moves:
        child = (current_state_coord[0] + move[0], current_state_coord[1] + move[1])
        if all(coordinates >= 0 for coordinates in child) and (child[0] <= (len(world[0])-1)) and (child[1] <= (len(world)-1)):
            children.append(child)

    return children


def push( stack: List[Tuple[Tuple[int, int], int, int, Tuple[int, int]]], state: Tuple[Tuple[int, int], int, int, Tuple[int, int]]) -> None:
    '''
    `push` is a helper function to append states to a stack of choice (frontier or explored).
    This is to help mimic priority queue functionality.
     
    **stack** List[Tuple[Tuple[int, int], int, int, Tuple[int, int]]]: `List` of states serving the functionality of a stack.
    **state** Tuple[Tuple[int, int], int, int, Tuple[int, int]]): `Tuple` containing state cooridnates, f_cost, g_cost, parent state.
    '''
    stack.append(state)

def pop( stack: List[Tuple[Tuple[int, int], int, int, Tuple[int, int]]]) -> Tuple[Tuple[int, int], int, int, Tuple[int, int]]:
    '''
    `pop` is a helper function that traverses states last to first and returns the lowest cost state. 
    This is to help mimic priority queue functionality.
     
    **stack** List[Tuple[Tuple[int, int], int, int, Tuple[int, int]]]: `List` of states serving the functionality of a stack.
     
    **next_state** Tuple[Tuple[int, int], int, int, Tuple[int, int]]: `Tuple` containing lowest f_cost state in stack.
    '''
    minimum_f_cost = float('inf')
    
    for state in reversed(stack):
        if state[1] < minimum_f_cost:
            minimum_f_cost = state[1]
            next_state = state
    
    stack.remove(next_state)
    
    return next_state

def is_child_in_stack( stack: List[Tuple[Tuple[int, int], int, int, Tuple[int, int]]], child: Tuple[int, int]) -> bool:
    '''
    `is_child_in_stack` returns a bool indicating if the child already exisists in the stack. 
    Graph searches will avoid repeating visited states to combat cycling. 
    If child not in the frontier and not in explored, it is added to the frontier. 
    If child in frontier, but a cheaper option that what currently exists, it is added to the frontier. 

    **stack** List[Tuple[Tuple[int, int], int, int, Tuple[int, int]]]: `List` of states serving the functionality of a stack.
    **child** Tuple[int, int]: Coordinates of state to check existence in stack

    **bool**: True if child in stack, False otherwise.
    '''
    
    for state in stack:
        if child == state[0]:
            return True
    return False

def get_existing_child_state_in_stack( stack: List[Tuple[Tuple[int, int], int, int, Tuple[int, int]]], child: Tuple[int, int]) -> Tuple[Tuple[int, int], int, int, Tuple[int, int]]:
    '''
    `get_existing_child_state_in_stack` returns the state of the child already visited in the frontier. 
    If child in frontier, the f_costs must be compared to determine best route. Child will be added to frontier if cheaper. 
 
    **stack** List[Tuple[Tuple[int, int], int, int, Tuple[int, int]]]: `List` of states serving the functionality of a stack.
    **child** Tuple[int, int]: Coordinates of state to check existence in stack
     
    **state** Tuple[Tuple[int, int], int, int, Tuple[int, int]]: `Tuple` containing matching state of the child.
    '''
    for state in stack:
        if child == state[0]:
            return state
    return None


def is_goal( child: Tuple[int, int], goal: Tuple[int, int]) -> bool:
    '''
    `is_goal` is run on each child state to see if it is the goal. If true, the search is stopped and the path is returned. If false, the loop is continued unit the goal state is found or the frontier is empty. **Used by**: [a_star_search](#a_star_search)

    **child** Tuple[int, int]: `Tuple` containing the child state coordinates.
    **goal** Tuple[int, int]: the desired goal position for the bot, `(x, y)`.
     
    **bool**: True if the state coordinates are equal to the goal, False otherwise
    '''
    if child == goal:
        return True
    return False

def get_terrain_cost( state_coord: Tuple[int, int], world: List[List[str]], costs: Dict[str, int]) -> int:
    '''
    `get_terrain_cost` returns the terrain costs given a coordinate. 
    Helper function used to calculate costs of any given state.
     
    **state_coord** Tuple[int, int]: `Tuple` containing the child state coordinates
    **world** List[List[str]]: the actual context for the navigation problem.
    **costs** Dict[str, int]: is a `Dict` of costs for each type of terrain in **world**.
     
    **costs[terrain]** int: the cost of traversing through the terrain
    '''
    terrain = world[state_coord[1]][state_coord[0]]
    
    return costs[terrain]

def uniform_cost( parent: Tuple[Tuple[int, int], int, int, Tuple[int, int]], state_coord: Tuple[int, int], world: List[List[str]], costs: Dict[str, int]) -> int:
    '''
    `uniform_cost` calculates the g_cost, or the cost to traverse from the start to the current state through each terrain value. 
    The g_cost is stored when calculated for each state, so this function can take the g_cost of the parent node and 
    add the current terrain cost to get the current total g_cost, thus eliminating unnecessary calculations. 
    As A* search is a combination of uniform cost search and a heuristic measure, this g_cost is derived from the uniform cost 
    search which aims to find the node on the frontier with the lowest total path cost. UCS is a breadth-first approach, 
    and therefore can be used in A* to garuntee completeness and optimality. 

    **parent** Tuple[Tuple[int, int], int, int, Tuple[int, int]]: parent state of the child being evaluated.
    **state_coord** Tuple[int, int]: `Tuple` containing the current state coordinates.
    **world** List[List[str]]: the actual context for the navigation problem.
    **costs** Dict[str, int]: is a `Dict` of costs for each type of terrain in **world**.
     
    **child_g_cost** int: the cost to traverse from the start to the current state through each terrain value
    '''
    parent_g_cost = parent[2]
    entry_cost = get_terrain_cost(state_coord, world, costs)
    
    child_g_cost = parent_g_cost + entry_cost 
    
    return child_g_cost

def path( explored: List[Tuple[Tuple[int, int], int, int, Tuple[int, int]]], state: Tuple[Tuple[int, int], int, int, Tuple[int, int]]) -> List[Tuple[int, int]]:
    '''
    `path` returns the optimal path over the terrain by working backwards through the explored stack. 
    Note, we are using the parent coordinates assigned to each state to walk backwards through the optimal path. 
    Path is computed until the parent of the current state is `not None` . The starting state is initialized with a parent of None. 
    This function is returned once the goal state has been reached or the frotnier is empty.
     
    **explored** List[Tuple[Tuple[int, int], int, int, Tuple[int, int]]]: `List` of explored states serving the functionality of a stack.
    **state** Tuple[Tuple[int, int], int, int, Tuple[int, int]]): `Tuple` containing state cooridnates, f_cost, g_cost, parent state.
     
    **optimal_path** List[Tuple[int, int]]: `List` of the optimal path through the grid
    '''
    parent = state[3]
    child = state[0]

    optimal_path = []

    while parent is not None:
        move = (child[0] - parent[0], child[1] - parent[1])
        optimal_path.insert(0, move)
        child = parent
        for state in explored:
            if parent == state[0]:
                parent = state[3]

    return optimal_path

def euclidean_heuristic( goal: Tuple[int, int], child: Tuple[int, int]) -> int:
    '''
    euclidean_heuristic calculates an admissible and consistent heuristic to support the f_cost calculation at each child node. 
    A* evaluates both costs from the start (g_cost) and estimated cost to the goal (h_cost). 
    h_cost is an estimate of that cost to the goal through a heuristic measure which allows A* serach to increase in 
    efficiency while also being complete. 
     
    **goal** Tuple[int, int]: the desired goal position for the bot, `(x, y)`.
    **child** Tuple[int, int]: Coordinates of state
     
    **child_h_cost** int: euclidean distance fromt the current node to the goal.
    '''

    child_h_cost = np.sqrt((child[0] - goal[0])**2 + (child[1] - goal[1])**2)
    
    return child_h_cost

def manhattan_heuristic( goal: Tuple[int, int], child) -> int:
    '''
    manhattan_heuristic calculates an admissible and consistent heuristic to support the f_cost calculation at each child node. 
    A* evaluates both costs from the start (g_cost) and estimated cost to the goal (h_cost). 
    h_cost is an estimate of that cost to the goal through a heuristic measure which allows A* serach to increase in 
    efficiency while also being complete. 
     
    **goal** Tuple[int, int]: the desired goal position for the bot, `(x, y)`.
    **child** Tuple[int, int]: Coordinates of state
     
    **child_h_cost** int: euclidean distance fromt the current node to the goal.
    '''
    child_h_cost = np.absolute(child[0] - goal[0]) + np.absolute(child[1] - goal[1])
    
    return child_h_cost

def create_state_tuple(current_state: Tuple[Tuple[int, int], int, int, Tuple[int, int]], child: Tuple[int, int], world: List[List[str]], goal: Tuple[int, int], costs: Dict[str, int], heuristic: callable) -> Tuple[Tuple[int, int], int, int,
Tuple[int, int]]:
    '''
    `create_state_tuple` creates a state_tuple containing state cooridnates, f_cost, g_cost, parent state. The function calls the specified uniform cost and heuristic functions to calculate the g_cost, h_cost and then the f_cost. The f_cost is stored to assist decision making throughout A* search. The g_cost is stored for the `uniform_cost` function which takes the g_cost of the parent node and adds the current terrain cost, thus eliminating unnecessary calculations. parent is stored for both the `uniform_cost` and `path` functions. **Uses**: [uniform_cost](#uniform_cost). **Used by**: [a_star_search](#a_star_search).
     
    **current_state** Tuple[Tuple[int, int], int, int, Tuple[int, int]]
    **child** Tuple[int, int]
    **world** List[List[str]]: the actual context for the navigation problem.
    **goal** Tuple[int, int]: the desired goal position for the bot, `(x, y)`.
    **costs** Dict[str, int]: is a `Dict` of costs for each type of terrain in **world**.
    **heuristic** Callable: is a heuristic function, $h(n)$. 
    
    **(child, child_f_cost, child_g_cost, current_state[0])** Tuple[Tuple[int, int], int, int,
    Tuple[int, int]]: `Tuple` containing state cooridnates, f_cost, g_cost, parent state.
    '''
    child_g_cost = uniform_cost(current_state, child, world, costs)
    child_h_cost = heuristic(goal, child)
    child_f_cost = child_g_cost + child_h_cost
            
    return (child, child_f_cost, child_g_cost, current_state[0])

def a_star_search( world: List[List[str]], start: Tuple[int, int], goal: Tuple[int, int], costs: Dict[str, int], moves: List[Tuple[int, int]], heuristic: Callable) -> List[Tuple[int, int]]:
    '''
    `a_star_search` is an optimal and complete search method that expands the fewest possible nodes to find the goal. 
    Code below creates a frontier and explored stack. All children states to be explored are stored on the frotnier, 
    while all explored states are stored in the explored stack. The search begins with the starting location and 
    based on a set of possible moves, finds all possible children nodes, or following states. Each child's cost is evaluated. 
    That cost is evaluated as the sum of the cost from the start to the current state and an estimated cost to the goal. 
    The estimated cost is calculated through a heuristic, like euclidean or manhattan distance. 
    All children not previously visited are added to the frontier. If the current child has been previously visited, 
    but the current child has a lower cost, it is added to the frontier and the other is removed from the frontier. 
    The search loops through the frontier, pulling the lowest costing node to explore next. This process is repeated until a 
    child state is at the goal or the frontier is empty.
     
    **world** List[List[str]]: the actual context for the navigation problem.
    **start** Tuple[int, int]: the starting location of the bot, `(x, y)`.
    **goal** Tuple[int, int]: the desired goal position for the bot, `(x, y)`.
    **costs** Dict[str, int]: is a `Dict` of costs for each type of terrain in **world**.
    **moves** List[Tuple[int, int]]: the legal movement model expressed in offsets in **world**.
    **heuristic** Callable: is a heuristic function, $h(n)$.
    
    **returns** List[Tuple[int, int]]: the offsets needed to get from start state to the goal as a `List`.
    '''
    frontier = []
    explored = []
    
    start_state = (start, heuristic(goal, start), get_terrain_cost(start, world, costs), None)
    push(frontier, start_state)

    while frontier:
        current_state = pop(frontier)
        push(explored, current_state)

        children = successors(current_state[0], moves, world)

        for child in children:
            child_state = create_state_tuple(current_state, child, world, goal, costs, heuristic)

            if is_goal(child, goal):
                push(explored, child_state)
                return path(explored, child_state)

            if (not is_child_in_stack(frontier, child)) and (not is_child_in_stack(explored, child)):
                push(frontier, child_state)
            elif (is_child_in_stack(frontier, child)) and (child_state[1] < get_existing_child_state_in_stack(frontier, child)[1]):
                frontier.remove(get_existing_child_state_in_stack(frontier, child))
                push(frontier, child_state)

    return path(explored, current_state)

def pretty_print_path(world: List[List[str]], path: List[Tuple[int, int]], start: Tuple[int, int], goal: Tuple[int, int], costs: Dict[str, int], print_flag: bool) -> int:
    '''
    `pretty_print_path` prints over a deepcopy of the world with arrows indicating movements. 
    It also calculates the cost of traversing through the terrain of said map.
     
    **world** List[List[str]]: the world (terrain map) for the path to be printed upon.
    **path** List[Tuple[int, int]]: the path from start to goal, in offsets.
    **start** Tuple[int, int]: the starting location for the path.
    **goal** Tuple[int, int]: the goal location for the path.
    **costs** Dict[str, int]: the costs for each action.
     
    **returns** int - The path cost.
    '''
    arrow_moves_dict = {(0, -1): '⏫', (1, 0): '⏩', (0, 1): '⏬', (-1, 0): '⏪'}

    current_state = start
    path_cost = 0

    world_deepcopy = copy.deepcopy(world)

    for move in path:
        terrain_cost = costs[world_deepcopy[current_state[1]][current_state[0]]]
        path_cost += terrain_cost

        world_deepcopy[current_state[1]][current_state[0]] = arrow_moves_dict[move]
        next_state = (current_state[0] + move[0], current_state[1] + move[1])
        current_state = next_state

    path_cost += get_terrain_cost(goal, world, costs)
    world_deepcopy[goal[1]][goal[0]] = '🎁'

    if print_flag:
        for row in world_deepcopy:
            st.write(''.join(str(terrain) for terrain in row))

    return path_cost 

def generate_world(row_number: int, column_number: int) -> List[List]:
    '''
    `generate_world` randomly generates a world to traverse based on user input dimensions.
     
    **row_number** int: User generated input, must be between arbitrary range: 2-25.
    **column_number** int: User generated input, must be between arbitrary range: 2-25.
     
    **random_world** List[List]: Randomly generated world. 
    '''
    terrains = ['🌾','🌲','🪨','🐊','🗻']
    random_world = []
    for _ in range(row_number):
        row = []
        for _ in range(column_number):
            random_terrain = random.choice(terrains)
            row.append(random_terrain)
        random_world.append(row)

    while random_world[0][0] == '🗻':
        random_world[0][0] = random.choice(terrains)

    while random_world[len(random_world) - 1][len(random_world[0]) - 1] == '🗻':
        random_world[len(random_world) - 1][len(random_world[0]) - 1] = random.choice(terrains)

    return random_world

def is_traversable(world: List[List[str]], start: Tuple[int, int], goal: Tuple[int, int], costs: Dict[str, int], moves: List[Tuple[int, int]], heuristic: Callable) -> bool:
    '''
    `is_traversable` determines if randomly generated world contains a traversable path with a terrain cost under 10,000. 
     
    **world** List[List[str]]: the actual context for the navigation problem.
    **start** Tuple[int, int]: the starting location of the bot, `(x, y)`.
    **goal** Tuple[int, int]: the desired goal position for the bot, `(x, y)`.
    **costs** Dict[str, int]: is a `Dict` of costs for each type of terrain in **world**.
    **moves** List[Tuple[int, int]]: the legal movement model expressed in offsets in **world**.
    **heuristic** Callable: is a heuristic function, $h(n)$.

    **Bool** bool: True or false if world is traversable.
    '''
    traversable_path = a_star_search(world, start, goal, costs, moves, heuristic)
    traversable_cost = pretty_print_path(world, traversable_path, start, goal, costs, False)

    if traversable_cost > 10000:
        return False
    
    return True

def generate_world_coordinates(world: List[List]) -> List[Tuple]:
    '''
    `is_traversable` determines if randomly generated world contains a traversable path with a terrain cost under 10,000. 
     
    **world** List[List[str]]: the actual context for the navigation problem.

    **world_coords** List[Tuple]: List of cooridnate options.
    '''
    world_coords = []

    for row in range(len(world[0])):
        for col in range(len(world)):
            world_coords.append((row, col))

    return world_coords

# Initialize some Variables    
world = full_world
costs = { '🌾': 1, '🌲': 3, '🪨': 5, '🐊': 7, '🗻': 10000} 
start = (0,0)
goal = (len(world[0]) - 1, len(world) - 1)
moves = [(0,-1), (1,0), (0,1), (-1,0)]
heuristic = euclidean_heuristic

# Streamlit Title
st.title("Module 4 Assignment - A* Search Deployment")
st.write("Author: Chris Puglisi")

st.write("JHU EN.605.645 Artificial Intelligence")

# Overview with text
st.header("Overview")
st.markdown(
    "Given a world consisting of a predetermined set of terrains, A* Search will calculate "
    "the shortest path between a given start and end goal."
)
st.markdown(f"Each terrain has a unique cost as follows: {costs}")
st.header("Select Hueristic Measure")

# Radio Button to choose heuristic type
heuristic_name = st.radio("Heuristic", ('Euclidean', 'Manhattan'))

# Determine heuristic used for calculation
if heuristic_name == 'Euclidean':
    heuristic = euclidean_heuristic   
else:
    heuristic = manhattan_heuristic

#Start world to traverse section
st.header("Generate World to Traverse")

#Generate user input for random world dimensions
generated_flag = False
row_number = st.number_input("Enter the number of rows of random world (between 2 and 25):", value=2, step=1, min_value=2, max_value=25)
column_number = st.number_input("Enter the number of columns of random world (between 2 and 25):", value=2, step=1, min_value=2, max_value=25)

#Buttons to generate world or reset to defualt.
generate_world_button = st.button("Generate Random World")
generate_default_button = st.button("Generate Default World")

# Create session state parameter and set to None.
if 'random_world' not in st.session_state:
    st.session_state.random_world = None

# Generate traversable world, save world in session state, display world.
if generate_world_button:
    random_world = generate_world(row_number, column_number)
    goal = (len(random_world[0]) - 1, len(random_world) - 1)
    while not is_traversable(random_world, start, goal, costs, moves, heuristic):
        random_world = generate_world(row_number, column_number)
    
    st.session_state.random_world = random_world
    generated_flag = True
    
    for row in random_world:
        st.write(''.join(str(terrain) for terrain in row))

# Generate default world
if generate_default_button:
    generated_flag = False
    st.session_state.random_world = None
    world = full_world

# Print default world
if st.session_state.random_world == None:
    st.write("A default world has been provided below:")
    for row in world:
        st.write(''.join(str(terrain) for terrain in row))

# Set world to random world if exists (needed as streamlit reruns the script with each parameter update)
if st.session_state.random_world != None:
    world = st.session_state.random_world

# Print current world
if st.session_state.random_world != None and not generated_flag:
    for row in world:
        st.write(''.join(str(terrain) for terrain in row))

#Shortest path section
st.header(f"Shortest Path: {heuristic_name}")

# Generate all possible start and end coordinates
if st.session_state.random_world == None:
    world_coords = generate_world_coordinates(world)
else:
    world = st.session_state.random_world
    world_coords = generate_world_coordinates(world)

# Display all possible coordinate options for user input.
start_coords = st.selectbox("Select start coordinates:", world_coords)
end_world_coords = list(reversed(world_coords))
end_world_coords.remove(start_coords)
end_coords = st.selectbox("Select end coordinates:", end_world_coords)

# Traverse world button
traverse_world_button = st.button("Traverse World")

# Print traversed world
if traverse_world_button:
    path = a_star_search(world, start_coords, end_coords, costs, moves, heuristic)
    small_path_cost = pretty_print_path(world, path, start_coords, end_coords, costs, True)
    st.write("Path Cost:", small_path_cost)
