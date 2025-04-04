# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def generalSearch(problem, frontier, heuristic=nullHeuristic):
    '''
    탐색 함수들이 중복됨.
    따라서 이를 해결하기 위해 하나의 일반화된 탐색을 구현하고, frontier를 달리하여
    중복 코드를 줄임
    '''
    start_point = problem.getStartState()
    costs = {start_point: 0}
    visited = set()
    initial_path = []

    # PriorityQueue와 아닐때로 나눠서 분리
    # PriorityQueue인 경우에는 heuristic 함수 사용
    if isinstance(frontier, util.PriorityQueue):
        frontier.push((start_point, initial_path), heuristic(start_point, problem))
    else :
        frontier.push((start_point, initial_path))

    # frontier가 빌 때까지 반복
    while not frontier.isEmpty():
        current_point, path = frontier.pop()
        
        # 종료 조건
        if problem.isGoalState(current_point):
            return path

        # PriorityQueue인 경우에는 costs를 사용하여 현재 cost를 찾음
        if isinstance(frontier, util.PriorityQueue):
            current_cost = costs[current_point]
        # 이외의 경우(BFS, DFS에서는 visited로 처리함)
        else :
            if current_point in visited:
                continue
            visited.add(current_point)
        
        # 자식 노드들을 확인
        for successor, action, next_cost in problem.getSuccessors(current_point):
            # 새로운 경로(누적): 지금까지 왔던 경로 + 바로 다음에 가야할 경로
            new_path = path + [action]

            # PriorityQueue로 구현되었다면
            if isinstance(frontier, util.PriorityQueue):
                # new_cost = 지금까지의 비용 + 바로 다음에 갈 수 있는 비용
                new_cost = current_cost + next_cost
                
                # 아직 방문을 하지 않았거나
                # 방문을 했어도, new_cost가 이미 방문했던 것보다 더 낮은 비용으로 갈 수 있을 때
                if successor not in costs or new_cost < costs[successor]:
                    costs[successor] = new_cost
                    prioirty = new_cost + heuristic(successor, problem)
                    frontier.push((successor, new_path), prioirty)
            else :
                if successor not in visited:
                    visited.add(current_point)
                    frontier.push((successor, new_path))

    

def depthFirstSearch(problem: SearchProblem):
    """
    DFS를 통해서 목적지에 도달한 actions의 list를 반환
    다음의 함수를 통해서 상태를 받아올 수 있습니다.

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """

    '''
    frontier(스택)에서 하나 꺼냄 pop()
    frontier에서 꺼낸 곳을 방문했는지 확인
    현재 위치가 목적지라면 반복을 종료시킴
    이미 방문했다면 패스
    아직 방문하지 않았다면, 방문 처리
    현재 위치에서 pac-man이 갈 수 있는 다음 위치(successors)를 가져옴
    다음 위치(successors)를 하나씩 방문했는지 확인
    방문하지 않았다면 frontier에 추가
    '''

    # generalSearch() 를 통해서 한 번에 사용할 수 있도록 함.
    return generalSearch(problem, util.Stack())

def breadthFirstSearch(problem: SearchProblem):
    return generalSearch(problem, util.Queue())


def uniformCostSearch(problem: SearchProblem):
    return generalSearch(problem, util.PriorityQueue())
  

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    '''
    만들어 놓은 generalSearch() 함수를 재사용 해야함.
    그러나 UCS 까지는 heuristic을 구현하지 않았기 때문에 heuristic 부분만 추가하면 됨.
    UCS -> Heuristic == 0
    '''
    return generalSearch(problem, util.PriorityQueue(), heuristic)

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
