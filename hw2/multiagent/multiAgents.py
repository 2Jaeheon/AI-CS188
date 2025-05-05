# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        # successorGameState - 상태를 제공, 이거로부터 필요한 정보들을 추출함.
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        # "*** YOUR CODE HERE ***"
        # Best Action일수록 Score가 높은 Evaluation Function을 작성하는 것이 목표
        
        # 평가 기준
        # - 팩맨과 음식과의 거리 (가까울수록 좋음)
        # - 팩맨과 유령과의 거리 (멀수록 좋음)
        # - 팩맨과 캡슐과의 거리 (가까울수록 좋음)
        # - 팩맨과 겁먹은 유령(캡슐 섭취시) 거리 (가까울수록 좋음)
        # - 음식의 개수 (적을수록 좋음)
        # - 팩맨의 정지
        
        # 1. 팩맨 - 음식 간 거리
        pacmanToFoodDistance = float('inf')
        newFoodList = newFood.asList()
        n = len(newFoodList)
        for i in range(n):
            distance = manhattanDistance(newPos, newFoodList[i])
            if distance < pacmanToFoodDistance:
                pacmanToFoodDistance = distance

        # 2. 팩맨 - 유령 간 거리
        pacmanToGhostDistance = float('inf')
        ghostNum = len(newGhostStates)
        # 유령이 여러마리가 존재할 수 있으니까 반복문 사용
        for i in range(ghostNum):
            currentGhost = newGhostStates[i]
            ghostPosition = currentGhost.getPosition()
            distance = manhattanDistance(newPos, ghostPosition)
            # 바로 manhattanDisatnce로 distance를 갱신하면 모든 유령의 거리가 고려되지 않음. 
            # 여러마리를 고려해서 조건분기로 수정
            if distance < pacmanToGhostDistance:
                pacmanToGhostDistance = distance

        
        # 3. 팩맨 - 캡슐 간 거리
        pacmanToCapsuleDistance = float('inf')
        capsules = successorGameState.getCapsules()
        if len(capsules) > 0:
            for capsulePosition in capsules:
                distance = manhattanDistance(newPos, capsulePosition)

                if(distance < pacmanToCapsuleDistance):
                    pacmanToCapsuleDistance = distance
        
        # 4. scared Ghost 간의 거리
        pacmanToScaredGhostDistance = float('inf')

        for ghost in newGhostStates:
            if ghost.scaredTimer > 0:
                ghostPosition = ghost.getPosition()
                distance = manhattanDistance(newPos, ghostPosition)
                
                if distance < pacmanToScaredGhostDistance:
                    pacmanToScaredGhostDistance = distance

        # 5. 음식 개수
        foodCount = len(newFoodList)


        # 점수 계산
        score = 0

        if pacmanToFoodDistance != float('inf'):
            score += 10 / (pacmanToFoodDistance + 1)

        if pacmanToGhostDistance <= 1:
            score -= 500
        else :
            score -= 3 / (pacmanToGhostDistance + 1)

        if pacmanToScaredGhostDistance != float('inf'):
            score += 50 / (pacmanToScaredGhostDistance + 1)

        if pacmanToCapsuleDistance != float('inf'):
            score += 5 / (pacmanToCapsuleDistance + 1)

        # 멈추는 현상이 발생해서 멈추지 않도록 score를 감소
        if action == Directions.STOP:
            score -= 20

        score -= 5 * foodCount

        return score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """

        # "*** YOUR CODE HERE ***"
        # Helper를 사용해서 재귀적으로 구현 -> 재귀함수를 호출해서 해결하는 형태로 구현

        # 동작 과정
        # 1. Pacman은 초기에 모든 Actions을 확인하고 Action 하나를 선택
        # 2. Ghost는 자신의 턴에 Pacman에게 최대한 안 좋은 방향으로 선택
        # 3. 위 두 과정을 깊이의 제한동안 반복 or 종료까지 반복
        # 4. 상태를 점수화해서 위로 전파
        # 5. 팩맨은 가장 높은 점수를 기바으로 Action을 선택

        bestScore = float('-inf')
        bestAction = None

        # STOP은 리스트에서 삭제하려고 하였으나, 채점결과 정상적으로 점수가 나오지 않음
        # legalActions = gameState.getLegalActions(0)
        # actions = []
        # for action in legalActions:
        #     if action != Directions.STOP:
        #         actions.append(action)

        for action in gameState.getLegalActions(0):
            # 여기서 한 번 움직이게 되는 것
            successor = gameState.generateSuccessor(0, action)
            # 따라서 agent 를 1부터 시작해야 정상적으로 작동함.
            score = self.minimax(successor, agent = 1, depth = 0)

            if (score >bestScore):
                bestScore = score
                bestAction = action
                
        return bestAction

    def minimax(self, state, agent, depth):
        # 종료 조건일 시에는 상태에 관한 평가를 해서 점수를 반환해줌.
        if depth == self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        # 문어발처럼 쭉 뻗어나가야함.
        # 따라서 maxEvalute와 minEvaluate에서 계속해서 minimax를 수행하고
        # 팩맨과 Ghost의 각 상태에서(각 입장에서) 가장 최선의 선택을 하도록 함.
        # 여기서 종료조건이 만나면 바로 종료해서 getAction으로 전달

        # 현재 에이전트가 팩맨인 경우에는 maxEvaluate() 호출
        if self.isPacman(agent):
            return self.maxEvaluate(state, depth)
        
        # 현재 에이전트가 Ghost인 경우에는 minEvaluate() 호출
        elif self.isGhost(agent):
            return self.minEvaluate(state, agent, depth)

    def maxEvaluate(self, state, depth):
        # 팩맨 차례: 가능한 모든 Action 중 가장 높은 점수를 택해야함.
        # max값을 계산하는 함수임
        bestScore = float('-inf')

        for action in state.getLegalActions(0):
            successor = state.generateSuccessor(0, action)
            # 모든 action에 대해서 minimax로 재귀를 돌린 뒤
            # 가장 높은 점수를 반환해줌
            score = self.minimax(successor, agent = 1, depth = depth)
            # 팩맨은 가장 최적의 수로 움직여야 함
            bestScore = max(bestScore, score)

        return bestScore 

    def minEvaluate(self, state, agent, depth):
        # 유령 차례: 모든 행동중 팩맨에게 가장 불리한 action을 선택
        bestScore = float('inf')
        nextAgent = agent + 1
        nextDepth = depth

        # 모든 유령이 움직였을 때 다시 팩맨 차례가 되어야함
        if nextAgent == state.getNumAgents():
            nextAgent = 0
            nextDepth += 1 # 다음 depth로 증가

        for action in state.getLegalActions(agent):
            # 각 action에 관해 새로운 상태 생성
            successor = state.generateSuccessor(agent, action)
            # 갈 수 있는 모든 방향에 대해서 minimax 수행
            score = self.minimax(successor, nextAgent, nextDepth)
            # Ghost는 Pacman에게 가장 최악의 수로 움직여야함.
            bestScore = min(bestScore, score)

        return bestScore

    def isPacman(self, agent):
        if agent == 0:
            return True
        else :
            return False

    def isGhost(self, agent):
        if agent != 0:
            return True
        else :
            return False

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        # "*** YOUR CODE HERE ***"
        # 알파-베타 pruning -> 위의 minimax와 유사한 형태
        # 하지만, maxEvaluate() 함수에서 beta보다 커지면 pruning 해야함
        # 또한, minEvaluate() 함수에서 alpha보다 작아지면 pruning 해야함
        # alpha: 팩맨이 확보한 최고점수
        # beta: ghost가 확보한 최고점수

        # 여기서 구현해야 하는 것은 결국 더 이상 노드를 탐색을 해도 best 값보다 
        # 더 나은 결과가 나올 수 없다면 그 아래는 탐색하지 않고 pruning 하는 것

        bestScore = float('-inf')
        bestAction = None
        alpha = float('-inf')
        beta = float('inf')

        for action in gameState.getLegalActions(0):
            # 여기서 한 번 움직이게 되는 것
            successor = gameState.generateSuccessor(0, action)
            # 따라서 agent 를 1부터 시작해야 정상적으로 작동함.
            score = self.minimaxAlphaBetaPruning(successor, agent = 1, depth = 0, alpha = alpha, beta = beta)

            if (score >bestScore):
                bestScore = score
                bestAction = action
            
            # getAction은 팩맨의 root에서 선택을 결정하는 함수
            # 여기서는 가장 높은 값을 반환해줘야함.
            # 행동할 수 있는 모든 action들에 대해서 alpha 값을 갱신을 해주어야
            # 정확한 pruning이 가능함.
            alpha = max(alpha, bestScore)
            # ex) A, B, C의 행동을 하는 것이 가능함.
            # A의 bestScore 값이 10이었음.
            # alpha값을 10으로 갱신하고 다음 action인 B에 관해서 alpha가 10일 때를 기준으로 pruning이 진행되어야 함.

        return bestAction

    def minimaxAlphaBetaPruning(self, state, agent, depth, alpha, beta):
        # 종료 조건일 시에는 상태에 관한 평가를 해서 점수를 반환해줌.
        if depth == self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        # 문어발처럼 쭉 뻗어나가야함.
        # 따라서 maxEvalute와 minEvaluate에서 계속해서 minimax를 수행하고
        # 팩맨과 Ghost의 각 상태에서(각 입장에서) 가장 최선의 선택을 하도록 함.
        # 여기서 종료조건이 만나면 바로 종료해서 getAction으로 전달

        # 현재 에이전트가 팩맨인 경우에는 maxEvaluate() 호출
        if self.isPacman(agent):
            return self.maxEvaluate(state, depth, alpha, beta)
        
        # 현재 에이전트가 Ghost인 경우에는 minEvaluate() 호출
        elif self.isGhost(agent):
            return self.minEvaluate(state, agent, depth, alpha, beta)

    def maxEvaluate(self, state, depth, alpha, beta):
        # 팩맨 차례: 가능한 모든 Action 중 가장 높은 점수를 택해야함.
        # max값을 계산하는 함수임
        bestScore = float('-inf')

        for action in state.getLegalActions(0):
            successor = state.generateSuccessor(0, action)
            # 모든 action에 대해서 minimax로 재귀를 돌린 뒤
            # 가장 높은 점수를 반환해줌
            score = self.minimaxAlphaBetaPruning(successor, agent = 1, depth = depth, alpha = alpha, beta = beta)

            # 알파베타 프루닝을 시도해야함 (beta보다 커질 때)
            bestScore = max(bestScore, score)
            if bestScore > beta: # 여기서 pruning이 진행됨.
            # bestScore가 beta(유령의 최소값)보다 크다면 그냥 바로 탐색을 안 해도 된다는 뜻
                return bestScore
                
            alpha = max(alpha, bestScore)

        return bestScore 

    def minEvaluate(self, state, agent, depth, alpha, beta):
        # 유령 차례: 모든 행동중 팩맨에게 가장 불리한 action을 선택
        bestScore = float('inf')
        nextAgent = agent + 1
        nextDepth = depth

        # 모든 유령이 움직였을 때 다시 팩맨 차례가 되어야함
        if nextAgent == state.getNumAgents():
            nextAgent = 0
            nextDepth += 1 # 다음 depth로 증가

        for action in state.getLegalActions(agent):
            # 각 action에 관해 새로운 상태 생성
            successor = state.generateSuccessor(agent, action)
            # 갈 수 있는 모든 방향에 대해서 minimax 수행
            score = self.minimaxAlphaBetaPruning(successor, nextAgent, nextDepth, alpha, beta)
            # Ghost는 Pacman에게 가장 최악의 수로 움직여야함.
            bestScore = min(bestScore, score)

            # 팩맨은 최대값을 원하기 때문에 ghost가 주는 최악의 점수 bestScore가 alpha값보다 작다면
            # 어차피 팩맨은 아래의 가지들을 모두 탐색하지 않아도 됨
            if bestScore < alpha:
                return bestScore

            beta = min(beta, bestScore)

        return bestScore

    def isPacman(self, agent):
        if agent == 0:
            return True
        else :
            return False

    def isGhost(self, agent):
        if agent != 0:
            return True
        else :
            return False

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
