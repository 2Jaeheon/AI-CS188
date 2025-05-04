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
            pacmanToGhostDistance = manhattanDistance(newPos, ghostPosition)

        
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
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

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
