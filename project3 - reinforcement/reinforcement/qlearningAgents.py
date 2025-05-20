# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        # (state, action) 쌍을 저장할 딕셔너리 구조가 존재해야함.
        self.QValues = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        # state, action 한 쌍을 반환해줌.
        # self.QValues에서 값을 가지고 오며, Counter의 특성상 Q가 없으면 0을 반환
        return self.QValues[(state, action)]


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        
        # state에서 가능한 action들 중 가장 큰 Q-value를 반환함.
        # 즉, max Q(s, a)를 계산하는 부분임
        legalActions = self.getLegalActions(state)

        # 만약 가능한 action이 없으면 0.0
        if not legalActions:
          return 0.0

        bestQValue = float('-inf')

        # state에서 모든 가능한 action에 대해서 가장 큰 value를 반환함.
        for action in legalActions:
          QValue = self.getQValue(state, action)
          if QValue > bestQValue:
            bestQValue = QValue

        return bestQValue


    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        # 현재 상태에서 Qvalue가 큰 최적의 행동을 구함.
        # 만약 여러 action이 동일한 경우에는 랜덤하게 뽑음
        legalActions = self.getLegalActions(state)

        if not legalActions:
          return None

        bestQValue = float('-inf')
        bestActions = []
        
        # 가능한 action들 중 가장 큰 Q-value를 모두 bestActions에 저장.
        for action in legalActions:
          QValue = self.getQValue(state, action)

          if QValue > bestQValue:
            bestQValue = QValue
            bestActions = [action]

          elif QValue == bestQValue:
            bestActions.append(action)

        # 저정한 bestAction들 중에서 하나를 랜덤으로 택해서 반환함.
        return random.choice(bestActions)


    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        
        legalActions = self.getLegalActions(state)
      
        if not legalActions:
          return None

        # 여기서 flipCoin은 self.epsilon의 확률로 true를 반환해주는 함수임.
        if util.flipCoin(self.epsilon):
          # 이렇게 epsilon의 확률인 경우는 하나를 랜덤하게 선택
          # 즉 여기서는 탐험을 진행함.
          return random.choice(legalActions)
        else :
          # 여기서는 탐색을 진행 action을 계산함.
          return self.computeActionFromQValues(state)
        
    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        currentQ = self.getQValue(state, action)
        nextQ = self.computeValueFromQValues(nextState)
        
        sample = reward + self.discount * nextQ
        self.QValues[(state, action)] = (1 - self.alpha) * currentQ + self.alpha * sample

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        # (state, action)에 관한 feature들을 추출해서
        # Q-value를 구해서 반환해줘야 함.
        # feature 추출 메서드는 featExtractors 파일에 구현되어 있음.
        features = self.featExtractor.getFeatures(state, action)

        total = 0
        # features들을 순회하며 sum(f(i))를 구해야함.
        for feature in features:
          value = features[feature]
          weight = self.weights[feature]
          total += weight * value
        
        return total

    def update(self, state, action, nextState, reward):
        """

           Should update your weights based on transition
        """
        # 가중치를 업데이트 하는 부분
        # difference = r + gamma * max(Q(s', a') - Q(s, a))
        # w <- w + alpha * difference * fi(s, a)

        currentQ = self.getQValue(state, action)

        legalActions = self.getLegalActions(nextState)

        # nextState에서의 최대 Q-value 계산
        if not legalActions:
          nextQ = 0.0
        else :
          nextQ = float('-inf')

          # 여기서 그냥 action으로 하면 에러 발생함.
          # 파라미터 action이 갱신 되어버림
          for nextAction in legalActions:
            Q = self.getQValue(nextState, nextAction)
            if Q > nextQ:
              nextQ = Q

        # TD error을 계산해야함
        target = reward + self.discount * nextQ
        difference = target - currentQ

        # 이제 특징을 추출하고 가중치를 계속 update 시켜주면 됨.
        features = self.featExtractor.getFeatures(state, action)

        for feature in features:
          value = features[feature]

          # 여기가 실제로 가중치를 update 하는 부분
          self.weights[feature] = self.weights[feature] + self.alpha * difference * value
        

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
          # 디버깅을 위한 부분임. 일단 구현할 필요 없음
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
