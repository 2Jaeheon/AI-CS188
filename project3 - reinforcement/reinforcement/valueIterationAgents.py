# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        # "*** YOUR CODE HERE ***"
        
        # Value Iteration의 시작점으로 Bellman Equation을 반복적으로 적용해서 각 상태의 value를 계산하는 것
        for i in range(self.iterations):
            # self.iterations가 V(k) 에서 k를 나타내는 것
            values = util.Counter();
            # values는 임시 저장소임 즉, 모든 state의 새 value들을 계산하고 최종 반영하는 형태로 진행함.

            # mdp에서 모든 상태를 가져와서 모든 상태를 순회
            for state in self.mdp.getStates():
                # 종료 조건
                if self.mdp.isTerminal(state):
                    continue

                # Q-Value를 꺼내서 가장 큰 것을 찾아냄
                bestQValue = float('-inf')
                # 하나의 state에서 가능한 Action들을 모두 꺼내서 해당 Action에 대해 Q-Value를 계산하고 가장 큰 value를 찾아냄.
                for action in self.mdp.getPossibleActions(state):
                    QValue = self.computeQValueFromValues(state, action)
                    bestQValue = max(bestQValue, QValue)

                # 가능한 action이 하나라도 존재하면 갱신해줌
                if bestQValue != float('-inf'):
                    values[state] = bestQValue

            # 모든 state에 대해서 갱신 (하나의 iteration이 끝나고 모든 state에 대한 value가 저장되었을 때 update 해줌)
            self.values = values


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        # "*** YOUR CODE HERE ***"
        # Q-Value를 구하는 과정임
        # runValueIteration() 함수에서 호출해서 사용하는 용도
        # 각 runValueIteration() 함수에서 하나의 state에서 
        # 가능한 모든 action에 관해 computeQValueFromValues() 함수를 호출함.

        QValue = 0
        # 이 행동을 했을 때 나올 수 있는 모든 pair (즉, state, action 쌍으로 구성된 리스트)
        transactions = self.mdp.getTransitionStatesAndProbs(state, action)

        # 모든 state-action pair 중 가장 Value가 큰 것을 반환해주면 된다.
        for next, probability in transactions:
            # 다음으로 갔을 때의 보상
            reward = self.mdp.getReward(state, action, next)

            # 다음 상태의 value 값
            nextValue = self.values[next]

            # Reward + 가치 + 감가율 (Q-value를 구하는 것)
            QValue += probability * (reward + self.discount * nextValue)

        return QValue

        

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        # "*** YOUR CODE HERE ***"
        # 여기서는 가장 최적의 정책을 반환해줘야 함. 즉, policy를 반환해주는 것.
        # 이게 결국 원하는 것! 책에서는 pi*를 찾는 것이 목표였음.
        # runValueIteration() 함수에서 갱신한 V_k(s)에서 각 Action별 value를 알고있음.
        # 따라서 이제 action별로 가장 큰 Q가 있는 Action을 전달해줘야 함. argmax임.

        if self.mdp.isTerminal(state):
            return None

        bestAction = None
        bestQValue = float('-inf')
        
        # state에 각 해당하는 action들에 대해서 반복해줌
        for action in self.mdp.getPossibleActions(state):
            # Q-Value를 구해서 가장 큰 Qvalue값으로 가는 aciton을 알려줘야 함.
            QValue = self.computeQValueFromValues(state, action)
            # bestAction을 계속해서 갱신해서 반환해야함.
            if QValue > bestQValue:
                bestQValue = QValue
                bestAction = action

        # V(k)에서 action을 했을 때 Q-Value를 구해서 가장 좋은 action을 반환함.
        return bestAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class PrioritizedSweepingValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        # "*** YOUR CODE HERE ***"
        # Value Iteration에서 조금 수정된 형태
        # 당장 갱신할 필요가 있는 거 먼저 갱신하자

        # 어떤 특정 상태 s의 이전 상태들을 저장
        predecessors = collections.defaultdict(set)

        # 반복해서 next State 입장에서 nextState가 될 수 있는 이전의 상태를 저장함.
        for state in self.mdp.getStates():
            for action in self.mdp.getPossibleActions(state):
                # 행동을 했을 때의 가능한 다음 상태와 확률들을 가지고 옴
                for nextStep, probability in self.mdp.getTransitionStatesAndProbs(state, action):
                    # 다음 상태에서 predecessor로 현재 상태를 추가
                    if probability > 0:
                        predecessors[nextStep].add(state)

        # 우선순위 큐 생성
        # 우선순위가 낮은 것들(변화가 클 것으로 예측되는 것)을 먼저 꺼내도록 해야함.
        priorityQueue = util.PriorityQueue();

        # 각 상태의 Q-Value중 가장 큰 것을 계산하자
        for state in self.mdp.getStates():
            if self.mdp.isTerminal(state):
                continue

            actions = self.mdp.getPossibleActions(state)
            if not actions:
                continue
            
            maxQValue = float('-inf')

            # 가능한 모든 action에 관한 Q-Value를 계산
            for action in actions:
                Q = self.computeQValueFromValues(state, action)
                if Q > maxQValue:
                    maxQValue = Q

            # Q-Value를 계산해서 기존의 값과 차이를 계산한 다음 이 차이를 PriorityQueue에 넣음
            diff = abs(self.values[state] - maxQValue)
            priorityQueue.update(state, -diff)

        # 주어진 반복 횟수만큼 값을 갱신 (몇 번 반복할건지가 들어있음.)
        for i in range(self.iterations):
            # 종료조건
            if priorityQueue.isEmpty():
                break;

            # priorityQueue에서는 가장 변화가 클 것으로 예측되는 상태를 pop함.
            state = priorityQueue.pop()

            # 다시 또 꺼낸 state에서 QValue를 다시 게산하고 update 시킴
            if not self.mdp.isTerminal(state):
                actions = self.mdp.getPossibleActions(state)
                bestQValue = float('-inf')

                for action in actions:
                    Q = self.computeQValueFromValues(state, action)
                    if Q > bestQValue:
                        bestQValue = Q

                self.values[state] = bestQValue

            # predecessor를 순회해서 Queue에 넣음
            # 이전 상태들을 업데이트 해줘야함.
            for predecessor in predecessors:
                # 종료 조건
                if self.mdp.isTerminal(predecessor):
                    continue
                
                actions = self.mdp.getPossibleActions(predecessor)

                if not actions:
                    continue

                bestQValue = float('-inf')

                for action in actions:
                    Q = self.computeQValueFromValues(predecessor, action)
                    if Q > bestQValue:
                        bestQValue = Q

                # 다시 가장 큰 Q-Value를 구하고 theta보다 크면 priorityQueue에 넣음
                # theta보다 작으면 변화가 거의 없는 것으로 간주하기 때문에 넣지 않아도 된다.
                diff = abs(self.values[predecessor] - bestQValue)
                if diff > self.theta:
                    priorityQueue.update(predecessor, -diff)
            