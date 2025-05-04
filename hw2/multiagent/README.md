# Q1: Reflex Agent 개선
ReflexAgent의 평가 함수를 구현하여 팩맨이 유령을 피하면서 음식을 잘 먹도록 만듦

- 음식과 유령의 위치를 고려해야함.

# Q2: Minimax Agent 구현
팩맨과 유령들을 포함하는 multi-agent minimax 알고리즘 구현

- 팩맨은 MAX, 적은 MIN
- 트리의 depth는 팩맨과 적(유령)이 한 번씩 움직이는 단위를 기준으로 함

# Q3: Alpha-Beta Agent 구현
minimax 알고리즘에 알파-베타 가지치기 추가하여 탐색 효율화

- Pruning을 통해서 불필요한 상태 탐색을 방지함.
- 상태 순서 변경 없이 구현

# Q4: Expectimax Agent 구현
유령들이 최적이 아닌 랜덤하게 움직일 수 있다는 가정을 고려한 Expectimax 알고리즘 구현

- 적(유령)은 min이 아닌 기대값을 기반으로 행동함
- 모든 적(유령)은 uniform probability로 행동한다고 가정

# Q5: Better Evaluation Function 구현
이전까지 구현한 agent들이 사용할 수 있는 상태 기반 평가 함수를 설계

- 행동이 아닌 상태를 평가
- 음식, 유령, 캡슐 등 여러 요소를 고려
