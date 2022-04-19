# 수학과 통계를 정리합니다.

### Joint probability distribution

[Joint probability distribution](https://blog.daum.net/gongdjn/57)

Given two random variables that are defined on the same probability space,[1] the joint probability distribution is the corresponding probability distribution on all possible pairs of outputs. The joint distribution can just as well be considered for any given number of random variables. The joint distribution encodes the marginal distributions, i.e. the distributions of each of the individual random variables. It also encodes the conditional probability distributions, which deal with how the outputs of one random variable are distributed when given information on the outputs of the other random variable(s).

<img src="https://github.com/Hyeseong0317/Math_-probability-statistics/blob/main/images/Joint probability distribution.PNG" width="40%">

[KL divergence, Kullback-Leibler divergence, Jensen-Shannon divergence](https://hyunw.kim/blog/2017/10/27/KL_divergence.html)

Cross entropy, H(p,q)를 전개해보면 그 안에 이미 확률분포 p의 엔트로피가 들어있습니다. 그 H(p)에 무언가 더해진 것이 cross entropy입니다. 이때 이 무언가 더해지는 것이 바로 “정보량 차이”인데, 이 정보량 차이가 바로 KL-divergence입니다. 직관적으로 정리를 해보겠습니다. KL-divergence는 p와 q의 cross entropy에서 p의 엔트로피를 뺀 값입니다. 결과적으로 두 분포의 차이를 나타냅니다.

<img src="https://github.com/Hyeseong0317/Math_-probability-statistics/blob/main/images/crossentropy.PNG" width="40%">

쿨백-라이블러 발산(Kullback–Leibler divergence, KLD)은 두 확률분포의 차이를 계산하는 데에 사용하는 함수로, 어떤 이상적인 분포에 대해, 그 분포를 근사하는 다른 분포를 사용해 샘플링을 한다면 발생할 수 있는 정보 엔트로피 차이를 계산한다. 상대 엔트로피(relative entropy), 정보 획득량(information gain), 인포메이션 다이버전스(information divergence)라고도 한다.

[MCMC, 마코프체인, 몬테카를로](https://4four.us/article/2014/11/markov-chain-monte-carlo)

[Gibbs Sampling](https://ratsgo.github.io/statistics/2017/05/31/gibbs/)

[아핀변환, 선형변환, 비선형성](https://hooni-playground.com/1271/)

결국 선형변환은 아핀변환의 특수한 형태(bias가 영벡터)라는 것을 알 수 있다. 다시 말해 선형변환은 아핀변환에 포함된다.

[Affine Transformation](https://en.wikipedia.org/wiki/Affine_transformation)

#### 아핀변환은 선형변환과 달리 아핀공간에서 원점을 보존하지 않는다. 원점도 이동이 가능하다. 뉴럴네트워크에서 아핀변환을 사용하면 선형변환을 거친 것보다 매핑을 더 직관적으로 할 수 있다. 아핀변환은 선형변환에서 bias b vector가 추가된 것이다.
#### Unlike a purely linear transformation, an affine transformation need not preserve the origin of the affine space. Thus, every linear transformation is affine, but not every affine transformation is linear. Examples of affine transformations include translation, scaling, homothety, similarity, reflection, rotation, shear mapping, and compositions of them in any combination and sequence.

### 아핀변환 사용예시

StyleGAN --> W가 AdaIN을 통해 style을 입힐 때 shape이 안맞아 Affine Transformation을 거쳐서 shape을 맞춰줍니다.

### argmin, argmax 함수들의 의미
함수 f(x)를 최솟값으로 만들기 위한 x 값을 구한다 또는 함수 f(x)를 최댓값으로 만들기 위한 x 값을 구한다.

함수 f(x)가 무엇이냐에 따라 x 값이 달라지게 되며, 만족하는 값이 여러 개 일 수도 있다.

#### Cardinality

집합론에서, 집합의 크기 또는 농도는 집합의 "원소 개수"에 대한 척도이다. 유한 집합의 크기의 표현은 자연수로 충분하다. 임의의 집합의 크기는 단사 함수 및 전단사 함수를 통해 비교할 수 있으며, 기수로서 대상화할 수도 있다.

<img src="https://github.com/Hyeseong0317/Math_-probability-statistics/blob/main/images/cardinality.png" width="40%">

#### Marginal Likelihood
한계 우도 함수 또는 적분 우도는 일부 모수 변수가 소외된 우도 함수입니다. 베이지안 통계와 관련하여, 증거 또는 모형 증거라고도합니다.

<img src="https://github.com/Hyeseong0317/Math_-probability-statistics/blob/main/images/marginalliklihood.PNG" width="40%">

### MAP - Maximum a Posteriori 최대 사후 확률
어떤 모수 θ 의 사전 확률 분포가 p ( θ ) 로 주어져 있고, 그 모수에 기반한 조건부 확률분포 f ( x | θ ) 와 그 분포에서 수집된 값 x 가 주어져 있습니다. 이떄 모수의 사후 확률분포는 베이즈 정리에 의해 다음과 같이 계산할 수 있습니다.

<img src="https://github.com/Hyeseong0317/Math_-probability-statistics/blob/main/images/사후확률분포.PNG" width="20%">
<img src="https://github.com/Hyeseong0317/Math_-probability-statistics/blob/main/images/MAP.PNG" width="60%">

#### MAP의 장점
MAP는 설계자가 알고 있는 사전지식(사전확률)을 반영한다는 점에서 매우 합리적인 방법으로 여겨집니다. 단순히 관측결과에만 의존하는 것보다 기존의 알고 있는 정보를 반영한다는 점에서 실제 인간의 학습방식을 모사하기에 적합해 보입니다.
#### MAP의 단점
MAP는 베이즈추론을 기반으로 합니다. 관측결과 뿐만 아니라 사전지식(사전확률)을 활용하기 때문에 사전지식에 대한 모델링이 필요합니다. 문제는 사전지식에 대한 모델링이 어렵다는 것이고, 사전지식에 대한 모델링에 따라 추론결과인 사후확률(Posteriori)의 정확도가 크게 좌우됩니다. 사전지식이 양날의 검인 샘입니다.

### Probability Density Function
Probability density function (PDF) is a statistical expression that defines a probability distribution (the likelihood of an outcome) for a discrete random variable (e.g., a stock or ETF) as opposed to a continuous random variable.

<img src="https://github.com/Hyeseong0317/Math_-probability-statistics/blob/main/images/확률밀도함수.PNG" width="60%">
