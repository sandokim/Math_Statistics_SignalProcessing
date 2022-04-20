# 수학과 통계를 정리합니다.

### Joint probability distribution

[Joint probability distribution](https://blog.daum.net/gongdjn/57)

Given two random variables that are defined on the same probability space,[1] the joint probability distribution is the corresponding probability distribution on all possible pairs of outputs. The joint distribution can just as well be considered for any given number of random variables. The joint distribution encodes the marginal distributions, i.e. the distributions of each of the individual random variables. It also encodes the conditional probability distributions, which deal with how the outputs of one random variable are distributed when given information on the outputs of the other random variable(s).

<img src="https://github.com/Hyeseong0317/Math_-probability-statistics/blob/main/images/Joint probability distribution.PNG" width="40%">

[KL divergence, Kullback-Leibler divergence, Jensen-Shannon divergence](https://hyunw.kim/blog/2017/10/27/KL_divergence.html)

쿨백-라이블러 발산(Kullback–Leibler divergence, KLD)은 두 확률분포의 차이를 계산하는 데에 사용하는 함수로, 어떤 이상적인 분포에 대해, 그 분포를 근사하는 다른 분포를 사용해 샘플링을 한다면 발생할 수 있는 정보 엔트로피 차이를 계산한다. 상대 엔트로피(relative entropy), 정보 획득량(information gain), 인포메이션 다이버전스(information divergence)라고도 한다.

#### Cross entropy, H(p,q)를 전개해보면 그 안에 이미 확률분포 p의 엔트로피가 들어있습니다. 그 H(p)에 무언가 더해진 것이 cross entropy입니다. 이때 이 무언가 더해지는 것이 바로 “정보량 차이”인데, 이 정보량 차이가 바로 KL-divergence입니다. 직관적으로 정리를 해보겠습니다. KL-divergence는 p와 q의 cross entropy에서 p의 엔트로피를 뺀 값입니다. 결과적으로 두 분포의 차이를 나타냅니다.

<img src="https://github.com/Hyeseong0317/Math_-probability-statistics/blob/main/images/crossentropy.PNG" width="40%">

KL-divergence의 정확한 식은 이렇습니다. 대개 DKL(p|q) 또는 KL(p|q)로 표현합니다.

<img src="https://github.com/Hyeseong0317/Math_-probability-statistics/blob/main/images/KL-divergence.PNG" width="60%">

우리가 대개 cross entropy를 minimize 하는 것은, 어차피 H(p)는 고정된 상수값이기 때문에 결과적으로는 KL-divergence를 minimize 하는 것과 같습니다.

#### 수식이 이해가 안가면 직접 써보면 이해를 잘할 수 있습니다. (진리)

<img src="https://github.com/Hyeseong0317/Math_-probability-statistics/blob/main/images/KL.jpg" width="40%">

#### 모평균 p(x)를 샘플 데이터의 기댓값(=평균)으로 구하고 학습가능한 θ에 대해 모델을 학습시켜 parametric distribution q(x|θ)를 p(x)에 근사시킬 수 있다! 유레카!

KL-divergence와 log likelihood 우리가 전체를 알 수 없는 분포 p(x) 에서 추출되는 데이터를 우리가 모델링하고 싶다고 가정해보겠습니다. 우리는 이 분포에 대해 어떤 학습 가능한 parameter θ의 parametric distribution q(x|θ) 를 이용해 근사시킨다고 가정해보겠습니다. 이 θ 를 결정하는 방법 중 하나는 바로 p(x)와 q(x|θ) 사이의 KL-divergence 를 최소화시키는 θ 를 찾는 것입니다. 우리가 p(x) 자체를 모르기 때문에 이는 직접 해내기는 불가능합니다. 하지만 우리는 이 p(x)에서 추출된 몇 개의 샘플 데이터(training set)는 압니다(xn, for n=1,…,N). 따라서 p(x)에 대한 기댓값은 그 샘플들의 평균을 통해 구할 수 있습니다.

<img src="https://github.com/Hyeseong0317/Math_-probability-statistics/blob/main/images/KL-divergence와 log likelihood.PNG" width="50%">

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

y가 주어졌을 때, x의 확률을 표준화 하면 베이즈 이론을 이용해서 x와 y의 순서를 바꾸고, 이에 분모에 marginal을 이용해서 표준화를 한다.  marginalization은 x이외에 y를 조건으로 했을 때, 영향을 미치는 모든 z를 결합 확률을 구해서 연속의 경우에는 적분을 하고, 이산일 경우에는 합친다. 여기서 y를 조건으로 하는 것은 동일하다는점을 기억하자.  조건부 확률에서의 기대치는 그냥 변수 x의  기대치와 공식이 같으며, 다만 확률 밀도 함수를  입력에 해당하는 x에 대해서만 적용하고 조건에 해당하는 y에 대해서는 무시한다. 하지만 확률 밀도 함수는 조건부 확률 p(x|y)를 쓴다는 점을 유념하자. 위에서 얼핏 보기에 f(x)가 확률 밀도  함수 처럼 보이지만, 실제로는 오른쪽에 있는 p(x|y)가  확률 밀도 함수의 역할을 하며, f(x)는 구하고자 하는 값에 해당한다. 


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

### Cumulative Distribution Function
확률론에서, 누적 분포 함수(cumulative distribution function, 약자 cdf)는 주어진 확률 변수가 특정 값보다 작거나 같은 확률을 나타내는 함수이다.

CDF는 normal distribution이 축적됐다고 생각하면 쉽게 보인다.

#### Normal distribution, 정규분포

<img src="https://github.com/Hyeseong0317/Math_-probability-statistics/blob/main/images/normaldistribution.PNG" width="60%">

#### Cumulative distribution, 누적분포

<img src="https://github.com/Hyeseong0317/Math_-probability-statistics/blob/main/images/cdf.PNG" width="60%">

#### ill-posed problem
A problem which may have more than one solution, or in which the solutions depend discontinuously upon the initial data. Also known as improperly posed problem.
#### Well-posed problem
수리물리학 문제가 "해가 유일하게 존재하고 데이터에 연속적으로 의존적일때" 그 문제는 Well-posed 되었다고 한다. 이러한 조건을 만족하지 않을때 그 문제는 Ill-posed 되었다고 한다.

ex) ill-posed problem의 예시로 한 장의 저해상도 이미지에 대응할 수 있는 고해상도 이미지는 다양한 경우의 수가 있다는 예시로 이해하시면 쉽습니다.

[underdetermined inverse problem, ill-posed problem, Image Restoration(IR)](https://jaejunyoo.blogspot.com/2019/05/image-restoration-inverse-problem-1.html)

#### underdetermined
변수항을 좌변으로 이항했을때,

0x=00x=0 이면, 해가 무수히 많고

0x=a0x=a(단,a는 0이 아니다)면, 해가 없다.

방정식 중 해가 하나 혹은 유한 개로 정해지지 않는 방정식의 통칭. 부정방정식의 부정(不定)은 '정할 수 없다'의 뜻으로, 마치 0으로 나누기에서 등장하는 0x=00x=0의 해처럼 '해의 값을 하나로 정할 수 없는'의 의미이다.
