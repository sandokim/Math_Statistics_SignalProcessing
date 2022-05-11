# 수학, 통계, 신호처리를 정리합니다.

#### Logistic Regression -> Maximum liklihood value가 가장 큰 line을 찾습니다. <-> Mean Squre Error는 R^2을 줄이는 방향으로 line을 regression을 합니다.

<img src="https://github.com/Hyeseong0317/Math_-probability-statistics/blob/main/images/logistic regression.PNG" width="60%">

[변분추론(Variational Inference), 몬테카를로방법, KL Divergence](https://ratsgo.github.io/generative%20model/2017/12/19/vi/)

역전파(Backpropagation)는 먼저 계산 결과와 정답의 오차를 구해 이 오차에 관여하는 값들의 가증치를 수정하여 오차가 작아지는 방향으로 일정 횟수를 반복해 수정하는 방법이다. 오차역전파 또는 오류역전파라고도 부릅니다. 반대말은 순전파입니다.

#### 논문 약어

s.t = subject to, ~에 대하여

w.r.t = with respect to, ~에 대해

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

[Marcov Chain Stationary Distribution 유튜브영상](https://www.youtube.com/watch?v=4sXiCxZDrTU)

MCMC 방식은 High dimensional vector space에서 잘 되기 어렵다고 한다. 사실 직관적으로도 그렇습니다. MCMC를 이용하여 probability distribution를 찾는다고 생각해보면 점을 여러 개?를 뽑아서 distribution을 유추하겠다는 것인데 사실 거의 불가능에 가깝다. (고차원으로 갈 수록 천문학적인 점들이 필요할테니...).

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

<img src="https://github.com/Hyeseong0317/Math_-probability-statistics/blob/main/images/normaldistribution.PNG" width="30%">

#### Cumulative distribution, 누적분포

<img src="https://github.com/Hyeseong0317/Math_-probability-statistics/blob/main/images/cdf.PNG" width="30%">

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

#### [이항분포](https://ko.wikipedia.org/wiki/%EC%9D%B4%ED%95%AD_%EB%B6%84%ED%8F%AC)
이항 분포(二項分布)는 연속된 n번의 독립적 시행에서 각 시행이 확률 p를 가질 때의 이산 확률 분포이다. 이러한 시행은 베르누이 시행이라고 불리기도 한다. 사실, n=1일 때 이항 분포는 베르누이 분포이다.

<img src="https://github.com/Hyeseong0317/Math_-probability-statistics/blob/main/images/이항분포.PNG" width="30%">

#### [베르누이분포](https://ko.wikipedia.org/wiki/%EB%B2%A0%EB%A5%B4%EB%88%84%EC%9D%B4_%EB%B6%84%ED%8F%AC)
베르누이 분포(Bernoulli Distribution)는 확률 이론 및 통계학에서 주로 사용되는 이론으로, 스위스의 수학자 야코프 베르누이의 이름에 따라 명명되었다. 베르누이 분포는 확률론과 통계학에서 매 시행마다 오직 두 가지의 가능한 결과만 일어난다고 할 때, 이러한 실험을 1회 시행하여 일어난 두 가지 결과에 의해 그 값이 각각 0과 1로 결정되는 확률변수 X에 대해서

<img src="https://github.com/Hyeseong0317/Math_-probability-statistics/blob/main/images/베르누이분포.PNG" width="60%">

를 만족하는 확률변수 X가 따르는 확률분포를 의미하며, 이항 분포의 특수한 사례에 속한다.

### 특이값 분해 (SVD, Singular Value Decomposition)의 활용 -> 이미지 압축 (Image Compression)

특이값 분해를 설명할 때 가장 대표적인 예시로 드는 것은 바로 이미지 압축 (Image Compression)입니다. 흑백 이미지의 경우, 픽셀의 정보들을 행렬로 나타낼 수 있습니다. 그래서 위에서 언급했던 Low-rank Approximation을 활용해서, 흑백 이미지를 압축할 수 있습니다. 컬러 이미지의 경우, R, G, B에 해당하는 값들을 모아 각각 행렬을 만든 뒤, 각각을 approximation하면 이미지를 압축할 수 있습니다.

<img src="https://github.com/Hyeseong0317/Math_-probability-statistics/blob/main/images/고양이특이값분해.PNG" width="60%">

왼쪽 위의 사진이 고양이를 촬영한 원본 사진이고, 나머지 세 사진은 각각 rank를 200, 50, 10으로 제약을 두었을 때의 best low-rank approximation입니다. rank를 10으로 두었을 때는 형태가 제대로 나타나지 않지만, rank가 50일 때에는 고양이의 형태가 흐릿하게 보이고, rank가 200일 때는 원본과 큰 차이가 나지 않는 모습을 볼 수 있습니다.

### Low-rank approximation

In mathematics, low-rank approximation is a minimization problem, in which the cost function measures the fit between a given matrix (the data) and an approximating matrix (the optimization variable), subject to a constraint that the approximating matrix has reduced rank. The problem is used for mathematical modeling and data compression. The rank constraint is related to a constraint on the complexity of a model that fits the data. In applications, often there are other constraints on the approximating matrix apart from the rank constraint, e.g., non-negativity and Hankel structure.

수학에서 낮은 순위 근사는 근사 행렬의 순위가 감소했다는 제약 조건에 따라 비용 함수가 주어진 행렬과 근사 행렬 사이의 적합성을 측정하는 최소화 문제입니다. 문제는 수학적 모델링 및 데이터 압축에 사용됩니다. -> 빈 행렬 원소값에 대해 추정을 하는 것으로 생각, 평점을 원소로 하는 행렬에서 모든 항목이 다 채워져있기를 기대하기는 힘듭니다. (모든 유저가 모든 아이템에 대해서 평점을 매겨야 하기 때문입니다.) 이러한 경우, 아직 유저가 아직 평가하지 않은 아이템의 평점을 예측하려면 어떻게 해야 할까요? 먼저 평가가 매겨지지 않은 위치에 값을 임의로 채우고 특이값 분해를 합니다. 그 후 rank를 제한을 두어서 low-rank approximation을 수행합니다. 그렇다면, low-rank approximation을 한 결과가 아직 평가하지 않은 항목에 대한 평점의 예측이 됩니다.

[Low-rank approximation 평점시스템 예시](https://www.secmem.org/blog/2019/06/15/matrix-decomposition/)

[Low-rank approximation Image Restoration 예시](https://jaejunyoo.blogspot.com/2019/05/image-restoration-inverse-problem-1.html)

<img src="https://github.com/Hyeseong0317/Math_-probability-statistics/blob/main/images/lowrankapproximationIR.PNG" width="60%">

#### Low-rank approximation is closely related to:

-principal component analysis,

-factor analysis

-total least squares

-latent semantic analysis

-orthogonal regression

-dynamic mode decomposition

[Deeplearning Based Image Restoration 포스팅](https://jaejunyoo.blogspot.com/2019/05/image-restoration-inverse-problem-2.html)

#### Mapping function F, parameter θ 이해

<img src="https://github.com/Hyeseong0317/Math_-probability-statistics/blob/main/images/deeplearningbased.PNG" width="60%">

GAN 수식 이해 -> 파라미터 세타로 표현되는 z

<img src="https://github.com/Hyeseong0317/Math_-probability-statistics/blob/main/images/파라미터세타.jpg" width="60%">

### Model-based optimization VS CNN based discriminative learning-based(Deep learning based) 

-> [deep image prior (CVPR 2018)](https://arxiv.org/pdf/1711.10925.pdf) 

-> DIP의 재미있는 점은 이제 CNN이 이미지 한 장 한 장에 대해 optimization 문제를 풀어 θ를 구한다는 것입니다. 마치 model-based optimization 방식들 같죠? 따라서 다른 model-based optimization이 그러하듯 DIP도 unsupervised 방식으로 IR 문제를 풀 수 있게 됩니다.

----
#### Model-based optimization은 여러가지 degradation에 대해 사용자가 유연하게 forward 모델을 만들어 사용될 수 있습니다. 
즉, image prior가 주어지면 H만 바꿔가며 같은 알고리즘으로 여러 IR 문제들을 풀 수 있습니다. 단점은 각 instance마다 새로운 optimization 문제를 풀어줘야하기 때문에 느리겠죠.

----
#### 반면에 CNN을 이용한 discriminative learning-based 방식은 그 특성상 parametric mapping function Fθ(⋅)이 학습 데이터와 매우 강하게 엮여 있습니다. 
때문에 Image prior 자체를 데이터로부터 배우면서 좀 더 강력한 representation이 가능하므로 더 좋은 성능을 보이며, optimization에 드는 cost를 training phase로 넘길 수 있고 test phase에서의 inference가 빠릅니다. 
#### 그러나 데이터에 의존적인 면이 있으며 하나의 degradation에 대해서만 적용이 가능하고 따라서 모델의 유연성이 떨어진다는 단점이 있습니다.

----
### 신호처리

[신호처리개요](https://jaejunyoo.blogspot.com/2019/05/signal-processing-for-communications.html)

#### 무한대와 무한소
무한대 -> 계속 증가하는 상태

무한소 -> 계속 감소하는 상태, 0은 아니고 0에 가까워지는 상태 -> 무한소는 0이 아니라 0에 가까워지는 상태이므로 극한에서 분모와 분자의 나눗셈이 가능하다.

#### 신호처리 3대 함수 rect, sinc, sinc^2 푸리에 변환

<img src="https://github.com/Hyeseong0317/Math_-probability-statistics/blob/main/images/rect.jpg" width="30%" align='left'/>
<img src="https://github.com/Hyeseong0317/Math_-probability-statistics/blob/main/images/sinc.jpg" width="30%" align='right'/>
<img src="https://github.com/Hyeseong0317/Math_-probability-statistics/blob/main/images/sinc^2.jpg" width="30%" align='center'/>
<img src="https://github.com/Hyeseong0317/Math_-probability-statistics/blob/main/images/sinc.PNG" width="40%" align='center'/>

[신호처리와 수학의 상관관계 포스팅](https://jaejunyoo.blogspot.com/2020/03/signal-processing-for-communications-3-1.html)

### Hilbert Space
신호가 벡터로 표현되는 순간, 선형대수학이나 미적분학에서 개발된 강력한 도구들을 사용할 수 있게 되는데요. 이에 더해 이 신호가 Hilbert 공간 안에 있다는 것이 보장되면, 길이가 유한한 신호이든 함수와 같은 무한한 신호이든 상관없이 우리가 익숙하고 편하게 알고 있던 기하학적인 도구들을 문제없이 그대로 사용할 수 있게 됩니다.

### 푸리에 변환은 좌표축 변환이다. 
하나의 좌표축으로 표현된 어떤 벡터를 다른 좌표축으로 바꿔 표현하는 것을 벡터에 대한 좌표축 "변환 (transform)"을 수행한다고 얘기합니다. 즉, Fourier transform은 별게 아니라 임의의 벡터를 Fourier bases로 좌표축을 바꿔서 표현하는 방법이라 볼 수 있습니다.

### 푸리에 시리즈
Fourier series는 임의의 함수를 무한 개의 sinusoidal basis 함수들의 합으로 표현이 가능하다는 것을 보여줍니다. 즉, 수렴합니다.

### 내적과 유사도 관계

<img src="https://github.com/Hyeseong0317/Math_-probability-statistics/blob/main/images/내적.PNG" width="30%" align='center'/>

여기서 θ는 두 벡터 사이의 각도인데, 각도가 90도가 되면 cos 값이 0이 되면서 내적도 0이 되죠. 따라서 이런 관점에서, 내적은 두 벡터 간의 각도 차(혹은 두 벡터가 이루는 모양의 차이)를 바탕으로 유사도를 측정하는 방법이라 할 수 있습니다.

이런 내적은 신호처리에서 마주할 수 있는 다양한 상황에서 사용되는데요. 우리가 알고 싶은 내용에 따라 다른 이름으로 불리곤 합니다. 여러 신호 중 특정 신호를 찾고 싶을 때는 correlation이라는 용어로 사용되기도 하고, 신호를 바꾸거나 다른 신호로 근사하고 싶을 때는 filtering이라고 부르는데, 결국에는 모두 다 내적을 사용하여 두 신호 혹은 벡터 간의 유사성을 정량적으로 측정하는 것입니다.

위와 같은 맥락으로 이해하면 신호들 간에 orthogonality는 자연스럽게 두 신호가 완벽하게 구별이 가능하다는 뜻으로 이해할 수 있게됩니다. 즉, 두 신호의 생김새가 완전히 다른 것이죠.

이렇듯 유사성을 정량적으로 측정하기 위해 *내적으로부터 "거리"라는 개념이 나오고 우리가 잘 아는 유클리디안 거리가 나오는 것입니다.

<img src="https://github.com/Hyeseong0317/Math_-probability-statistics/blob/main/images/유클리디안거리.PNG" width="50%" align='center'/>

### Subspaces and Bases
Basis는 어떤 공간에 대한 구조를 이루는 뼈대이다. 공간을 망가뜨리지 않고 늘리거나 휘거나 돌리는 등의 선형 변환(linear transformation)을 할 수 있다.
#### Change of basis & Fourier transform
유연한 성질을 바탕으로, 기존 공간의 성질들을 망가뜨리지 않는 선에서 뼈대를 맘대로 바꾸는 것을 "change of basis"라고 합니다. 이렇게 뼈대를 바꾸는 까닭은 보통 현재의 구조로는 살펴보기가 힘든 정보가 있어서, 이를 좀 더 쉽게 살펴볼 수 있도록 시점을 바꾸기 위해서 입니다. 예를 들면 앞선 글에서 말했었듯 차후 배울 Fourier transform 역시도 대표적인 change of basis 중 하나입니다.
#### Subspace
간혹 벡터 공간 전체가 아닌 좀 더 한정된 공간을 깊게 분석하기 위해 공간의 일부분만을 떼어서 살펴볼 때가 있습니다. 이를 subspace라고 부르는데요. Subspace는 벡터 공간 안에서 허용되는 operation들에 대해 닫혀있는 공간이기 때문에, 마치 샌드박스처럼 해당 공간을 벗어나지 않으면서도 하고 싶은 분석을 할 수 있게 됩니다. 당연하지만 subspace도 공간이기 때문에 자기 자신만의 뼈대를 가지고 있습니다.
#### Gramm-Schmitt (linearly independent --> orthonormal)
벡터가 서로 orthogonal하지만 normal하지는 않을 수도 있는데 사실 서로 선형 독립이기만 하면 Gramm-Schmitt로 언제나 orthonormal하게 만들 수 있기 때문에 별 문제 될 것은 없습니다.
#### subderivative
수학에서 하방미분(subdifferential, subderivative)은 미분을 일반화하여 미분가능하지 않은 볼록 함수에 적용할 수 있도록 하는 방법이다. 볼록 최적화 등 볼록 함수를 연구하는 해석에서 중요하게 사용된다.
#### supremum of convex functions
볼록 함수의 극한
#### supp
수학에서, 함수의 지지집합(支持集合, 영어: support 서포트[*]) 또는 받침은 그 함수가 0이 아닌 점들의 집합의 폐포이다.
#### 폐포
위상수학에서, 폐포(閉包, 영어: closure)는 주어진 위상 공간의 부분 집합을 포함하는 가장 작은 닫힌집합이다.[1] 이는 그 부분 집합의 원소와 극한점으로 구성된다.
#### 위상수학
위상수학(位相數學, 영어: topology)은 연결성이나 연속성 등, 작은 변환에 의존하지 않는 기하학적 성질들을 다루는 수학의 한 분야이다.
#### Pseudo code
의사코드는 프로그램을 작성할 때 각 모듈이 작동하는 논리를 표현하기 위한 언어이다. 특정 프로그래밍 언어의 문법에 따라 쓰인 것이 아니라, 일반적인 언어로 코드를 흉내 내어 알고리즘을 써놓은 코드를 말한다.
#### multiple critical points
여러 임계점 -> 도함수가 0이 되는 점
#### 임계점
수학에서, 임계점 또는 정류점 또는 정상점은 함수의 도함수가 0이 되는 점이다. 극대점이나 극소점, 또는 안장점으로 분류된다.
#### 하한 lower bound(파란밑줄)을 찾음으로써 우리가 아는 모델(파란밑줄)로 바로 계산을 할 수 없는 실제 모델에 approximate를 시킨다.
<img src="https://github.com/Hyeseong0317/Math_-probability-statistics/blob/main/images/VAE.jpg" width="60%">

#### Fixed_noise pattern
고정 패턴 노이즈는 특정 픽셀이 평균 강도보다 더 밝은 강도를 제공하기 쉬운 긴 노출 촬영 중에 종종 눈에 띄는 디지털 이미징 센서의 특정 노이즈 패턴에 주어진 용어입니다.

<img src="https://github.com/Hyeseong0317/Math_-probability-statistics/blob/main/images/Fixed_noise pattern.PNG" width="30%">

### 선형대수

#### Norm
<img src="https://github.com/Hyeseong0317/Math_-probability-statistics/blob/main/images/norm.jpg" width="50%">

#### dot product
내적은 두벡터가 얼마나 닮았는가. 닮은 정도를 나타냅니다. 한 벡터의 성분이 다른방향의 벡터의 성분을 얼마나 가지고 있느냐라고 생각할 수 있습니다.

#### Hessian matrix
함수를 볼록하게 만들어줍니다. 더 볼록해진 부분의 방향으로의 고유벡터의 값은 크게 나타나며 이미지내에서는 고유벡터값이 큰 부분이 뚜렷하게 나타나게 됩니다.

##### Hessian matrix는 symmetric하다. 연속되는 두번의 미분의 순서는 상관이 없다.
The Hessian matrix is a symmetric matrix, since the hypothesis of continuity of the second derivatives implies that the order of differentiation does not matter (Schwarz's theorem). The determinant of the Hessian matrix is called the Hessian determinant.

<img src="https://github.com/Hyeseong0317/Math_-probability-statistics/blob/main/images/Hessian matrix.PNG" width="50%">

#### Jacobian matrix

비선형 변환을 국소적으로 선형변형으로 근사한것.

<img src="https://github.com/Hyeseong0317/Math_-probability-statistics/blob/main/images/Jacobian.jpg" width="40%">

행렬식의 기하학적 의미 : 행렬식은 선형변환할 때 단위면적이 얼마만큼 늘어나는가를 말해줍니다. 따라서, Jacobian 행렬의 행렬식의 의미는 원래 좌표계에서 변환된 좌표계로 변환될 때의 넓이의 변화비율을 말해줍니다.

### Score matching과 Jacobian

Jacobain은 비선형 변환을 국소적으로 선형변환하기 때문에 변환 이후 Subspace의 차원이 축소될 수 있으며 이로 인해 변환된 Subspace보다 고차원의 차원은 고려할 수 없게 되는 단점이 있습니다. 

<img src="https://github.com/Hyeseong0317/Math_-probability-statistics/blob/main/images/score matching.PNG" width="60%">

#### tsne

tsne는 neural network를 거쳐나온 데이터들을 클러스터링한 것을 시각화하여 보여줍니다. 

<img src="https://github.com/Hyeseong0317/Math_-probability-statistics/blob/main/images/tsne_mnist.png" width="40%">

PCA와 Local Linear Embedding은 차원 축소 방법을 선형적으로 접근하지만 T-SNE는 비선형적으로 접근하기 때문에 표현력이 증가됩니다. 따라서 위 시각화 결과와 같이 T-SNE는 클래스 간 분별력이 있게 시각화 할 수 있습니다. PCA 주성분 분석을 통해 각 데이터들의 분산을 가장 넓게 잘표현하는 쪽으로 차원을 축소합니다. 주성분은 벡터에서는 기저벡터의 스칼라값이 가장 큰 성분을 의미합니다.

#### Largrangian Multiplier와 Local minimum의 기하학적 상관관계

<img src="https://github.com/Hyeseong0317/Math_-probability-statistics/blob/main/images/Lagrangian multiplier.jpg" width="40%">


#### Largrangian Multiplier와 Local minimum -> Solution

<img src="https://github.com/Hyeseong0317/Math_-probability-statistics/blob/main/images/Lagrangian multiplier solution.jpg" width="40%">

#### Newton method는 사실상 1차 Taylor series입니다. 

->Taylor series로 주어진 함수를 1차 함수로 근사시켜서 주어진 함수의 zero finding을 할 수 있습니다. 

->Taylor series로 주어진 함수를 2차 함수로 근사시켜서 주어진 함수의 zero finding을 할 수 있습니다.

### 라그랑주 승수법 (Lagrange Multiplier)

라그랑지안 승수법(Lagrange multiplier method)은 제약식에 형식적인 라그랑지안 승수를 곱한 항을 최적화하려는 목적식에 더하여, 제약된 문제를 제약이 없는 문제로 바꾸는 기법입니다. 

목적함수 = 제약식 + 라그랑주 승수 알파 * 항 ; equality constraints problem(제약조건이 있는 문제) -> inequaility constraints problem(제약이 없는 문제)

라그랑주 승수법 (Lagrange multiplier method)은 프랑스의 수학자 조세프루이 라그랑주 (Joseph-Louis Lagrange)가 제약 조건이 있는 최적화 문제를 풀기 위해 고안한 방법이다. 라그랑주 승수법은 어떠한 문제의 최적점을 찾는 것이 아니라, 최적점이 되기 위한 조건을 찾는 방법이다. 즉, 최적해의 필요조건을 찾는 방법이다.

<img src="https://github.com/Hyeseong0317/Math_-probability-statistics/blob/main/images/라그랑주제약식.PNG" width="60%">

### 상계,하계,상한,하한 (upper bound, lower bound, supremum, infimum0

<img src="https://github.com/Hyeseong0317/Math_-probability-statistics/blob/main/images/상계하계.PNG" width="50%">
