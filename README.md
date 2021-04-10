https://kimchangheon.tistory.com/5

# 분자구조 이미지 SMILES(분자구조 문자열 표현식) 변환 AI 개발

![image](https://user-images.githubusercontent.com/39324421/114260535-6e8fbc80-9a10-11eb-8e3f-59c942b727bd.png)

작년 9월 LG 사이언스파크에서 주최한 분자구조 이미지 SMILES 변환 해커톤에 참가하여 최종 8위를 기록하며 LG 추천인 10명에 선발되는 성과를 내었습니다. 이후 면접을 준비하며 모델 성능을 개선하여 리더보드 점수 0.9941를 기록하며 4위에 이름을 올릴 수 있었습니다.

![image](https://user-images.githubusercontent.com/39324421/114260525-6768ae80-9a10-11eb-93c4-2c509cb3e8bd.png)

## 대회 배경
참가한 대회의 주제는 "분자구조 이미지를 SMILES(분자구조 문자열 표현식)으로 변환하는 AI"를 개발하는 것이였습니다.

SMILES(Simplified Molecular Input Line Entry System)는 기존의 분자식[footnote]분자식은 화학식의 일종으로서 분자를 구성하는 원자의 종류와 수를 전부 적어주는 식이다[/footnote]보다 컴퓨터에서 다루기 편한 화학분자표현 방법입니다. SMILES는 AI를 이용한 신약개발,신소재개발 등에 활용이 됩니다. AI를 활용하면 기존의 전통적인 방식에 비해 획기적으로 비용과 시간을 절약할 수 있게 됩니다.

![image](https://user-images.githubusercontent.com/39324421/114260551-9121d580-9a10-11eb-8f9d-37aea848fd27.png)

따라서 전 세계적으로 논문, 특허 등 문서에서 쏟아져 나오는 새로운 화학물질 구조 이미지를 SMILES로 변환하여 데이터베이스에 저장하는 것은 굉장히 중요합니다. 

하지만 이를 사람이 직접 레이블링 하기에는 전문적인 도메인 지식이 필요하며 시간이 오래 걸립니다.

또한 컴퓨터를 활용하여도SMILES를 이미지로 변환하는 것은 쉽지만 이미지를SMILES로 변환하는 것은 어려운 문제입니다

예를 들면, Nc1ncn2c1ncn2C1OC(COP(=O)(O)O)C(OP(=O)(O)(O)0)C10와 같은 물질의 SMILES길이는47이므로 각 문자를 맞출 확률이95%라고 하더라도 32개를 다 맞출 확률은 1%미만입니다.

![image](https://user-images.githubusercontent.com/39324421/114260567-aeef3a80-9a10-11eb-8ef1-545fad2f9235.png)

따라서 이번 에세이에서는 어떻게 하면 분자구조 이미지를 SMILES로 변환하는 AI모델을 만들 수 있는지 알아보도록 하겠습니다.

## 활용기술
인공지능이란?
컴퓨터과학 분야에서 인공지능(AI)란 기계가 스스로 생각하고 판단하여 행동을 하도록 하는 것입니다. 여기서 활용된 AI알고리즘은 인간의 뇌에 있는 뉴런구조를 본 떠서 만든 인공 신경망(Artificial Neural Network)이라는 것입니다. 인공 신경망은 입력뉴런층, 은닉뉴런층, 출력 뉴런 층으로 이루어지는데, 입력뉴련층에는 분자구조 이미지가 들어가고 출력 뉴련층에서 예측한 SMILES가 나오는 방식입니다. 일반적으로 은닉 뉴런층의 깊이가 3이상이 되면 심층신경망(Deep Neural Network)라고 하는데 이것이 바로 딥러닝(Deep learning)이며 AI라는 범주 아래 머신러닝이 있고 그 아래 딥러닝이 포함되는 개념입니다.

![image](https://user-images.githubusercontent.com/39324421/114260571-badafc80-9a10-11eb-92e1-f9fbe8708602.png)
출저:https://m.blog.naver.com/godinus123/222024409883?view=img_1


신경망은 여러 개의 층(layer)로 이루어져 있고 층(layer)은 여러 개의 뉴런으로 이루어져 있습니다. 각 뉴런은 가중치(Weight)와 편향(Bias)를 가지며 뉴런에 입력되는 데이터를 x 라고 하면 Wx+b 의 결과를 다음 뉴런에 전달합니다. 이러한 뉴런들을 잘 짜여진 구조로 다양하게 배치하면 이미지분류, 주가예측, 물건 판매량예측, 화학물질 물성, 독성 예측, 전기차 배터리 용량, 수명예측, 부품, 완성품불량예측 등 다양한 산업분야 문제를 해결할 수 있고 기존 방법에 비해서 월등한 성능을 냅니다. 이를 위해서 신경망에 있는 뉴런 가중치를 적절하게 학습시키는 것이 중요한데 이는 gradient descent(경사 하강법)알고리즘을 이용하면 가능합니다. 이는 함수 최적화(optimization)방법 중 하나로써 미분을 이용하여 어떤 비용함수(cost function)의 값을 최소화 시키기 위한 파라미터(wieght, bias)를 점진적으로 찾는 것입니다.

이번 문제에서는 딥러닝 알고리즘 중에서도 특히 이미지를 설명하는 이미지 캡셔닝 기술을 응용했습니다. 이미지 캡셔닝은 CNN[footnote]합성곱 신경망(Convolutional neural network, CNN)은 시각적 영상을 분석하는 데 사용되는 다층의 피드-포워드적인 인공신경망의 한 종류이다[/footnote](Convolution Neural network)을 encoder로 RNN[footnote]순환 신경망(Recurrent neural network, RNN)은 인공 신경망의 한 종류로, 유닛간의 연결이 순환적 구조를 갖는 특징을 갖고 있다.[/footnote](Recurrent Neural Network)을 decoder로 활용하는 Encoder-to-Decoder모델입니다. 특정 이미지를 넣으면 해당 이미지에 대한 설명을 문장으로 출력해줍니다. 마찬가지로 Encoder에 분자구조 이미지를 넣으면 context vector(문맥벡터)를 얻어내고 이를 통해 SMILES단어들을 하나씩 순서대로 예측하게 됩니다. 학습된 Encoder로부터 이미지의 정보를 잘 압축한 문맥 벡터를 얻어내고 이를 Decoder에 입력으로 주어 원하는 단어(문자)를 순차적으로 뱉어내게 만드는 것이 Encoder-to-Decoder모델의 원리인 것입니다.

![image](https://user-images.githubusercontent.com/39324421/114260582-d0e8bd00-9a10-11eb-94af-0ccf893a5ae7.png)

위에서 언급된 CNN은 이미지 인식이나 특징추출을 위한 신경망이며 사람보다 인식률이 좋다고 평가받고 있습니다. RNN은 문장에서 나올 다음 단어를 예측하는 것과 같이 순차데이터를 예측하기 위한 신경망입니다.

![image](https://user-images.githubusercontent.com/39324421/114260737-99c6db80-9a11-11eb-865e-7f4b061b6334.png)
기존 연구내용도 encoder to decoder model을 활용했음을 볼 수 있습니다.

## 데이터 수집 및 생성
이제 모델에 학습시킨 데이터는 어떻게 수집하고 정제했는지 말씀을 드리겠습니다. PubChem[footnote]PubChem은 화학 분자 및 생물학 논문에 대한 활동의 데이터베이스이다.[/footnote] 데이터베이스에서 약 1억 1천만개의 SMILES데이터를 받아올 수 있었고 원활한 학습을 위해서 길이 70이하 SMILES만 선택했습니다. 그리고 전체 SMILES에서 등장하는 문자 70개 중 빈도가 낮은 하위 35개를 포함한 SMILES는 학습에서 제외 시켰습니다. 이러한 전처리[footnote]주어진 원데이터를 그대로 사용하기보다는 원하는 형태로 변형해서 분석하는 경우가 굉장히 많다. 따라서 분석과 모델링에 적합하게 데이터를 가공하는 작업을 ’데이터 전처리’라고 한다.[/footnote] 과정을 거치고 나니 총 9300만개의 SMILES를 얻을 수 있었고, Rdkit[footnote]RDKIT은 화학물질의 정보를 담고 있는 파일형식의 데이타를 이용해서 화학물질의 구조이미지(구조식)를 만들어내는 툴[/footnote]을 이용해 SMILES를 이미지로 변환하여 학습에 사용하였습니다.
![image](https://user-images.githubusercontent.com/39324421/114260596-e6f67d80-9a10-11eb-8924-01778b20af70.png)




## 모델 구조
입력된 분자구조이미지로부터 예측된 smiles를 출력하기 위해 Encoder-to-Decoder 구조에 Attention Mechanism(어텐션 메커니즘)을 추가한 모델을 사용하였습니다.

모델의 구조는 크게 세 파트로 나뉩니다. 이미지를 인식하고 해당 이미지에서 필요한 특성(feature)들을 뽑아내는 Encoder 파트, 학습하는 과정에서 이미지의 어떤 부분을 집중해서 봐야 할지 알려주는 Attention 파트, 그리고 Encoder 와 Attention에서 알려준 정보를 가지고 알맞은 Token 을 순차적으로 내뱉는 Decoder 부분이 있습니다.
![image](https://user-images.githubusercontent.com/39324421/114260603-f4ac0300-9a10-11eb-95a0-3085997023b2.png)

## 인코더
![image](https://user-images.githubusercontent.com/39324421/114260716-7e5bd080-9a11-11eb-984d-5520db14d3f3.png)

Encoder는 CNN구조이며 EfficientNetB0을 적용하였습니다. 인코더에서는 분자구조이미지텐서를 받아서 features텐서를 리턴합니다.

인코더 내부의 자세한 과정에 대한 설명을 덧붙이자면 다음과 같습니다

- 입력 : (512,224,224,3)의 분자구조 이미지 텐서 (batch크기, width, height, depth)

- 출력 : (512, 49, 512)의 encoder 피쳐 (batch크기, 49, embedding_dim)

1.  인자로 받은 image tensor를 efficientnetB0을 이용하여 피쳐맵(batch크기,7,7,1280)을 추출합니다(image tensor는 tfrecord로부터 가져온 image_raw를    decode_image함수에 인자로 넣어서 얻은 값입니다.)

2.  dropout을 적용합니다.

3.  추출된 피쳐맵을 (batch크기, 49,1280)로 reshape합니다

4.  embedding_dim(512)크기의 Dense층을 적용하고 relu 활성화 함수를 적용합니다. 


## 디코더
Decoder는 RNN구조이며 바다니우 어텐션을 적용하였습니다

디코더에는 3가지 입력이 들어갑니다. 1. 현재 시점(t)의 word (초기 입력으로 smiles의 시작을 의미하는 '<'가 들어갑니다) tensor의 shape은 (512,49,512) 입니다 2. 인코더에서 나온 피쳐 3. 디코더의 이전 시점(t-1) 은닉상태[footnote]RNN의 메모리 셀이 출력층과 다음시점 연산으로 내보내는 결과를 은닉상태라고 한다[/footnote]

출력은 3가지 입니다. 1. 현 시점(t)의 예측 word 텐서 (batch크기,37) 2. 디코더의 현 시점(t) 은닉상태 3. attention weight(어텐션 가중치) --> 리턴(출력)하지만 이후 사용되지는 않습니다.

어텐션 메커니즘은 기존 인코더 투 디코더 모델의 컨텍스트 벡터를 개선함으로써 성능을 향상시키는 기법입니다.

보통 인코더가 RNN모델에 기반해 있는 인코더 투 디코더 모델(seq2seq)은 인코더의 마지막 은닉 상태를 컨텍스트 벡터로 하고 이를 디코더 RNN 셀의 첫번재 은닉 상태로 사용합니다.
이 모델에서는 인코더가 CNN모델에 기반 해 있으므로 인코더의 최종 출력이 컨텍스트 벡터의 역할을 할 수 있습니다.

컨텍스트 벡터를 디코더의 초기 은닉 상태로만 사용하는것에서 더 나아가 컨텍스트 벡터를 디코더가 word를 예측하는 매 시점마다 하나의 입력으로 사용할 수도 있습니다. 이에 더 나아가 어텐션 메커니즘을 사용하면 더욱 문맥을 반영할 수 있는 입력을 디코더에 줄 수 있습니다.

디코더에 적용된 어텐션 메커니즘은 출력 단어를 예측하는 매 시점마다 분자이미지전체(인코더를 통해 얻은 피쳐)를 참고하는 것입니다.(기계번역 모델이라면 전체 입력 문장을 참고한다고 보면 됩니다).단 해당 시점에서 예측해야할 word와 연관이 있는 피쳐부분을 좀더 집중(attention)하여 보는 것입니다.

디코더 내부의 gru[footnote]GRU(Gated Recurrent Unit) 셀은 2014년에 K. Cho(조경현) 등에 의해 제안된 LSTM셀의 간소화된 버전이라고 할 수 있다. LSTM과 같이 RNN의 장기 의존성 문제를 해결한다.[/footnote]층에는 "현재 시점(t)의 word를 embedding한 vector"와 "context_vector"를 연결하여 입력벡터로 줍니다.

Decoder 내부의 과정에 대하여 자세하게 설명드리겠습니다. 

1.  features(인코더에서 나온 피쳐), hidden(디코더의 이전 시점(t-1) 은닉상태)를 이용하여 context_vector와 attention_weight를 구합니다.     
  -  feature와 hidden을 이용하여 어텐션 스코어를 구하고 이에 소프트맥스를 적용하여 attention_weight를 구하는 것입니다.   
  - 어텐션 스코어를 구하는 방법은 여러가지가 있습니다. 그중 바다니우가 제시한 'concat'방법을 사용합니다.    
  - attention_weight와 hidden을 가중합하여 context_vector를 구합니다. 이는 Attention value라고도 합니다.   

2.  x(현재 시점(t)의 word)를 512차원의 vector로 변환합니다.   

3.  context_vector와 x(임베딩된 word 벡터)를 연결(concatenate)하여 gru층의 입력으로 사용합니다. 

4.  dropout을 거치고, 2개의 Dense layer를 거쳐 다음에 올 단어 예측값을 출력합니다


## 학습방법
생성한 9300만장의 이미지 데이터와 인코더 투 디코더 모델을 이용하여 학습을 시켰습니다.

##- 훈련데이터 포맷 : Tfrecord
새로 생성한 분자구조 이미지와 라벨을 tfrecord 형태로 google cloud storage에 저장하였습니다. 이는 이후 학습시에 TPU를 사용하기 위함입니다.

데이터 양이 많을 경우 이를 Binary로 Seralization한 뒤 파일 형태로 저장하고 있다가 이를 다시 읽어들이는 방식으로 처리하면 처리 속도가 향상됩니다. 특히 데이터를 네트워크로 주고받을 때에는 더욱 큰 이득을 볼 수 있습니다. 학습시 TPU를 사용할 계획이므로 GCS에 저장후 네트워크로 데이터를 TPU에 전송해야합니다. 텐서플로우의 TFrecords를 사용하면 Protocol Buffer형태로 Serialization을 수행하여 이미지와 레이블을 저장합니다

* Protocol Buffer는 구글에서 개발하고 오픈소스로 공개한, 직렬화 데이타 구조 (Serialized Data Structure)

## - 학습 환경 : 
Kaggle TPU v3-8, 8Cores , 128GB , 420teraflops
Colab pro TPU v2-8 : 8Cores, 64GiB , 180 teraflops
kaggle과 colab에서는 TPU를 이용할 수 있습니다.  1000만원이 넘는 가격의  v100이 112 teraflops의 성능을 내는 것을 보면 colab과 kaggle에서 제한적이지만 무료로 사용가능한 TPU는 굉장히 좋은 조건입니다. 

## - loss function : SparseCategoricalCrossentropy
레이블 class가 두개 이상일때 손실함수로 crossentropy loss function를 사용합니다. 이는 label(real)과 prediction(pred)간의 crossentropy loss를 계산합니다.

SparseCategoricalCrossentropy함수는 레이블은 정수로 입력받으며, 원핫 벡터로 입력받고 싶으면 CategoricalCrossentropy를 사용해야합니다.

여기서 real값은 디코더에서 예측할 현 시점(t)의 입력 word(글자)입니다.즉 0부터 36사이 정수(word를 숫자로 표현한것)이며, pred값은 디코더에서 예측한 현 시점(t)의 예측된 37차원 word 텐서입니다 shape :(batch크기,37)

pred과 real값의 예시입니다.

pred는 37차원 텐서 : [ 0.08155002, -0.01409034, -0.16667187, ..., 0.1283614 , 0.01665916, 0.1495304 ]

real은 1차원 텐서 : 1

한번에 배치사이즈 512만큼 연산하므로 pred의 shape은 (512,37)이 되고 real의 shape은 (512,)이 됩니다

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0)) #0(패딩)이 아닌것만 True로 mask한다. 
    loss_ = loss_object(real, pred) # real과  pred간의 loss를 구한다.

    mask = tf.cast(mask, dtype=loss_.dtype) # True 를 1로 변경한다.
    loss_ *= mask# loss와 mask를 곱한다.

    return tf.reduce_mean(loss_) #loss의 평균을 구한다.
    
### - optimizer : Adam

## 앙상블 방법
총 10개의 모델을 모델 페어 기반으로 유사도 점수를 매겨 앙상블 하였습니다.
![image](https://user-images.githubusercontent.com/39324421/114260639-3177fa00-9a11-11eb-9274-5c09dc0824d8.png)


## 결과
![image](https://user-images.githubusercontent.com/39324421/114260645-3472ea80-9a11-11eb-8725-4546345e0365.png)

9300만장의 분자구조 이미지를 모델에 학습 시킨 결과 2만장의 시험 데이터에 대해서 성능지표인 tanimoto similarity[footnote]Fingerfrint으로 표시되는 화학 구조를 비교 하는 데 가장 널리 사용되는 유사성 측정 값[/footnote] 기준 0.99394수준의 성능을 얻을 수 있었고 이는 2만개 중 19878개의 분자구조를 정확히 맞춘 수준입니다.

지금까지 분자구조 이미지를 SMILES로 변환하는 AI개발에 대하여 간략하지만 핵심적인 부분들을 설명하였습니다. 본 기재에서 설명하는 AI기술은 논문, 기사, 특허 데이터에 있는 화학물질구조 수집과정을 가속화하여 QSAR[footnote]구조-활성의 정량적 관계 (Quantitative structure–activity relationship, QSAR)모델은 화학, 생물학, 공학에서 사용되는 회귀 또는 분류모델로, 화학구조와 예측하고자 하는 활성 간의 정량적인 수학적 모델을 말한다.[/footnote]/QSPR[footnote]구조-물성의 정량적 관계 (QSPR)는 화학적 물성을 사용하는 것이며, 다양한 화학분자의 물성이나 거동 등이 QSPR분야에서 연구되고 있다.[/footnote] 모델링를 하는데 큰 도움이 될 것입니다.

Github
github.com/Kimchangheon/LG_SMILES


Kimchangheon/LG_SMILES

Contribute to Kimchangheon/LG_SMILES development by creating an account on GitHub.

github.com

## 참고 문헌

1)“LG SMILES 경진대회”,데이콘. 2020년 12월 9일 수정 2020년 12월9일 접속 https://dacon.io/competitions/official/235640/overview/

2)“딥러닝을 이용한 자연어 처리 입문”. 위키독스. 2020년 12월 9일 접속. https://wikidocs.net/22893

3)“Chemical_similarity” Wikipedia. 2020년 12월 9일 접속, https://en.wikipedia.org/wiki/Chemical_similarity

4)Kohulan Rajan et al(2020) DECIMER: towards deep learning for chemical image recognition : Journal of Cheminformatics. https://jcheminf.biomedcentral.com/articles/10.1186/s13321-020-00469-w

5)Joshuaet al(2019) Molecular Structure Extraction From Documents Using Deep Learning : J. Chem. Inf. Model. 2019, 59, 3, 1017–1029 Publication Date:February 13, 2019

https://doi.org/10.1021/acs.jcim.8b00669


각주
분자식은 화학식의 일종으로서 분자를 구성하는 원자의 종류와 수를 전부 적어주는 식이다 [본문으로]

합성곱 신경망(Convolutional neural network, CNN)은 시각적 영상을 분석하는 데 사용되는 다층의 피드-포워드적인 인공신경망의 한 종류이다 [본문으로]

순환 신경망(Recurrent neural network, RNN)은 인공 신경망의 한 종류로, 유닛간의 연결이 순환적 구조를 갖는 특징을 갖고 있다. [본문으로]

PubChem은 화학 분자 및 생물학 논문에 대한 활동의 데이터베이스이다. [본문으로]

주어진 원데이터를 그대로 사용하기보다는 원하는 형태로 변형해서 분석하는 경우가 굉장히 많다. 따라서 분석과 모델링에 적합하게 데이터를 가공하는 작업을 ’데이터 전처리’라고 한다. [본문으로]

RDKIT은 화학물질의 정보를 담고 있는 파일형식의 데이타를 이용해서 화학물질의 구조이미지(구조식)를 만들어내는 툴 [본문으로]

RNN의 메모리 셀이 출력층과 다음시점 연산으로 내보내는 결과를 은닉상태라고 한다 [본문으로]

GRU(Gated Recurrent Unit) 셀은 2014년에 K. Cho(조경현) 등에 의해 제안된 LSTM셀의 간소화된 버전이라고 할 수 있다. LSTM과 같이 RNN의 장기 의존성 문제를 해결한다. [본문으로]

Fingerfrint으로 표시되는 화학 구조를 비교 하는 데 가장 널리 사용되는 유사성 측정 값 [본문으로]

구조-활성의 정량적 관계 (Quantitative structure–activity relationship, QSAR)모델은 화학, 생물학, 공학에서 사용되는 회귀 또는 분류모델로, 화학구조와 예측하고자 하는 활성 간의 정량적인 수학적 모델을 말한다. [본문으로]

구조-물성의 정량적 관계 (QSPR)는 화학적 물성을 사용하는 것이며, 다양한 화학분자의 물성이나 거동 등이 QSPR분야에서 연구되고 있다. [본문으로]
