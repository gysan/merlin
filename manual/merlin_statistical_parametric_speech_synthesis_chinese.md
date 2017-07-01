# Merlin:中文统计参数语音合成实战

Author: 

肖鉴津Jackiexiao 
张楚雄 

Date:

20170701

本文目标是详细解释如何基于开源Merlin项目搭建中文统计参数语音合成系统，但笔者目前尚未实现中文语音合成，本文记录了笔者的进展并且会持续更新直到实现中文语音合成为止。

目录

 * [目前研究进展](#目前研究进展)
 * [Merlin的安装与运行](#merlin的安装与运行) 
 * [Merlin开源代码的学习](#merlin开源代码的学习)
 * [英文语音合成理论研究](#英文语音合成理论研究)
 * [英文语音合成实现](#英文语音合成实现)
 * [中文语音合成理论研究](#中文语音合成理论研究)
 * [中文语音合成实现](#中文语音合成实现)
 * [参考文献](#参考文献)
 * [术语表](#术语表)
 * [扩展](#扩展)
 * [TODO List](#todo-list)

# 目前研究进展

百度Deep Voice

Deep Voice在发音前先将文字转化为音素（最小的语音单位），然后再依靠自己的语音合成网络将其变为你所听到的声音。该系统包含 5 个重要基础：定位音素边界的分割模型、字母到音素的转换模型、音素时长预测模型、基础频率预测模型、音频合成模型。为了更好的表达情感，需要人为的控制音素、音节的加重、缩短以及拖长。

 在几乎无须人工介入的前提下，只需短短数小时便能学会说话。开发人员还可以对其要传达的感情状态进行设定，这样合成出来的语音听起来就会非常真实、自然。

DeepMind WaveNet

WaveNet采用的是参量式TTS模型，通过直接将音频信号的原始波形进行建模，并且一次产生一个样本，通过vocoders的信号处理算法来输出它的结果，以生成语音信号。此外，使用原始波形，意味着WaveNet可以对包括音乐在内的任何音频进行建模，这样子生成的语音听起来会更自然。

WaveNet是一个完全的卷积神经网络。在这其中，卷积层拥有不同的扩张因素，能够让它的接受域随着成千上万的timesteps的深度和覆盖呈现指数型的增长。

在训练时，输入序列是来自人类演讲者的真实的波形。在训练后，我们可以取样这个网络来生成合成的语音。每一步的采样值是由网络计算得出的概率分布得到的。这个值随后会重新回到输入端，然后在下一步生成一个新的预测。构建一个像这样能够一次进行一步取样的网络是需要大量的计算的，所以在效率方面是WaveNet比较头痛的问题。

使用的开源库：Merlin语音合成系统

*Merlin is a toolkit for building Deep Neural Network models for  statistical parametric speech synthesis. It must be used in combination  with a front-end text processor (e.g., Festival) and a vocoder (e.g.,  STRAIGHT or WORLD).*

*The system is written in Python and relies on the Theano numerical computation library.*

*Merlin comes with recipes (in the spirit of the Kaldi automatic  speech recognition toolkit) to show you how to build state-of-the art  systems.*

*Merlin is free software, distributed under an Apache License  Version 2.0, allowing unrestricted commercial and non-commercial use  alike.*

# Merlin的安装与运行

## 1.基础知识

Merlin只能在unix类系统下运行，使用Python2.7，并用theano作为后端，因此在使用Merlin之前，至少需要如下基础

- 熟练掌握linux系统，熟悉shell脚本


- 掌握python，了解常用Python库的使用


- 掌握theano


- 机器学习基础知识


- 语音识别和语音合成的知识

熟练地掌握上述知识需要花费大量的时间。在文章的末尾，我们会贴出我们所使用/推荐的学习资料。

## 2.安装Merlin

Merlin的Python语言采用的是Python2.7编写，所以我们需要在Python2.7的环境下运行Merlin，为避免python不同版本之间的冲突，我们采用Anaconda对Python运行环境进行管理。

使用Anaconda创建Merlin运行环境具体操作如下：

打开终端，使用下面命令查看一下现有python环境

`conda env list`

使用下面命令创建一个名为merlin的python环境

`conda create --name merlin python=2.7`

先进入merlin环境中

`source activate merlin`

在这个环境下安装merlin

```
sudo apt-get install csh
pip install numpy scipy matplotlib lxml theano bandmat
git clone [https://github.com/CSTR-Edinburgh/merlin.git](https://github.com/CSTR-Edinburgh/merlin.git)
cd merlin/tools
./compile_tools.sh

```

如果一切顺利，此时你已经成功地安装了Merlin，但要注意的是Merlin不是一个完整的TTS系统。它提供了核心的声学建模功能：语言特征矢量化，声学和语言特征归一化，神经网络声学模型训练和生成。但语音合成的前端（文本处理器）以及声码器需要另外配置安装。此外，Merlin目前仅提供了英文的语音合成。

此外，上述安装默认只配置支持CPU的theano，如果想要用GPU加速神经网络的训练，还需要进行其他的步骤。由于语料库的训练时间尚在笔者的接受范围之内（intel-i5，训练slt_arctic_full data需要大概6个小时），因此这里并没有使用GPU进行加速训练。

## 3.运行Merlin demo

`.～/merlin/egs/slt_arctic/s1/run_demo.sh`

该脚本会使用50个音频样本进行声学模型和durarion模型的训练，并合成5个示例音频。在此略去详细的操作步骤，具体可参见：Getting started with the Merlin Speech Synthesis Toolkit [installing-Merlin](https://jrmeyer.github.io/merlin/2017/02/14/Installing-Merlin.html)

# Merlin开源代码的学习

## 0 文件含义

Folder        |    Contains
------------- | -------------------
recordings    |     speech recordings, copied from the studio
wav           |     individual wav files for each utterance
pm            |     pitch marks
mfcc          |     MFCCs for use in automatic alignment
lab           |     label files from automatic alignment
utt           |     Festival utterance structures
f0            |     Pitch contours
coef          |     MFCCs + f0, for the join cost
coef2         |     coef2, but stripped of unnecessary frames to save space, for the join cost
lpc           |     LPCs and residuals, for waveform generation
bap           |     band aperiodicity

1 免费的语料库

Merlin使用了网络上免费的语料库slt_arctic，可以在以下网址进行下载：[http://104.131.174.95/slt_arctic_full_data.zip](http://104.131.174.95/slt_arctic_full_data.zip)

2 训练数据的处理

Merlin自带的demo（merlin/egs/slt_arctic/s1 ）已经事先完成了label文件的提取，所以这里不需要前端FrontEnd对数据进行处理。

Merlin通过脚本文件setup.sh在～/merlin/egs/slt_arctic/s1 目录下创建目录experiments，在experiments目录下创建目录slt_arctic_demo，完成数据的下载与解压，并将解压后的数据分别放到slt_arctic_demo/acoustic_mode/data，slt_arctic_demo/duration_model/data目录下，分别用于声学模型和持续时间模型的训练。

3 Demo语料库的训练

run_demo.sh文件会进行语音的训练以及合成。这里有许多的工程实现细节，在这里略去说明，其主要进行了如下步骤

![img](https://images-cdn.shimo.im/fXAcxtvH2dosP58A/image.png!thumbnail)

其中语料库包含了文本和音频文件，文本需要首先通过前端FrontEnd处理成神经网络可接受的数据，这一步比较繁琐，不同语言也各不相同，下面会着重讲解。音频文件则通过声码器（这里使用的是STRAIGHT声码器）转换成声码器参数（包括了mfcc梅谱倒谱系数，f0基频，bap：band aperiodicity等）再参与到神经网络的训练之中。

4 Demo语料库的合成

Demo中提供了简单的合成方法，使用demo（merlin/egs/slt_arctic/s1 ）下的脚本文件：merlin_synthesis.sh即可进行特定文本的语音合成。

同样的，由于merlin没有自带frontend，所以其demo中直接使用了事先经过frontend转换的label文件作为输入数据来合成语音。如果想要直接输入txt文本来获得语音，需要安装FrontEnd（下文会提及）并根据merlin_synthesis.sh文件的提示用FrontEnd来转换txt文本成label文件，再进行语音合成。

对于英文语音合成，merlin中需要首先通过Duration模型确定音素的发音时间，然后根据声学模型合成完整的语音。

5.Merlin的训练网络

*内容来源：*[*Merlin: An Open Source Neural Network Speech Synthesis System *](http://ssw9.net/papers/ssw9_PS2-13_Wu.pdf)

Merlin一共提供了4类神经网络用于HMM模型的训练，分别是

- 前馈神经网络


- 基于LSTM的RNN网络


- 双向RNN网络


- 其他变体（如blstm）

1）前馈神经网络

前馈神经网络是最简单的网络类型。有了足够多层，该网络结构就会变成深度神经网络（DNN），网络的大致结果如下：

![img](http://img.blog.csdn.net/20170218154932245?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbHVqaWFuMTk4OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

通过若干隐藏单元层，用输入来预测输出，每个隐藏单元执行一个非线性函数，如下：

![img](http://img.blog.csdn.net/20170218152640868?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbHVqaWFuMTk4OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

其中H(.)是隐藏层的非线性激活函数，W(xh)和W(hy)是权重矩阵，b(h)和b(y)是偏置向量，W(hy).h(t)是一个线性回归，用于从之前隐藏层的激活来预测目标特征。

我们可以利用网络特性（节点足够多的情况下可以以任意精度接近非线性网络）来训练我们的声码器参数，通过以语言特征为输入，声码器参数维目标输出的训练过程，得到一个较为良好的网络来预测我们所需要的声码器参数。

2）基于LSTM的RNN网络

在DNN网络中，语言特征被逐帧映射到声码器特征中，没有考虑到语言的连续性，以所得的参数合成的声音缺少自然性和流畅性。自卷积网络RNN可以解决这个问题，长短时记忆单元是实现RNN网络的常用方法，该单元的结构如下

![img](http://img.blog.csdn.net/20170218162758450?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbHVqaWFuMTk4OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

从上图不难看出，该单元结构较为复杂，但我们可以大致将其分割为输入门（Input gate），遗忘门（Forgret gate），记忆单元（cell）以及输出门（Output gate），该输入单元将输入信号和前一时间实例的隐藏激活传送通过输入门，遗忘门，记忆单元和输出门来产生激活。内部关系可以由以下公式进行表征：

![img](http://img.blog.csdn.net/20170218161029805?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbHVqaWFuMTk4OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

其中i(t),f(t)和o(t)分别是输入门，遗忘门，输出门。c(t)是所谓的记忆单元，h(t)是t时刻的隐藏激活，x(t)是输入信号，W(*)和R(*)是应用到输入和递归隐藏单元的权重矩阵。p(*)和b(*)是peep-hole连接和偏置。δ(.)和g(.)是sigmoid和双曲线正切激活函数。

当然RNN网络并不只局限于以上的网络单元，可以有很多的变种与变化，根据不同的需求我们采用不同的网络结构。

3）双向RNN网络

在单向RNN网络中，只有过去时间的样本上下文信息在网络训练时被考虑，但在现实生活中，语言特征并不只和先前相关，同样也存在着后续相关的问题，为了解决这一问题，双向RNN网络应运而生，双向RNN网络的结构单元可以由如下公式表征：

![img](http://img.blog.csdn.net/20170218163217506?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvbHVqaWFuMTk4OQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

其中[前向h(t)]和[后向h(t)]分别是来自正反方向的隐藏激活，W(x[前向h(t)])和W(x[后向h(t)])是输入信号的权重矩阵，R([前向h(t)][前向h(t)])和R([后向h(t)][后向h(t)])分别是前后向的递归矩阵。

4）在Merlin中也存在在其他的网络结构及变种

这里就不一一阐述了。

# 英文语音合成理论研究

如“基于HMM模型的语音合成系统”图片所示，英文语音合成分成训练和合成阶段，而训练阶段又由以下几个步骤组成

- 文本分析——对应FrontEnd


- HMM模型聚类——对应Question File


- 音频特征参数提取——对应Vocoder


- HMM模型训练——对应HMM模型训练

合成阶段则包括

- HMM解码——对应HMM模型训练


- 文本分析——对应FrontEnd


- 语音合成——对应Vocoder

由于网上已有大量关于HMM模型的介绍，而且由于篇幅所限，本文不对HMM模型进行详细的说明。

![img](https://images-cdn.shimo.im/88ifmqdAkocO0FZG/image.png!thumbnail)

训练过程中的文本分析和音频特征参数提取

## 1 前端FrontEnd 

语音合成前端（Front-End）实际上是一个文本分析器，属于 NLP(Natural Language Processing)的研究范畴，其目的是

 – 对输入文本在语言层、语法层、语义层的分析

 – 将输入的文本转换成层次化的语音学表征

 • 包括读音、分词、短语边界、轻重读等 

 • 上下文特征（context feature）

（1）Label的分类

在Merlin中，Label有两种类别，分别是

- **state align **（使用HTK来生成，以发音状态为单位的label文件，一个音素由几个发音状态组成）


- **phoneme align（**使用Festvox来生成，以音素为单位的label文件）

（2）txt to utt

文本到文本规范标注文件是非常重要的一步，这涉及自然语言处理，对于英文来说，具体工程实现可使用Festival，参见：

Creating .utt Files for English [http://www.cs.columbia.edu/~ecooper/tts/utt_eng.html](http://www.cs.columbia.edu/~ecooper/tts/utt_eng.html)

Festival 使用了英文词典，语言规范等文件，用最新的EHMM alignment工具将txt转换成包含了文本特征（如上下文，韵律等信息）的utt文件

（3）utt to label    

在获得utt的基础上，需要对每个音素的上下文信息，韵律信息进行更为细致的整理，对于英文的工程实现可参见：

Creating Label Files for Training Data [http://www.cs.columbia.edu/~ecooper/tts/labels.html](http://www.cs.columbia.edu/~ecooper/tts/labels.html)

label文件的格式请参见：[http://www.cs.columbia.edu/~ecooper/tts/lab_format.pdf](http://www.cs.columbia.edu/~ecooper/tts/lab_format.pdf)

（4）label to training-data（HMM模型聚类）TODO

由于基于上下文信息的HMM模型过于庞大，有必要对HMM模型进行聚类，即使用问题集Question file.（可以参考决策树聚类[http://blog.csdn.net/quhediegooo/article/details/61202901](http://blog.csdn.net/quhediegooo/article/details/61202901)）（这个Question sets目测可以看HTS的文档来获得进一步的解释）

Question file 的解释：

The questions in the question file will be used to convert the full-context labels into binary and/or numerical features for vectorization. It is suggested to do a manual selection of the questions, as the number of questions will affect the dimensionality of the vectorized input features.

![img](https://images-cdn.shimo.im/Ht0PfKJc8WE5SB2Z/image.png!thumbnail)

在Merlin目录下，merlin/misc/questions目录下，有两个不同的文件，分别是：

questions-radio_dnn_416.hed        questions-unilex_dnn_600.hed

查看这两个文件，我们不难发现，questions-radio_dnn_416.hed定义了一个416维度的向量，向量各个维度上的值由label文件来确定，也即是说，从label文件上提取必要的信息，我们可以很轻易的按照定义确定Merlin训练数据training-data；同理questions-unilex_dnn_600.hed确定了一个600维度的向量，各个维度上的值依旧是由label文件加以确定。

## 2 声码器Vocoder

Merlin中自带的vocoder工具有以下三类：Straight，World，World_v2

这三类工具可以在Merlin的文件目录下找到，具体的路径如下merlin/misc/scripts/vocoder

在介绍三类vocoder之前，首先说明几个概念：

**MGC特征**：通过语音提取的MFCC特征由于维度太高，并不适合直接放到网络上进行训练，所以就出现了MGC特征，将提取到的MFCC特征降维（在这三个声码器中MFCC都被统一将低到60维），以这60维度的数据进行训练就形成了我们所说的MGC特征

**BAP特征**： Band Aperiodicity的缩写

LF0：LF0是语音的基频特征

Straight

音频文件通过Straight声码器产生的是：60维的MGC特征，25维的BAP特征，以及1维的LF0特征。

通过 STRAIGHT 合成器提取的谱参数具有独特 特征(维数较高), 所以它不能直接用于 HTS 系统中, 需要使用 SPTK 工具将其特征参数降维, 转换为 HTS 训练中可用的 mgc(Mel-generalized cepstral)参数, 即, 就是由 STRAIGHT 频谱计算得到 mgc 频谱参数, 最后 利用原 STRAIGHT 合成器进行语音合成

World

音频文件通过World声码器产生的是：60维的MGC特征，可变维度的BAP特征以及1维的LF0特征，对于16kHz采样的音频信号，BAP的维度为1，对于48kHz采样的音频信号，BAP的维度为5

网址为：[https://github.com/mmorise/World](https://github.com/mmorise/World)

World_v 2

音频文件通过World_v2声码器产生的是：60维的MGC特征，5维的BAP特征以及1维的LF0特征，现World_v2版本还处在一个测试的阶段，存在着转换过程不稳定这一类的问题

3 训练模型——Duration和声学模型

语音合成和语音识别是一个相反的过程, 在语音 识别中, 给定的是一个 HMM 模型和观测序列(也就是 特征参数, 是从输入语音中提取得到), 要计算的是这 些观测序列对应的最有可能的音节序列, 然后根据语 法信息得到识别的文本. 而在合成系统中, 给定的是 HMM 模型和音节序列(经过文本分析得到的结果), 要 计算的是这些音节序列对应的观测序列, 也就是特征 参数.  

HTS的训练部分的作用就是由最初的原始语料库经过处理和模型训练后得到这些训练语料的HMM模型[5]。建模方式的选择首先是状态数的选择,因为语音的时序特性,一个模型的状态数量将影响每个状态持续的长短,一般根据基元确定。音素或半音节的基元,一般采用5状态的HMM;音节的基元一般采用10个状态。在实际的建模中,为了模型的简化,可以将HMM中的转移矩阵用一个时长模型(dur)替代,构成半隐马尔可夫模型HSMM hidden semi-Markov Model。用多空间概率分布对清浊音段进行联合建模,可以取得很好的效果。HTS的合成部分相当于训练部分的逆过程,作用在于由已经训练完成的HMM在输入文本的指导下生成参数,最终生成语音波形。具体的流程是:

(1)通过一定的语法规则、语言学的规律得到合成所需的上下文信息,标注在合成label中。

(2)待合成的label经过训练部分得到的决策树决策,得到语境最相近的叶结点HMM就是模型的决策。

(3)由决策出来的模型解算出合成的基频、频谱参数。根据时长的模型得到各个状态的帧数,由基频、频谱模型的均值和方差算出在相应状态的持续时长帧数内的各维参数数值,结合动态特征,最终解算出合成参数。

(4)由解算出的参数构建源-滤波器模型,合成语音。源的选取如上文所述:对于有基频段,用基频对应的单一频率脉冲序列作为激励;对于无基频段,用高斯白噪声作为激励

HSMM半隐马尔可夫模型的解释如下

A hidden semi-Markov model (HSMM) is a statistical model with the same structure as a [hidden Markov model](https://en.wikipedia.org/wiki/Hidden_Markov_model) except that the unobservable process is [semi-Markov](https://en.wikipedia.org/wiki/Semi-Markov_process) rather than [Markov](https://en.wikipedia.org/wiki/Markov_process). This means that the probability of there being a change in the hidden state depends on the amount of time that has elapsed since entry into the current state. This is in contrast to hidden Markov models where there is a constant probability of changing state given survival in the state up to that time

# 英文语音合成实现

Merlin自带英文语音合成，所以实现起来相对简单。你只需要训练Merlin自带的slt_arctic_full音频文件，安装FrontEnd，即可合成拥有基准效果的英文语音。

具体步骤如下参见：[https://github.com/Jackiexiao/merlin/blob/master/manual/Create_your_own_label_Using_Festival.md](https://github.com/Jackiexiao/merlin/blob/master/manual/Create_your_own_label_Using_Festival.md)

# 中文语音合成理论研究

0 汉语语言特点分析

汉语是单音节，声调语言。

**音节**

音节是由音素构成的。如啊”（ā）（1个音素），“地”（dì）（2个音素），“民”（mín）（3个音素）。

音节示例：如“建设”是两个音节，“图书馆”是三个音节，“社会主义”是四个音节。汉语音节和汉字基本上是一对一，一个汉字也就是一个音节。

音节包含了声母、韵母、音调三个部分。

声母： 声母指音节开头的辅音，共有23个。如dā（搭）的声母是d

韵母： 韵母指音节里声母后面的部分，共38。jiǎ（甲）的韵母是iǎ

音节： 声调指整个音节的高低升降的变化。普通话里dū（督）、dú（毒）、dǔ（赌）、dù（度）

根据《现代汉语词典》，汉语标准音节共 418 个

**声调**

普通话的孤立音节有阴平、阳平、上声、去声和轻声五种音调

1 前端FrontEnd 

中文语音前端的主要步骤如下，详见论文：面向汉语统计参数语音合成的标注生成方法

![img](https://images-cdn.shimo.im/TwA6MgSoal8WzWjG/image.png!thumbnail)

中文label文件

可参考论文：面向汉语统计参数语音合成的标注生成方法 中提高的标注方法

label文件见：[https://github.com/Jackiexiao/merlin/blob/master/manual/chinese_label.md](https://github.com/Jackiexiao/merlin/blob/master/manual/chinese_label.md)

2 声码器Vocoder

声码器与语言种类无关，因而这里使用与英文相同的声码器。

# 中文语音合成实现

由于中文语音合成系统复杂，工作琐碎繁多，需要花费大量的时间，在有限的一个月时间里，学习语音合成的基础知识占据了大多数的时间，因此笔者在提交报告为止尚未实现中文语音合成，在这里简要谈谈我们已经完成的步骤以及后续所需要做的工作。

1 语料库

我们使用了指导老师提供的语料库casia。事实上，网络上有清华大学免费提供的语料库，可见：[http://www.cslt.org/news.php?title=News-20160204](http://www.cslt.org/news.php?title=News-20160204)，该语料库质量上乘，提供了用于文本处理的语言模型和词典（Language model and lexicon)，以音素为单位对文本进行了标记，因此事实上清华大学的语料库要比老师提供的语料库更为完成且支持更为全面。因此后续工作可以先用清华大学的语料库进行中文语音合成的训练。

除了上面谈到的中文语音公开数据集，还有下面两个数据集

gale_mandarin: 中文新闻广播数据集(LDC2013S08, LDC2013S08)

hkust: 中文电话数据集(LDC2005S15, LDC2005T32)

2 语料库的规范处理

不同的语料库中的格式不尽相同，因此，首先要对语料库中的标记数据进行规范化，变成FrontEnd可以采用的数据。以指导老师提供的casia语料库为例，我们编写了脚本cmu_transf.py 

lab_transf.py

对语料库进行处理

**处理前：**

```
[SampleRate = 16000]
[ElementNum = 33]
1        SIL
[start = 0]
[end = 1904]
2        jiang1
[start = 1904]
[end = 6363]
[peaknum = 76]
[2918 2962 3006 3050 3094 3138 3183 3227 3272 3317 3363 3408 3453 3499 3544 3589 3635 3681 3726 3772 3818 3863 3909 3955 4001 4047 4093 4139 4185 4231 4277 4323 4369 4415 4462 4508 4554 4600 4646 4692 4738 4783 4829 4875 4921 4966 5012 5057 5102 5148 5193 5238 5283 5328 5374 5419 5464 5509 5554 5599 5644 5688 5733 5777 5821 5865 5910 5954 5999 6044 6089 6135 6181 6227 6274 6322 ]
3        hai2
[start = 6363]
[end = 9935]
[peaknum = 28]
[7838 7901 7965 8029 8093 8159 8228 8299 8375 8454 8537 8622 8710 8798 8886 8974 9061 9147 9231 9315 9396 9475 9551 9625 9699 9772 9844 9917 ]
```


**处理后（只取部分）：**

文件一：

```
0.119 SIL
0.398 jiang1
0.621 hai2
0.829 shi5
1.090 lao3
1.124 SIL
1.229 de5
1.501 la4
1.713 SIL
```

文件二：

```
( casia_00001 "jiang1 hai2 shi5 lao3 de5 la4 yi2 wei4 jiao1 she4 hui4 xue2 de5 lao3 shi1 da3 le5 ge4 you1 mo4 de5 bi3 yu4 shuo1" )
( casia_00002 "yi1 qian1 si4 bai3 er4 shi2 yi1 yu3 si1 bu4 xi1 xi1 sheng1 zi4 you2 yi3 tu2 gou3 an1 de5 ren2" )
```

3 中文FrontEnd

考虑基于Festvox自行开发中文的FrontEnd，其中结合上述的中文语音合成理论研究以及网络上开源的中文文本分析器。这一步最为困难。

# 参考文献

1 论文部分

**主要参考论文**

范会敏, 何鑫. 中文语音合成系统的设计与实现[J]. 计算机系统应用, 2017(2):73-77.

郝东亮, 杨鸿武, 张策,等. 面向汉语统计参数语音合成的标注生成方法[J]. 计算机工程与应用, 2016, 52(19):146-153.

Merlin: An Open Source Neural Network Speech Synthesis System 

英文：[http://ssw9.net/papers/ssw9_PS2-13_Wu.pdf](http://ssw9.net/papers/ssw9_PS2-13_Wu.pdf)

中文：[http://blog.csdn.net/lujian1989/article/details/56008786](http://blog.csdn.net/lujian1989/article/details/56008786)

**其他论文**

侯亭武. 基于语料库的中文语音合成技术研究[D]. 华中科技大学, 2015.张德良. 深度神经网络在中文语音识别系统中的实现[D]. 北京交通大学, 2015.蔡明琦. 融合发音机理的统计参数语音合成方法研究[D]. 中国科学技术大学, 2015.车浩. 汉语语音合成韵律预测技术研究[J]. 2015.

张斌, 全昌勤, 任福继. 语音合成方法和发展综述[J]. 小型微型计算机系统, 2016, 37(1):186-192.

**相关论文（未读）**

基于声韵母基元的嵌入式中文语音合成系统[http://www.speakit.cn/Group/file/Embeded_SP05.pdf](http://www.speakit.cn/Group/file/Embeded_SP05.pdf)

可變速中文文字轉語音系統

[http://www.aclweb.org/anthology/O10-1016HMM-based](http://www.aclweb.org/anthology/O10-1016HMM-based) 

Mandarin Singing Voice Synthesis UsingTailored Synthesis Units and Question Sets [https://aclweb.org/anthology/O/O13/O13-5005.pdf](https://aclweb.org/anthology/O/O13/O13-5005.pdf)

基于深度神经网络的汉语语音合成的研究

[http://www.jsjkx.com/jsjkx/ch/reader/view_abstract.aspx?file_no=20156A018&flag=1](http://www.jsjkx.com/jsjkx/ch/reader/view_abstract.aspx?file_no=20156A018&flag=1)

Statistical Analysis for Standard Chinese Syllables and Phoneme System

孙敬伟. 统计参数语音合成中的关键技术研究[D]. 中国科学院声学研究所, 2009.

 [http://159.226.59.140/handle/311008/556](http://159.226.59.140/handle/311008/556)

顾香. 面向统计参数语音合成的方言文本分析的研究[D]. 西北师范大学, 2014.

[http://xueshu.baidu.com/s?wd=paperuri%3A%28b9485b772c7ab7487d7264d119052f94%29&filter=sc_long_sign&tn=SE_xueshusource_2kduw22v&sc_vurl=http%3A%2F%2Fwww.doc88.com%2Fp-1166631413866.html&ie=utf-8&sc_us=10261735401629849101](http://xueshu.baidu.com/s?wd=paperuri%3A%28b9485b772c7ab7487d7264d119052f94%29&filter=sc_long_sign&tn=SE_xueshusource_2kduw22v&sc_vurl=http%3A%2F%2Fwww.doc88.com%2Fp-1166631413866.html&ie=utf-8&sc_us=10261735401629849101)

温正棋. 统计参数语音合成中语音参数化方法研究[J]. 2013.

（暂时找不到原文）

2 工程实现教程部分

Getting started with the Merlin Speech Synthesis Toolkit

[http://jrmeyer.github.io/merlin/2017/02/14/Installing-Merlin.html](http://jrmeyer.github.io/merlin/2017/02/14/Installing-Merlin.html)

Merlin官方教程（正在建设中）

[http://104.131.174.95/Merlin/dnn_tts/doc/build/html/](http://104.131.174.95/Merlin/dnn_tts/doc/build/html/)

**Columbia University TTS manual

[http://www.cs.columbia.edu/~ecooper/tts/](http://www.cs.columbia.edu/~ecooper/tts/)

国人使用merlin的经验分享

[http://shartoo.github.io/merlin-tts/](http://shartoo.github.io/merlin-tts/)

HTS tutorial 

[http://hts.sp.nitech.ac.jp/?Tutorial](http://hts.sp.nitech.ac.jp/?Tutorial)

Festvox教程（利用wav 和标记数据创造label）

[http://festvox.org/bsv/](http://festvox.org/bsv/)

speech.zone build-your-own-dnn-voice

[http://www.speech.zone/exercises/build-your-own-dnn-voice/](http://www.speech.zone/exercises/build-your-own-dnn-voice/)

3 代码部分

Merlin语音合成系统 Github：[https://github.com/CSTR-Edinburgh/merlin](https://github.com/CSTR-Edinburgh/merlin)

Festvox

HTK [http://htk.eng.cam.ac.uk](http://htk.eng.cam.ac.uk/) 

HTS

SPTK

World

4 语音识别/合成基础知识

上下文相关的GMM-HMM声学模型

[http://www.cnblogs.com/cherrychenlee/p/6780460.html](http://www.cnblogs.com/cherrychenlee/p/6780460.html)

知乎-语音识别的技术原理是什么？

[https://www.zhihu.com/question/20398418](https://www.zhihu.com/question/20398418)

A beginners’ guide to statistical parametric speech synthesis

英文：[http://www.cstr.ed.ac.uk/downloads/publications/2010/king_hmm_tutorial.pdf](http://www.cstr.ed.ac.uk/downloads/publications/2010/king_hmm_tutorial.pdf)

中文：[https://shartoo.github.io/texttospeech/](https://shartoo.github.io/texttospeech/)

语音产生原理与特征参数提取

[http://blog.csdn.net/u010451580/article/details/51178190](http://blog.csdn.net/u010451580/article/details/51178190)

台湾-语音信号处理教程（包含了语音合成教程）

[http://www.mirlab.org/jang/books/audioSignalProcessing/](http://www.mirlab.org/jang/books/audioSignalProcessing/)

浅谈语音识别基础

[http://www.jianshu.com/p/a0e01b682e8a](http://www.jianshu.com/p/a0e01b682e8a)

语音识别与TTS相关博客

[http://blog.csdn.net/zouxy09/article/category/1218766](http://blog.csdn.net/zouxy09/article/category/1218766)

[English tutorial] for Chinese Spoken Language Processing

[http://iscslp2016.org/slides.html](http://iscslp2016.org/slides.html)

中文语音合成基本概念

[http://staff.ustc.edu.cn/~zhling/Course_SSP/slides/Chapter_13.pdf](http://staff.ustc.edu.cn/~zhling/Course_SSP/slides/Chapter_13.pdf)

# 术语表

Front end 前端

vocoder 声音合成机（声码器）

MFCC 参见[http://blog.csdn.net/zouxy09/article/details/9156785/](http://blog.csdn.net/zouxy09/article/details/9156785/)

受限波尔曼兹机

bap band aperiodicity 非周期性 [http://blog.csdn.net/xmdxcsj/article/details/72420051](http://blog.csdn.net/xmdxcsj/article/details/72420051)

ASR：Automatic Speech Recognition自动语音识别

AM：声学模型

LM：语言模型

HMM：Hiden Markov Model 输出序列用于描述语音的特征向量，状态序列表示相应的文字

HTS：HMM-based Speech Synthesis System语音合成工具包

HTK：Hidden Markov Model Toolkit 语音识别的工具包

自编码器

SPTK：speech signal precessing toolkit

SPSS : 统计参数语音合成statistical parametric speech synthesis

# 拓展

1 wavenet

DeepMind基于深度学习的原始语音生成模型－WaveNet

英文：[https://deepmind.com/blog/wavenet-generative-model-raw-audio/](https://deepmind.com/blog/wavenet-generative-model-raw-audio/)

中文：[http://www.w2bc.com/article/171751](http://www.w2bc.com/article/171751)

谷歌WaveNet源码详解[https://zhuanlan.zhihu.com/p/24568596](https://zhuanlan.zhihu.com/p/24568596)

基于wavenet的中文语音合成

[https://github.com/auzxb/Chinese-speech-to-text](https://github.com/auzxb/Chinese-speech-to-text)

A TensorFlow implementation of DeepMind's WaveNet paper

[https://github.com/ibab/tensorflow-wavenet](https://github.com/ibab/tensorflow-wavenet)

2 中文语音识别-开源

开源中文语音识别：[https://github.com/kaldi-asr/kaldi/tree/master/egs/thchs30](https://github.com/kaldi-asr/kaldi/tree/master/egs/thchs30)

3 百度DeepVoice

理解百度Deep Voice的语音合成原理

[http://mt.sohu.com/20170316/n483703904.shtml](http://mt.sohu.com/20170316/n483703904.shtml)

4 微软的中文语音研究包

[https://www.microsoft.com/en-us/research/publication/speech-lab-in-a-box-a-mandarin-speech-toolbox-to-jumpstart-speech-related-research/](https://www.microsoft.com/en-us/research/publication/speech-lab-in-a-box-a-mandarin-speech-toolbox-to-jumpstart-speech-related-research/)

# TODO List

 - [ ] 清华大学开源的语音包：[http://www.cslt.org/news.php?title=News-20160204](http://www.cslt.org/news.php?title=News-20160204)

 - [ ] \(optional)微软的中文语音包

 - [ ] 中科院的教程[http://iscslp2016.org/slides.html](http://iscslp2016.org/slides.html)

 - [ ] FrontEnd的学习研究

 - [ ] 阅读文章：中科院的统计参数语音合成中的关键技术研究 [http://159.226.59.140/handle/311008/556](http://159.226.59.140/handle/311008/556)

 - [ ] 阅读论文：面向统计参数语音合成的方言文本分析的研究 [http://xueshu.baidu.com/s?wd=paperuri%3A%28b9485b772c7ab7487d7264d119052f94%29&filter=sc_long_sign&tn=SE_xueshusource_2kduw22v&sc_vurl=http%3A%2F%2Fwww.doc88.com%2Fp-1166631413866.html&ie=utf-8&sc_us=10261735401629849101](http://xueshu.baidu.com/s?wd=paperuri%3A%28b9485b772c7ab7487d7264d119052f94%29&filter=sc_long_sign&tn=SE_xueshusource_2kduw22v&sc_vurl=http%3A%2F%2Fwww.doc88.com%2Fp-1166631413866.html&ie=utf-8&sc_us=10261735401629849101)

