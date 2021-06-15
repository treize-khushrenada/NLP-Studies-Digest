nlp self-learn

***On Topic Modeling***

Nearest neighbor algorithm can be a method for topic modeling

Sentence Embedding is the collective name for a set of techniques in natural language processing where sentences are mapped to vectors of REAL NUMBERS.

GuidedLDA is a way to let developer defines the 'seed topic' first, then it will start modelling the topics in the corpus in that paradigm.

Top2Vec is an also for topic modelling and semantic search. It automatically detects topics present in text and generates jointly embedded topic, document and word vectors. We have to train the model first, then we can get the results. It works on short text.

Sequence-to-sequence learning: It is about training models to convert sequences(e.g. text strings) from one domain to another. Examples include translating English to French, or Q&A (from question to answer) 

KILT: a project by facebook AI lab, aimed to provide a unified platform for different research teams to continuously develop NLP knowledge models with a wikipedia snapshot. "unified benchmark to help AI researchers build models that are better able to leverage real-world knowledge to accomplish a broad range of tasks."


**Machine Learning versus Deep Learning**

Machine learning is a branch of AI that deals with the development of algorithms that can learn to perform tasks automatically based on a large number of examples, while deep learning refers to the branch of machine learning that is based on **artificial neural network architectures**.  most recently, deep learning has also been frequently used to build nlp apps.

When ML is trying to 'learn' from training data, it would create 'features', which is a numeric representation, of the training data, and use that to learn the patterns in those examples. 

Machines learning algorithms can be grouped into 3 paradigms:  1 supervised learning, 2 unsupervised learning, and 3 reinforcement learning. for 1, it will try to learn the mapping function from input to output given a large number of examples in the form of **input-output pairs** (the 'training data', the 'labels' or 'ground truth' ). for 2, it refers to a set of machine learning methods that aim to find hidden patterns in given input data without any reference output, so in contrast, unsupervised learning works with large collection of unlabelled data. another common case is between 1 and 2- semi-supervised learning, where we have a small labeled dataset and a large unlabelled dataset, using both sets to learn the task at hand. 3 deals with methods to learn tasks via trial and error and is characterised by the absence of neither labeled or unlabelled data in large quantities. the learning is done in a self-contained environment and improves via feedback. this form of learning is not common in applied NLP yet.

**Transformers and their friends**

Transformers are the latest entry in the league of deep learning models for NLP. 

It can model the textual context, but not in a sequential manner. Given a word in the input, it looks at the words AROUND it (which is known as **self-attention**)  and represent each word with respect to the context.

Self-attention

A self-attention module takes in n inputs and return n outputs, while is allows the inputs to interact with each other (i.e. 'self') and find out 'who' they should pay more 'attention' to, then generates an output with the aggregated interactions and attention scores. ++check nlp folder/ attention file for detailed illustration, and a use case of Softmax in the process.

“multi-headed” attention: It expands the model’s ability to focus on different positions. (because it could be dominated by the actual word itself if single head). In multi heads setting, each of these sets is randomly initialized.

Transformer-based architectures are primarily used in modelling language understanding tasks. they trust entirely on self-attention mechanisms to draw global dependencies between inputs and outputs. They are the the architecture of the 'xxxBERT's.

The abstraction that is common to all the encoders is that they receive a list of vectors each of the size 512
In the bottom encoder that would be the word embeddings, but in other encoders, it would be the output of the encoder that’s directly below. The size of this list is hyperparameter we can set – basically it would be the length of the longest sentence in our training dataset.

Word Embedding/ Positional Encoding 

Other than word vectorising the word in the sequence, there's encoding to include positional information of the word- Positional Encoding accounts for the order of the words in the input sequence, It helps determine the position of each word, or the distance between different words in the sequence. It combines with word embeddings as the input. As for the parameters and assigning mechanism in positional encoding, it uses sine/ cosine values (http://jalammar.github.io/images/t/transformer_positional_encoding_large_example.png)

A word's own path in each encoder:
word-> embedding(vector)-> [encoder_sublayer1 (self-attention)]-> vector-> [encoder_sublayer2(feed forward neural network)]->

The various paths can be executed in parallel while flowing through the feed forward neural network(ffnn) layer.
In each sublayer, there's a residual connection named 'layer-normalisation' step. The LayerNorm() operation combine input embeddings and output embeddings of the sublayer. >> more details to be found out.

Feed Forward Neural Network (ffnn):

**>> more details to be found out.**

the commonly heard 'simple' neural network. Also known as Artificial Neural Network (ANN)
Perceptron('neuron')


[encoders chain] -> [decoders chain]


Decoders

Output from the last encoder in the encoders chain is transformed into a set of attention vectors, K and V, and proceed withe the decoding phase, they will be used by each decoder in their 'encoder-decoder attention layer', helping the decoder to focus. **>> more details to be found out. **

Also positional encoding was embedded to the decoder inputs.

For the next Time Step **>> more details to be found out** 
the output of previous step is fed to the bottom decoder, and the decoders bubble up the results.  After decoding time step n, the output word x gets out from the decoder chain, and then feeding to the bottom in the decoder chain as one of the previous outputs.

In the decoder, the self-attention layer is only allowed to attend to earlier positions, all the 'future positions' are masked (by setting them to -inf, before the softmax step in the self-attention sublayer.). In other words, the layer here only focuses to the output words that were 'translated' already, as the next word(s) are something we deduce in the future). While each position in the encoder can attend to all positions in the previous layer of the encoder.

i.e. Self-attention layers in the decoder allow each position in the decoder to attend to all positions in the decoder **up to and including that position. **

After the decoding process, the output is a vector of floats. We turn it to a word with a linear layer and a softmax layer.

The linear layer

It is a fully connected neural network, it projects a vector called `logits vector`, which is a much larger vector. the size of the vector is the size of the number of unique words in the training dataset, each element corresponding to the score of a unique word.

The softmax layer

From the scores obtained in the linear layer, softmax layer will transcode them to probabilities. the cell with the highest probability is chosen, and the word associated with it is produced as the output of this time step.


Training stage

During training, an untrained model would go through the pipeline. Since we are training it on a labeled training dataset, we can compare its output with the actual correct output.

'one-hot encoding'

Once we define our output vocabulary, we use a vector of the same width to indicate each word vocabulary. For the actual correct output, we can mark the P(word) in the vector as 1, while other unique words are 0. We can then use that to compare the output from the softmax layer.

loss function





What is pre-training and fine-tuning? (From BERT paper)

	After you have trained a model for a previous task (on data set A), you are now interested in training a network to perform a new task on a different data set B. Instead of starting from training with randomly initialized weights, you can use the weights (some of the parameters) you saved from the previous training on the same network, as the 'initial weight values' for your new experiment. 

	The parameters and activities involved is 'pre-training', it gives the network a head start as if 'it has seen the data before'. You then train the model on 𝐵, which is called fine-tuning. Using a pre-trained network generally makes sense if both tasks or both datasets have something in common. The bigger the gap, the less effective pre-training will be.

	This is one form of transfer learning. So you can transfer some of the knowledge obtained from dataset 𝐴 to dataset 𝐵.

What is ablation (消融) studies? (From BERT paper)

	你朋友说你今天的样子很帅，你想知道发型、上衣和裤子分别起了多大的作用，于是你换了几个发型，你朋友说还是挺帅的，你又换了件上衣，你朋友说不帅了，看来这件衣服还挺重要的。

What are sentence embeddings and paragraph embeddings? (From BERT paper)

	Word embedding: representation of words for text analysis, typically in the form of a real-valued vector that encodes the meaning of the word such that the words that are closer in the vector space are expected to be similar in meaning.

What is 'downstream task'? (From BERT paper)

	Downstream tasks is what the field calls those supervised-learning tasks that utilize a pre-trained model or component

What is Natural Language Inference?
Natural language inference is a task of asking nlp models for determining the logical relationship between 2 sets of statements- a "premise" and a "hypothesis". 

E.g. given a premise "A man inspects the uniform of a figure in some East Asian country.", and hypothesis is "The man is sleeping.", what is the relationship between the two statements? There are 3 labels for the model to tag on: "contradiction", "neutral", and "entailment". In the example, the label is "contradiction".


- Bert

	BERT is a deeply bidirectional, unsupervised language representation, pre-trained using only a plain text corpus. Context-free models such as word2vec or GloVe generate a single word embedding representation for each word in the vocabulary, where BERT takes into account the context for each occurrence of a given word. For instance, whereas the vector for "running" will have the same word2vec vector representation for both of its occurrences in the sentences "He is running a company" and "He is running a marathon", BERT will provide a contextualized embedding that will be different according to the sentence.

Why unidirectional language model to learn general language representations is bad for questions and answers where its crucial to incorporate context from both directions.

why is multi-headed attention model good with the 'subspace' and 'not dominated by the the actual word itself"?

'in OpenAI GPT, the authors use a left-to-right architecture, where every token can only attend to previous tokens in the self-attention layers of the Transformers'

alleviates

training sentence representations- what are the objectives and how do they work

what is a supervised downstream task

the annotated transformer

why traditional left to right / right to left model / concatenation of both models?
how do models predict a word a a 'multi-layer' context?
why the bidirectional conditioning will allow a word to 'see itself'?
the relationship between tokens, embeddings and vectors

cross entropy loss:
	香农提出了熵的定义：无损编码事件信息的最小平均编码长度。


what are 'all parameters' to initialise end-task model versus 'only sentence embeddings are transferred' in previous works?
Inference step(at prediction time)

self-attention mechanism in the Transformer

bidirectional cross attention

- Beyond Accuracy

What is 'Generalisation' in NLP training
What is 'held-out' datasets
prediction invariance
duplicated question detection
machine comprehension
MFT/ INV/ DIR

- A Neural attention model for sentence summarization
What are the deletion based  sentence compression techniques
Paraphrasing
Generalization
Re-ordering

Nlp algorithm symbols how to read and understand
Attention based encoder

- Distributed representations of sentences and documents
bag of n grams and bag of words

N-grams are simply all combinations of adjacent words or letters of length n that you can find in your source text. For example, given the word fox, all 2-grams (or “bigrams”) are fo and ox. You may also count the word boundary – that would expand the list of 2-grams to #f, fo, ox, and x#, where # denotes a word boundary.

You can do the same on the word level. As an example, the hello, world! text contains the following word-level bigrams: # hello, hello world, world #.


Stochastic gradient descent and backpropagation  
Prediction time
Convergence stage
Using previous word vectors to form  the input of a neural network
A sequence of training words
Huffman tree
And neural language models

Support vector machines/ standard classifier

BLEU

Captures the amount of n-gram overlap between the output sentence and the reference ground truth sentence. It has many variants. Mainly used in machine-translation tasks. Recently adapted to other text-generation tasks, such as paraphrase generation and text summarization.

Get to the point:
pointer-generator model

	weighting P(gen) and 1-P(gen) and decide to use the word from attention distribution, or vocabulary distribution 

sequel to sequel model
bidirectional LSTM






https://www.linkedin.com/pulse/explanation-attention-based-encoder-decoder-deep-keshav-bhandari
https://twitter.com/quocleix
https://www.groundai.com/project/document-embedding-with-paragraph-vectors/1
https://zh.wikipedia.org/wiki/%E4%BA%A4%E5%8F%89%E7%86%B5
https://bbs.cvmart.net/topics/3939
https://www.jiqizhixin.com/graph/technologies/1786086f-5b63-4eee-b9ed-dad4d64cdc86
https://math.stackexchange.com/questions/386286/why-differentiate-between-subset-and-proper-subset
https://www.mathsisfun.com/sets/symbols.html
https://www.cliffsnotes.com/study-guides/algebra/algebra-i/terminology-sets-and-expressions/quiz-set-theory
https://www.onlinemath4all.com/symbols-used-in-set-theory.html
https://ranko-mosic.medium.com/googles-bert-nlp-5b2bb1236d78
https://martin-thoma.com/ml-glossary/
https://stats.stackexchange.com/questions/193082/what-is-pre-training-a-neural-network
http://jalammar.github.io/illustrated-bert/

Practical NLP book:

4 major building blocks of human language:
- phonemes- the unit within a word (vowels)
- morphemes and lexemes (the composition of a word// grouping of the morphemes is a lexeme, for example 'run' and 'running')
- syntax (the set of rules to construct grammatically correct  sentences out of words and phrases in a language)
- context- semantics (direct meaning of the words and sentences) and pragmatics (adding world knowledge and external context to the conversation)


