## understanding the conceptual differences between vecotirzation and tokenization

in machine learning, we need to provide numerical attributes as `features`, and perform futher mathemitical operation. there are various scenarios, like sentiment analysis, where there is no numerical attribute in the datasets. to 'freaturize' it is to convert those types of signals into numrical form. 

there are multiple ways for us to do conversion. some major methods:

- bag of words
- tf-idf
- word2vec

tokenizatin is also called text segmentation or lexical analysis. the idea is to split the data into small chunk of words. some libraries such as hugging face would then convert those chunks into some ids, and put the real words in a look-up table setup. this is a step in pre-processing.