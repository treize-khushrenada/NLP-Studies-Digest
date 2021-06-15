Maarten Grootendorst published an article on leveraging BERT for topic modeling, it had received quite a lot of positive feedbacks from the community. he then decided to further develop the technique and name it BERTopic.

### what is this about?

there are 2 main parts:
1. BERT Embeddings (with hugging face transformers)
2. class-based TF-IDF

with that we can create dense clusters for easily interpreatble topics and keep important words in topic descriptions.

the author claimed that BERTopic has gotten enough traction and development and it can even 'replace', or at least compliment other conventional techniques.

### packages we need

use the `sentence-transformers` package for generating BERT embeddings. pytorch is suggested to install prior to this.

`pip install sentence-transformers`

as the resulting embeddings have shown to be of high quality and typically work quite well for document-level embeddings.

use `distilbert` to further train the model-->>>

use UMAP to reduce dimensionality

`pip install umap-learn`

use `hdbscan to create clusters

`pip install hdbscan`

### datasets we need

- the author used the 20 newsgroups dataset.
- use `scikit-learn` to download and prepare the data.
- select the `train` subset to speed up training.


### how does it work?

1. create embeddings

2. lower the dimensionality of the embeddings (because many algos are handling them poorly). with UMAP, suggested to reduce the dimensionality to 5, keeping the size of the local neighborhood at 15. (too low dimensionality results in a loss of information while a too high dimensionality results in poorer clustering results.)

3. create clusters with similar topics

4. derive topics from clustered documents. we treat all documents in a single cluter as a single document, then apply tf-idf, in hopes of the score to demonstrate the important words in a certain topic.