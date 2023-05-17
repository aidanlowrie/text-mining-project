# text-mining-project
A repository for our text mining group project (Elliot, Boray, Aidan). 

## Outline
The aim of this project is to create a network that provides a visualisation of the association between biological terms in a corpus of PubMed articles. The steps will be as follows:
1. Preprocess the Data.
- In this step, we will use Biopython and E-Utilities to search for and download the relevant PubMed article abstracts into a text file.
- With the data downloaded, we will preprocess the data to remove stopwords, punctuation and other information deemed irrelevant, using NLTK and SpaCy.
2. Key-Word Extraction.
- We will train a neural network BERT model to extract key words from the preprocessed corpus.
3. Compute Term Associations.
- This step involves finding co-occurrence data of the extracted key words throughout the corpus.
- Then, we will calculate the SPPMI of this data.
4. Creating and Visualising a Network.
- Firstly, we will create an adjacency matrix. 
- Then we will pass this matrix into a NetworkX object, and save this data into a file.
- We will use techniques to analyse the network. 
5. Knowledge Graph? (Need to look into this still). 
- https://www.kaggle.com/code/pavansanagapati/knowledge-graph-nlp-tutorial-bert-spacy-nltk#BERT:-Bidirectional-Encoder-Representations-from-Transformers
- (I have linked to this because I think it might be a useful resource for us to look at.)

Keyword extraction keybert --> plug in biobert to keybert.
--> PubMedBert?
6. Extracting Relations
- The file relation.py has methods to extract relations(that file will be updated soon)
- The main thing it does is iterating over a list of sentences and finding the subject object and the verb and storing it in a pandas dataframe.
- In the end, we want to have a visualised data that shows the important connections that our corpus contains.
- Depending on our end product, we might be able to find important connections in the genetics with qualitative analysis using the clustered data.
- Word similarities, relations, and keywords in the corpus will play a big role in this task.
- Although up to now we've limited ourselves with genetics articles, the methods we provide are generic.
