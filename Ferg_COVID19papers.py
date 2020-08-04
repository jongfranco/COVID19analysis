
# coding: utf-8

# # An Analysis of COVID-19 Research Papers
# ## Robyn Ferg

# ## Introduction
# 
# COVID-19 is a highly infectious, novel respiratory disease. Originating in Wuhan, China in late 2019, COVID-19 has quickly become a worldwide pandemic. Being a novel virus, much is unknown about COVID-19. Numerous academic studies have been and continue to be performed on various aspects of the virus, from vaccine research to social implications. Inspecting each of these studies individually is and onerous task. Instead, we rely on natural language processing techniques to extract meaningful information from a corpus of research papers relating to COVID-19.
# 
# In this report we extract research paper information from an online repository, cluster papers into topics, provide words and papers automatically generated to represent each of those topics, describe an algorithm for providing a summary of each paper, and describe how we could extract papers that may provide breakthroughs in treatment and prevention of COVID-19.

# ## Data
# 
# Our data comes from an online repository of COVID-19 research papers compiled by the MIDAS Network Coordination Center. 
# 
# We load in the xml file and convert to a data frame. We only consider the title of the article, journal name, and abstract. Other variables exist in the xml files, but many have high levels of missingness or are presumably unimportant.

# In[1]:


import requests
from bs4 import BeautifulSoup
import pandas as pd

url = 'https://raw.githubusercontent.com/midas-network/COVID-19/master/documents/mendeley_library_files/xml_files/mendeley_document_library_2020-03-25.xml'
document = requests.get(url)
soup = BeautifulSoup(document.content, 'lxml-xml')

first_child = soup.find('xml')
second_child = first_child.find('records')

# create pandas data frame
data = []
for paper in second_child.contents:
	title = paper.find('titles').find('title').text
	# journal title
	periodical = paper.find('periodical')
	if len(periodical)==0:
		journal = ' '
	else:
		journal = periodical.find('full-title').text
	# abstract
	abstract = paper.find('abstract')
	if abstract is None:
		abstract = ' '
	else:
		abstract = abstract.text
	# append to data
	data.append([title, journal, abstract])

df = pd.DataFrame(data, columns=['title', 'journal', 'abstract_raw'])


# Before we perform any analyses on the data, we first pre-process the data. The following code pre-process the text of the abstracts in the following ways:
# * Removes punctuation
# * Removes stopwords
# * Stems words, i.e. removing suffixes such as -s, -ing, etc.
# * Some abstracts contain competing interest statments, funding statements, etc. that are unimportant to the content of the research being presented. We remove those statements.
# * Remove formatting symbols

# In[2]:


from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

def removeStmts(abstract):
	if 'Competing Interest Statement' in abstract:
		abstract = abstract[0:abstract.find('Competing Interest Statement')]
	return(abstract)

def preProcessingFcn(text, removeWords=list(), stem=True, removeURL=True, removeStopwords=True, 
	removeNumbers=False, removeHashtags=True, removeAt=True, removePunctuation=True):
	ps = PorterStemmer()
	text = text.lower()
	text = re.sub(r"\\n", " ", text)
	text = re.sub(r"&amp", " ", text)
	if removeHashtags==True:
		text = text.replace('#', ' ')
	if removeNumbers==True:
		text=  ''.join(i for i in text if not i.isdigit())
	if removePunctuation==True:
		text = re.sub(r"[,.;@#?!&$]+\ *", " ", text)
	if removeStopwords==True:
		text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
	if len(removeWords)>0:
		text = ' '.join([word for word in text.split() if word not in removeWords])
	if stem==True:
		text = ' '.join([ps.stem(word) for word in text.split()])
	return text

removeWords = ['<p>', '</p>', '<bold>', '</bold>']

df['abstract_noStmt'] = [removeStmts(abst) for abst in df['abstract_raw']]
df['abstract'] = [preProcessingFcn(abst, removeWords) for abst in df['abstract_noStmt']]


# Note that some of the papers have no abstract. We create a word-document matrix from the set of abstracts. That is, we create a matrix $w$ such that $w_{ij}$ gives the number of times word $j$ appears in document $i$. By doing this, we are using a bag-of-words model, where the ordering of words within a text are not taken into account.

# In[3]:


import numpy as np

vectorizer = CountVectorizer(strip_accents='unicode')
textsNew = vectorizer.fit_transform(df['abstract'])
wOriginal = textsNew.toarray()
textsIndex = list(np.where(np.sum(wOriginal, axis=1)>0)[0])
w = wOriginal[textsIndex,:]
words = vectorizer.get_feature_names()


# ## Clustering
# 
# The goal of this section is to automatically group similar papers together. To accomplish this goal, we make use of Latent Dirichlet Allocation, Latent Semantic Analysis, and clustering, each described below.
# 
# Latent Dirichlet Allocation (LDA) is perhaps to most common method of modeling topics in a corpus of texts (Blei, Ng, Jordan 2003). LDA is a hierarchical Bayesian model that assumes each document in a corpus of texts is comprised of a distribution of topics, where a topic is defined as a probability distribution over words. This assumption of each text containing multiple topics is appropriate for our context: paper abstracts may contain, for example, background information, biological breakthrough, and societal impacts. More formally, let $M$ denote the number of documents and $N_i$ denote the number of words in document $i$. Document $i$ has topic distribution $\theta_i$, where $\theta_i\sim Dirichlet(\alpha)$. Each word $w_{ij}$ in document $i$ belongs to topic $z_{ij}\sim\theta_i$, and $w_{ij}\sim Dirichlet(\beta)$, where $\beta$ is the prior on the per-topic word distribution. In this model, only the words $w$ are known, the remaining variables are latent. 
# 
# For a chosen number of $k$ topics, the LDA algorithm assigns each text a $k$-dimensional probability distribution across the $k$ topics. If two texts are in some sense similar, the probability distribution across topics may also be similar. Therefore, we calculate the distance between two texts as the Euclidean distance between the $k$-dimensional probability distributions for the two texts.
# 
# Alternatively, we can calculate distance between texts using Latent Semantic Analysis (LSA) (Dumais et al. 1988). LSA, like LDA, first creates a word-document matrix $w$, where each row of $w$ refers to a single document and each column refers to a single word, where $w_{ij}$ is the number of times word $j$ appears in document $i$. $w$ is a large and often sparse matrix. Dimension reduction, such as singluar value decomposition, is then applied to $w$. The distance beteen two texts $X$ and $Y$ is calculated using cosine similarity: $$dist(X, Y) = 1-\frac{X \cdot Y}{||X||\times||Y||}$$
# 
# Note that both LDA and LSA are bag-of-words methods, so the ordering of words within a document are not taken into account.
# 
# Both distances, as calculate by LDA and LSA, may carry important information. Furthermore, human inspection of words most highly expressed in each topic in LDA and most heaving weighted words in the components using LSA suggest that both methods appear to work fairly well. To hopefully capture advantages of both methods, we create two distance matrices, one for LDA and one for LSA, normalize so all distances are between 0 and 1, and average the two normalized distance matrices.
# 
# After obtaining the final distances between each abstract in the data set of papers, we apply a clustering method to sort each paper into just one latent topic. Since we only have distances between documents as opposed to points in space, we apply k-medoids clustering to the distance matrix.
# 
# The following chunks of code calculate distances between the abstracts using LDA, calculate the distances between abstracts using LSA, average the two distance matrices, and apply k-medoids. For LDA we use 7 clusters, having the maximum coherence score (code omitted from this write-up). For LSA, we reduce $w$ to 14-dimensions. Following LDA, we perform k-medoids clustering using 7 clusters.

# In[4]:


# LDA
from sklearn.decomposition import LatentDirichletAllocation as LDA
from scipy.spatial import distance
import random

random.seed(123)

numberTopics = 7
lda = LDA(n_components=numberTopics, random_state=0)
ldaFit = lda.fit(w)
topicDistributions = lda.transform(w)
distsLDA = distance.cdist(topicDistributions, topicDistributions, 'euclidean')


# In[5]:


# LSA
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

random.seed(234)

n_components = 14
svd_model = TruncatedSVD(n_components=n_components, algorithm='randomized', n_iter=100, random_state=122)
svd_model.fit(w)
wDimReduced = svd_model.fit_transform(w)
distsLSA = 1-cosine_similarity(wDimReduced)


# In[6]:


# Average Distance Matrix
distsLDA_normalized = distsLDA/np.max(distsLDA)
distsLSA_normalized = distsLSA/np.max(distsLSA)

meanDists = (distsLDA_normalized + distsLSA_normalized)/2


# In[7]:


# Clustering: k-medoids
from pyclustering.cluster.kmedoids import kmedoids

random.seed(345)

numberTopics = 7
kmedoids = kmedoids(meanDists, range(numberTopics), data_type='distance_matrix')
kmedoids.process()
clusters = kmedoids.get_clusters()
medoids = kmedoids.get_medoids()


# The distribution of number of papers in each cluster is: 

# In[8]:


[len(clusters[i]) for i in range(numberTopics)]


# The centers of each resulting cluster can be considered to best represent the group.

# In[9]:


medoidClusterComb = list()
for i in range(meanDists.shape[0]):
	for clustNum in range(numberTopics):
		if i in clusters[clustNum]:
			medoidClusterComb.append(clustNum)
            
df_clusters = pd.DataFrame({'cluster':medoidClusterComb}, index=textsIndex)
df_combined = df.join(df_clusters, how='outer')

medoids_index = [textsIndex[i] for i in medoids]

for indx in medoids_index:
    print('Cluster ' + str(df_combined['cluster'][indx]))
    print('Title:')
    print(df_combined['title'][indx])
    print('In:')
    print(df_combined['journal'][indx])
    print('Abstract:')
    print(df_combined['abstract_noStmt'][indx])
    print(' ')
#print([df_all['abstract_noStmt'][i] for i in medoids_lsa])


# To assign word tags to each of the above clusters, we want to find frequent words that are more highly expressed in one cluster compared to the others. For each cluster, we find its distribution across all words and consider only the words that appear more than 100 times in the entire corpus of abstracts. For each of these frequent words we find the mean and standard deviation of the proportion across clusters. We consider a word $i$ a tag for a given cluster $k$ if the frequency of word $i$ in cluster $k$ is at least the mean of word $i$ across clusters plus 1.5 times the standard deviation of word $i$ across all of the clusters. This gives the following tags for each cluster:

# In[10]:


# tags
df_freqs = pd.DataFrame({'word':words, 'total':w.sum(axis=0)})
for i in range(numberTopics):
	wi = w[clusters[i],]
	df_freqs['w'+str(i)] = wi.sum(axis=0)/sum(wi.sum(axis=0))
df_highfreqs = df_freqs.loc[df_freqs['total']>=100]

upper = 1.5*df_highfreqs.drop('total', axis=1).std(axis=1) + df_highfreqs.drop('total', axis=1).mean(axis=1)

clusterWords = [[] for i in range(numberTopics)]
for i in upper.index:
	if sum(df_highfreqs.drop(['total','word'], axis=1).loc[i,:] >= upper.loc[i]) > 0:
		highFreqClust = [j for j, val in enumerate(df_highfreqs.drop(['total','word'], axis=1).loc[i,:] >= upper.loc[i]) if val]
		for c in highFreqClust:
			clusterWords[c].append(df_highfreqs.loc[i,'word'])
            
for clust in range(numberTopics):
    print('Tags for Cluster ' + str(clust))
    print(clusterWords[clust])
    print(' ')


# These words help us to understand what papers in each cluster are about: 
# * Cluster 0 discusses how the virus was spreading, especially intially through China, and epidemiological models.
# * Cluster 1 appears to discuss research into testing.
# * Cluster 2 discusses sequencing the COVID-19 genome.
# * Cluster 3 discusses the spread of the virus.
# * Cluster 4 discusses the biology of the virus.
# * Clsuter 5 potentially discusses detection of the virus.
# * Cluster 6 discusses medical patient care for those diagnosed with COVID-19.
# 
# Furthermore, these general topics appear to match the medoids of each cluster as given earlier.

# ## Paper Summarization
# 
# Next we provide a method to summarize each abstract with a with a 1 (or 2) sentence summary. If a given abstract is either one or two sentences, we simply use the entire abstract. If an abstract contains more than 2 sentences, our goal is to determine which sentence best summarizes the entire abstract. That is, we find the sentence that is some sense "close" to all the other sentences in the abstract. To do this, we consider each abstract as an individual corpus, and each sentence as an individual document. Similar to the abstract clustering above, we seek to find which document(s) best represent the entire corpus. In the previous section we used LDA and LSA to calculate a distance measure between documents. LDA and LSA, however, do not work well on short documents. Insead, we use a new measure of distance between each of the sentences using only the individual words found within each sentence. We perform the following steps on each corpus (abstract) individually:
# 
# * Similar to above, we first create a word-document matrix $w$, where $w_{ij}$ gives the number of times word $j$ appears in sentence $i$.
# * Create a word co-occurrence matrix $c$, where $c_{ij}$ represents the number of times words $i$ and $j$ appear in the same sentence together. $C$ is a symmetric matrix and $c_{ii}$ is the number of sentences in the corpus that contain word $i$.
# * Create a distance matrix $d$ between words using $c$. We use $$d_{ij} = 2-\left[P(word~i\in sentence|word~j\in sentence) + P(word~j\in sentence|word_i\in sentence)\right]$$ There are many possible distance measure, but the measure above worked well in various experiments of short texts.
# * Create a distance matrix $s$ between sentences based on the word distance matrix $d$. To calculate the distance between sentences $i$ and $j$, we restrict $D$ such that the columns are words found in sentence $i$, and rows corresponding to words found in sentence $j$. Then let $s_{ij}$ be the mean of this restricted matrix.
# * A sentence that best summarizes the entire abstract is ideally somewhat related to many of the other sentences in the abstract. For each sentence we calculate the mean distance from that sentence to every other sentence in the abstract. The best summary sentence is chosen as the sentence with the smallest mean distance. 
# 
# The above method returns only one summary sentence per abstract. If we wanted $k$ summary sentences, we can apply k-medoids to the sentence-level distance matrix $s$ and return the $k$ different medoids. We stick to just one summary sentence in this report.

# In[11]:


# summary function
import nltk
nltk.download('punkt')

def abstractSummary(entireAbstract):
    if len(entireAbstract)==0 or entireAbstract=='' or entireAbstract==' ':
    	return('No Abstract')
    if len(entireAbstract)==0:
    	return('No Abstract')
    abstractCorpus_orig = nltk.tokenize.sent_tokenize(entireAbstract)
    abstractCorpus = [preProcessingFcn(sent) for sent in abstractCorpus_orig]
    abstractCorpus_orig = [abstractCorpus_orig[i] for i in range(len(abstractCorpus)) if abstractCorpus[i]!='']
    abstractCorpus = [sent for sent in abstractCorpus if sent != '']
    if(len(abstractCorpus)<=2):
        return(entireAbstract)
    else:
        # word matrix
        vectorizer = CountVectorizer(strip_accents='unicode')
        textsNew = vectorizer.fit_transform(abstractCorpus)
        w = textsNew.toarray()
        # word distance matrix
        c = w.T.dot(w)
        t = np.divide(c, np.diag(c)).transpose()
        d = 2 - t - t.T
        # sentence distance matrix
        n = len(abstractCorpus)
        s = np.zeros([n, n])
        for i in range(n):
            for j in range(n):
                if i<=j:
                    iWordsUse = np.where(w[i,:]>0)[0]
                    jWordsUse = np.where(w[j,:]>0)[0]
                    dtweetsij = d[np.ix_(iWordsUse, jWordsUse)]
                    entries = dtweetsij.shape[0]*dtweetsij.shape[1]
                    if entries !=0:
                        ij = np.sum(dtweetsij) / entries
                        s[i,j] = ij
                        s[j,i] = ij
        sumDists = s.mean(1)
        minDistEntry = np.argmin(sumDists)
        summarySentence = abstractCorpus_orig[minDistEntry]
        return(summarySentence)


# We apply the summarization algorithm to a random sample of 5 abstracts to demonstrate its effectiveness.

# In[12]:


# apply to sample of abstracts
import random

random.seed(567)
sample = random.sample(range(df.shape[0]), 8)

for samp in sample:
    print('Entire Abstract: ')
    print(df['abstract_noStmt'][samp])
    print(' ')
    print('Summary:')
    print(abstractSummary(df['abstract_noStmt'][samp]))
    print(' ')
    print(' ')


# ## Potential Breakthroughs
# 
# Breakthrough in COVID-19 research can take many forms, from breakthroughs in vaccine research to breakthroughs in transmission to breakthoughs in origin. We consider breakthroughs in treatment and prevention of COVID-19 and search for papers that may provide these breakthroughs.
# 
# We reason that if a paper is a breakthrough in the prevention or cure of the virus, it will contains both words relating breakthrough and words relating to treatments and cures. We create a list of words for both. Note that these lists are not comprehensive and were created by hand. We display abstracts that contain at least two words from each of these lists.

# In[13]:


breakthroughWords = ['breakthrough', 'discover', 'discovery', 'develop', 'improve']
treatmentWords = ['treament', 'cure', 'vaccine', 'inoculate', 'immunize','prevent', 'drug']

ps = PorterStemmer()
bkStemmed = [ps.stem(word) for word in breakthroughWords]
trStemmed = [ps.stem(word) for word in treatmentWords]

for indx in df_combined.index:
    if sum([word in df_combined['abstract'][indx] for word in bkStemmed])>1 and sum([word in df_combined['abstract'][indx] for word in trStemmed])>1:
        print('Title: ' + df_combined['title'][indx])
        print('Abstract: ' +df_combined['abstract_noStmt'][indx])
        print('')
        print('')


# ## References
# Blei, David M., Andrew Y. Ng, and Michael I. Jordan. "Latent dirichlet allocation." Journal of machine Learning research 3.Jan (2003): 993-1022.
# 
# Dumais, Susan T., et al. "Using latent semantic analysis to improve access to textual information." Proceedings of the SIGCHI conference on Human factors in computing systems. 1988.
