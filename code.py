'''
Import the required libraries 
'''
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import regex as re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pickle
from gensim import corpora,models

'''
Import the Indian News Headlines Dataset
'''
data = pd.read_csv('india-news-headlines.csv')
data.drop_duplicates(subset='headline_text',keep='first',inplace=True)
'''
Subset the dataset using the publish date
'''
data['publish_date'] = pd.to_datetime(data['publish_date'],format='%Y%m%d')
df = data[data['headline_category'] == 'unknown']
textData = df[['headline_text']]

def TextPreprocessing(text):
    text = re.sub(r'[^a-z\s]', '', str(text).lower().strip())
    text = word_tokenize(text)
    stop_words = stopwords.words('english')
    stop = ['u','us','india','indian','pics','get','take']
    stop_words = stop_words + stop
    text = [word for word in text if word not in stop_words]
    word_lem = WordNetLemmatizer()
    news_text = [word_lem.lemmatize(word,pos='v') for word in text]
    cleaned_text = " ".join(news_text)
    return cleaned_text

textData['cleanText'] = textData['headline_text'].apply(TextPreprocessing)
textData.drop(['headline_text'],axis=1,inplace=True)

vect = TfidfVectorizer(max_features=100000,stop_words='english',ngram_range=(2,3))
vect_text=vect.fit_transform(textData['cleanText'])


score = []
perplexity = []
for k in range(5,20):
    lda = LatentDirichletAllocation(n_components=k,max_iter=1)
    model = lda.fit_transform(vect_text)
    coherence = lda.score(vect_text)
    perp = lda.perplexity(vect_text)
    score.append((k,coherence))
    perplexity.append((k,perp))

x = []
y = []

for i in range(0,15):
    x.append(score[i][0])
    y.append(score[i][1])

plt.plot(x,y)
plt.scatter(x,y)
plt.title('Topics vs Coherence score')
plt.show()


xp = []
yp = []

for i in range(0,10):
    xp.append(perplexity[i][0])
    yp.append(perplexity[i][1])

plt.plot(xp,yp)
plt.scatter(xp,yp)
plt.title('Topics vs Perplexity score')
plt.show()

lda = LatentDirichletAllocation(n_components=5,max_iter=1)
model = lda.fit_transform(vect_text)


vocab = vect.get_feature_names_out()
for i, comp in enumerate(lda.components_):
    vocab_comp = zip(vocab, comp)
    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:10]
    print("Topic "+str(i)+": ")
    for t in sorted_words:
        print(t[0],end=" ")
    print("\n")


def draw_word_cloud(index):
  imp_words_topic=""
  comp=lda.components_[index]
  vocab_comp = zip(vocab, comp)
  sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:50]
  for word in sorted_words:
    imp_words_topic=imp_words_topic+" "+word[0]

  wordcloud = WordCloud(width=600, height=400).generate(imp_words_topic)
  plt.figure( figsize=(5,5))
  plt.imshow(wordcloud)
  plt.axis("off")
  plt.tight_layout()
  plt.show()
  plt.savefig("Category " +str(index) +".png")



words = textData['cleanText'].to_list()
wordForDictionary = [i.split() for i in words]
dictionaryForDocumentTermMatrix = corpora.Dictionary(wordForDictionary)
documentTermMatrix = [dictionaryForDocumentTermMatrix.doc2bow(w) for w in wordForDictionary]
LDA = models.ldamodel.LdaModel
ldaModel = LDA(num_topics=5,corpus=documentTermMatrix,id2word=dictionaryForDocumentTermMatrix, iterations=1)
ldaModel.print_topics()

topic = 0 
while topic < 5:
    topic_words_freq = dict(ldaModel.show_topic(topic, topn=60))   
    #Word Cloud for topic using frequencies
    wordcloud = WordCloud(background_color="white").generate_from_frequencies(topic_words_freq) 
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Category --"+str(topic+1)+"\n",fontsize='22')
    plt.tight_layout()
    plt.savefig("Category"+str(topic+1)+".png")
    topic += 1 

modelname = 'model.pkl'
dictionaryname = 'dictionary.pkl'
pickle.dump(ldaModel,open(modelname,'wb'))
pickle.dump(dictionaryForDocumentTermMatrix,open(dictionaryname,'wb'))
             
'''
lsa_model = TruncatedSVD(n_components=5, algorithm='randomized', n_iter=10, random_state=42)
modelLsa = lsa_model.fit_transform(vect_text)


for i, comp in enumerate(lsa_model.components_):
    vocab_comp = zip(vocab, comp)
    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:10]
    print("Topic "+str(i)+": ")
    for t in sorted_words:
        print(t[0],end=" ")
    print("\n")
'''        