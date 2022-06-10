import pickle
import streamlit as st
from PIL import Image
from textPreprocessing import TextPreprocessing



dictionary_model = pickle.load(open('dictionary.pkl','rb'))
loaded_model = pickle.load(open('model.pkl','rb'))

def categorize(text):
    var = ''
    dtm = dictionary_model.doc2bow(TextPreprocessing(text))
    dtm_score = sorted(loaded_model[dtm],key = lambda x: -1*x[1])[0][1]
    if dtm_score >= 0.2:
        var = "Category number:" + str(sorted(loaded_model[dtm],key = lambda x: -1*x[1])[0][0]+1)
    else:
            var = 'Not in this category'
    if text == '':
        var = 'News is missing,Please enter the news'
    return var
    
    
    
    '''
    var = ""
    content = TextPreprocessing(text)
    vect = TfidfVectorizer(max_features=1000) 
    try:
        vect_text=vect.fit_transform(content)
    except ValueError:
        return 'News is missing, please enter the news'
    vocab = vect.get_feature_names_out()
    for i ,comp in enumerate(loaded_model.components_):
    	vocab_comp = zip(vocab,comp)
    	sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)
    for i in range(len(sorted_words)):
        if sorted_words[i][1] >= 0.2:
            var = "Topic "+str(i)
            if i > 5:
                var = "Topic 5"
        else:
            var = "Not in this Category"
        # if text =='':
        #     return 'News is missing, please enter the news'
    return var
            
    
    #model = loaded_model.fit_transform(vect_text)
    #dtm_score = sorted(loaded_model.topic_word_prior_,key = lambda x: -1*x[1])[0][1]
    vocab = vect.get_feature_names_out()
    for i, comp in enumerate(loaded_model.components_):
        vocab_comp = zip(vocab, comp)
        sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)
        print("Topic "+str(i))
        for t in sorted_words:
            print(t[0],end=" ")
        print("\n")
    score = loaded_model.doc_topic_prior_
    
    if score >= 0.2:
        var = "Category Number : "  
    else:
            var = 'Not in this category'
    if text == '':
        var = 'News is missing,Please enter the news'
    return var
'''

def main():
    
    st.title('INDIAN NEWS HEADLINES TOPIC MODELLING')
    news = st.text_area('Enter the News')
    
    if st.button('Categorize'):
        categorize(news)
    result = categorize(news)
    if result == "Category number:1":
        cat1 = Image.open('Category1.png')
        st.text('Entertainment News')
        st.image(cat1)
    elif result == "Category number:2":
        cat2 = Image.open('Category2.png')
        st.text('Sports News')
        st.image(cat2)
    elif result == "Category number:3":
        cat3 = Image.open('Category3.png')
        st.text('Cinema News')
        st.image(cat3)
    elif result == "Category number:4":
        cat4 = Image.open('Category4.png')
        st.text('Economic News')
        st.image(cat4)
    elif result == "Category number:5":
        cat5 = Image.open('Category5.png')
        st.text('City News')
        st.image(cat5)
    st.success(result)

if __name__ == '__main__':
    main()
