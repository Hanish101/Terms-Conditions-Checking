from flask import Flask,render_template, request

import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.corpus import stopwords
from nltk import word_tokenize, tokenize
from gensim.summarization import summarize
# from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import unicodedata
import pickle

from collections import Counter

# ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()
vectorizer = pickle.load(open('bow_vectorizer_1.pkl', 'rb'))
model = pickle.load(open('voting_model_1.pkl', 'rb'))

app = Flask(__name__)

def sample_preprocess(text):

  #Lowercasing the sentence
  lowercase_text = text[0].lower()
  #print(lowercase_text)

  #removing html tags, numbers, and dates from sentence
  pattern = re.compile("<.*?>|[0-9]+|[0-9]{,2}\/[0-9]{,2}\/[0-9]{2}")
  html_removed_text = pattern.sub(r'',lowercase_text)
  #print(html_removed_text)

  #Removing bracket of contents in sentence
  pattern = re.compile('[\(\[].*?[\)\]]')
  bracket_removed_text =  pattern.sub(r'',html_removed_text)
  #print(bracket_removed_text)

  #Removing stopwords from sentence
  stoplist  = set(stopwords.words('english'))
  word_tokenized_sent = [word for word in word_tokenize(bracket_removed_text) if not word in stoplist]
  sw_removed_text = ' '.join(word_tokenized_sent)
  #print(sw_removed_text)

  #Removing punctuation from sentence
  exclude = "!#$%&'()*’+,-/:;<=>?@[\]^_`{|}~"
  punc_removed_text =  sw_removed_text.translate(str.maketrans('', '', exclude))
  #print(punc_removed_text)

  #Summarizing if the sentence is bigger
  if(len(punc_removed_text.split("."))>10):
    summarized_text = summarize(punc_removed_text, ratio=0.3, split=True)
    summarized_text = "".join(summarized_text)
    text_pre = summarized_text
  else:
    text_pre = punc_removed_text
  #print(text_pre)

  #Stemming the words in the sentence
  lemmatized_words =  " ".join([lemmatizer.lemmatize(word) for word in text_pre.split()])
  # print(lemmatized_words)

  #Removing fullstop from sentence
  final_text = lemmatized_words.translate(str.maketrans('', '', '.'))
  final_text = [final_text]
  return final_text

def sent_tokenized_text(terms):
  sent_tokenize_terms = tokenize.sent_tokenize(terms)
  #print(sent_tokenize_terms)
  cleaned_text=[]
  for i in range(len(sent_tokenize_terms)):
    #print(sent_tokenize_terms[i])
    clean_text = unicodedata.normalize("NFKD",sent_tokenize_terms[i])
    clean_text = re.sub("\n|\r", " ",clean_text)
    clean_text = [clean_text]
    cleaned_text.append(clean_text)
    #print(cleaned_text)
  return cleaned_text

def predict_terms(cleaned_text):
  dictionary_of_sentences = {}
  values = []
  for j in range(len(cleaned_text)):
    sentence_to_predict = cleaned_text[j]
    processed_text = sample_preprocess(sentence_to_predict)
    vectorized_text = vectorizer.transform(processed_text).toarray()
    prediction = model.predict(vectorized_text)
    #print(sentence_to_predict," :",voting.predict(vectorized_text))
    dictionary_of_sentences[sentence_to_predict[0]] = prediction[0]
    values.append(prediction[0])

  return dictionary_of_sentences, values

@app.route('/')

def home():
    return render_template('index.html')

@app.route('/submitted', methods=['POST'])
def form():
    terms = request.form.get('terms')
    cleaned_text = sent_tokenized_text(terms)
    result_dict, values = predict_terms(cleaned_text)
    counter_dict = Counter(values)
    print(counter_dict)
    good  = counter_dict.get(0)
    print(good)
    medium = counter_dict.get(1)
    if medium == None:
        medium = 0
    print(medium)
    bad = counter_dict.get(2)
    if bad == None:
        bad = 0
    print(bad)
    total = good + bad + medium

    if bad > 0:
        review = "Go through RED conditions once again."
    elif medium > 0:
        review = "Go ahead responsibily"
    else:
        review = "Good, you can go ahead."


    result_dict = dict(sorted(result_dict.items(), key=lambda item: item[1], reverse=True))
    return render_template('op.html', result = result_dict, good = good, average = medium, bad = bad, total =total, review=review)
    # return render_template('op.html', result=result_dict, counter_dict = counter_dict)

if __name__ == "__main__":
    app.run(port=8020,debug=True)