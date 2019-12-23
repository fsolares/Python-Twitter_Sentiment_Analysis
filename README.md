# Twitter_Sentiment_Analysis
This is a Sentiment Analysis project developed by Felipe Solares da Silva and is part of his professional portfolio.
Acknowledgment

Thank you Jaques D'Erasmo and Eduardo Passos, old friends and also Data Science students (12/19/2019) for your immeasurable contribution on this project, all your feedback, code sujection and support during my path gave me the strength to overcome this challenge. Congratualation for us, that was an ammazing and real team work.

#### SOFTWARES & PACKAGES VERSIONS

INSTALLED VERSIONS

commit           : None<br>
python           : 3.6.9.final.0<br>
python-bits      : 64
OS               : Windows<br>
OS-release       : 10<br>
machine          : AMD64<br>
processor        : Intel64 Family 6 Model 58 Stepping 9, GenuineIntel<br>
byteorder        : little<br>
LC_ALL           : None<br>
LANG             : None<br>
LOCALE           : None.None<br>

pandas           : 0.25.1<br>
numpy            : 1.17.2<br>
pytz             : 2019.2<br>
dateutil         : 2.8.0<br>
pip              : 19.3.1<br>
setuptools       : 41.6.0<br>
Cython           : 0.29.13<br>
pytest           : 5.0.1<br>
hypothesis       : None<br>
sphinx           : 2.2.0<br>
blosc            : None<br>
feather          : None<br>
xlsxwriter       : 1.2.1<br>
lxml.etree       : 4.4.1<br>
html5lib         : 1.0.1<br>
pymysql          : None<br>
psycopg2         : None<br>
jinja2           : 2.10.1<br>
IPython          : 7.8.0<br>
pandas_datareader: None<br>
bs4              : 4.8.0<br>
bottleneck       : 1.2.1<br>
fastparquet      : None<br>
gcsfs            : None<br>
lxml.etree       : 4.4.1<br>
matplotlib       : 3.1.1<br>
numexpr          : 2.6.4<br>
odfpy            : None<br>
openpyxl         : 2.6.3<br>
pandas_gbq       : None<br>
pyarrow          : None<br>
pytables         : None<br>
s3fs             : None<br>
scipy            : 1.3.1<br>
sqlalchemy       : 1.3.8<br>
tables           : 3.4.2<br>
xarray           : None<br>
xlrd             : 1.2.0<br>
xlwt             : 1.3.0<br>
xlsxwriter       : 1.2.1<br>

Sentiment Analysis using Twitter API and NLTK Naive Bayes Classifier
Project purpose

Perform a sentiment analysis using NLTK Naive Bayes Classifer to identify the sentiment from tweets users related with Netflix services (or any other high tech company as Apple, Microsoft, Google, for example).




# Step 1 - Importing libraries

README!

It is recommended that you install the most recent version of Anaconda distribution.

Additionally, few other packaged will need to be installed separately in case you do not have those installed. Bellow you can find each package and the installation command that can be executed from jupyter notebook cell except for the worldcloud package that need to be executed from the Anaconda Terminal (Windows) or the Terminal (Linux or MAC)

tweepy - !pip install tweepy<br>
textblob - !pip install textblob<br>
worldcloud - conda install -c https://conda.anaconda.org/conda-forge wordcloud<br>

    #!pip install tweepy

    #!pip install textblob

#### General Libraries

    import pickle
    import os
    import tweepy #tweepy is a libraries that work similar as 'twitter' used during the course. I decided 
                  #to use thatsince I previuosly new about that and I feel more confortble. It is important
                  #to mention that the principles behind both libriries are the same


    import pandas as pd
    import nltk
    from textblob import TextBlob #it is necessary install the textblob library if you don't have it yet! 
                                  #This is a veryuseful library that will be used during this work in two ways, 
                                  #as a translator for tweets in different languages and in a simplefied 
                                  #sentiment analysis that will be aslo presented
            
            

if you don't have the wordcloud library instaled, one option, if using Anaconda, you can install this 
package with the command bellow (type exactly as it is bellow on your Anaconda Prompt):
conda install -c conda-forge wordcloud


#### Visualization Libraries

    from operator import itemgetter
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    %matplotlib inline


#### Preprocess Libraries

    import re #module used to work with regular expressions
    from nltk.tokenize import word_tokenize
    from string import punctuation
    from nltk.corpus import stopwords


    nltk.download('movie_reviews')
    nltk.download("punkt")
    nltk.download("stopwords")


# Step 2 - Saving my credentials and creating the OAuth object to access tweets on Twitter

    #This strucre will save your credentials at the first time you exetute it, after that, erase your
    #credentials to avoind anyone else to use that

    if not os.path.exists('secret_twitter_credentials.pkl'):
        Twitter={}
        Twitter['Consumer Key'] = '-== Your Consumer Key ==-'
        Twitter['Consumer Secret'] = '-== Your Consumer Secret ==-'
        Twitter['Access Token'] = '-== Your Access Token ==-'
        Twitter['Access Token Secret'] = '-== Your Access Token Secret ==-'
        with open('secret_twitter_credentials.pkl','wb') as f:
            pickle.dump(Twitter, f)
    else:
        Twitter=pickle.load(open('secret_twitter_credentials.pkl','rb'))

#### Authenticating and creating an API object

    auth = tweepy.OAuthHandler(Twitter['Consumer Key'], Twitter['Consumer Secret'])
    auth.set_access_token(Twitter['Access Token'], Twitter['Access Token Secret'])
    api = tweepy.API(auth, wait_on_rate_limit=True,wait_on_rate_limit_notify=True)

# Step 3 - Getting tweets as test data

#Function that will access the Tweet and get tweets based on a search term, number of tweets, inicial 
#date and final date. This function will return a list of dictionaries where the key is the tweet text 
#and the values is a label, in this casa we are considering 'None' in the labor. This format is necessary
#for futures aplication

    def get_tweets(word='google',number=100, since='2019-11-01', until='2019-11-28'):

        search_term = str(input('Hi! What are you looking for today? - '))
        num_of_terms = int(input('How many tweets do you want? - '))
        since = input('From which date? ex.: 2019-11-01 - ')
        until = input('Until which date? ex.: 2019-11-28 - ')

        data = []

        tweets = tweepy.Cursor(api.search,
                                    #since = f'{since},
                                    #until = f'{until},
                                    q=search_term,
                                    lang='en').items(num_of_terms)                                

        for tweet in tweets:
            try:
                data.append(tweet.extended_tweet.full_text)
            except:
                data.append(tweet.text)


        new_data = set(data) #converting in a set to remove the duplicates

        new_data_dic = []

        for i in new_data:
            new_data_dic.append({'text':i, 'label':None}) #this strucure as a dictionary is necessary for  
                                                          #further purposes

        print()

        #Will print a message to the user 
        print('Great Job! We got '+str(len(data))+' tweets with the term '+word+'!!')


        return [i for i in new_data_dic]

#### Creating a tweets objecting and printing the first 4 tweets
    tweets = get_tweets()
    tweets[:3]


### IMPORTANT!!!

#### The Twitter API have limitations on the total of tweets and how many requests can be made.
#### A recomendation is do not request more than 3000 tweets in an interval of 15 min to avoid long waiting
#### time.

#### Checking the final lengh of the request we can see the total is not the 1000 tweets requested. The 
#### reason for that is that the tweet api return duplicates and the above function do the job of removing
#### those.

    len(tweets)

# Step 4 - Getting training data

We'll use the Niek Sander's tweets corpus woth ~5000 classified tweets labelled as positive, negative, neutral or irrelevant. Those tweets are related with tech companies like, Apple, Google, Twitter, Youtube, Microsoft and others. Said that, the model that will be created is recommended to a search term related with tech companies. The performance lower if the sentiment analysis is related with another kind of topic.

**Getting Nick Sander's tweets corpus!**

* access the link https://github.com/zfz/twitter_corpus;
* download the file;
* we will use the **full-corpus.csv** file. This file already contain the tweets texts and it respective labels of *positive*, *negative*, *neutral* or *irrelevant*.


After download the file it is recommend that you explore it using Pandas for a better comprehension 

#Function to read the nick_sanders_corpus csv file and return a list containing the text and the label
#for each tweet

#### reading the nick_sanders_corpus
    df = pd.read_csv('full_corpus.csv')

#### function
    def trainingData():
        trainingData = [{'text':row[4], 'label':row[1]} for index,row in df.iterrows()]
        return trainingData

#The function return a list of dict containing the key (tweets text) and the values (the labels)


    df.head()

    training_Data = trainingData()
    training_Data[:3]

# Step 5 - Preprocessing tweets from test and training data

The preprocess step will use few python tools to work with strings as detailed bellow:

        USING .LOWER() STRING FUNCTION
        
            1 - Convert to lower case
        
        USING REGULAR EXPRESSIONS
            2 - Replace links with the string 'url'
            3 - Replace @ ... with 'at_user'
            4 - Replace #word with the word itself
            5 - Remove emoticons using ASCII encode and decode
        
        USING NLTK
            7 - Tokenize the tweet into words (a list of words)
            8 - Remove stopwords (including url and user, RT and '...')

#### Creating a class to preprocess the test and training tweets

    class Preprocess:
        def __init__(self):

            lst = ['AT_USER','URL','rt','...', "'s", "n't", "``", "''"]
            self._stopwords = set(stopwords.words('english')+list(punctuation)+lst)


        def processTweets(self, tweets):
            #tweets is a list of dict with Keys, 'text' and 'label'
            processedTweets = []
            #this list will be a list of tuple. Each tuple is a tweet which is a list of words and its label

            for tweet in tweets:
                processedTweets.append((self.cleanTweet(tweet['text']),tweet['label']))
                                             #it will apply the cleanTweet function only to the tweets text
            return processedTweets


        def processTweets_words(self, tweets):

            processedTweets = []
            #this list will be a list of each word in all tweets

            for tweet in tweets:
                processedTweets.append(self.cleanTweet(tweet['text']))
                                             #it will apply the cleanTweet function only to the tweets text
            return processedTweets


        def cleanTweet(self,tweet):
            #1 - Convert to lower case
            tweet = tweet.lower()

            #2 - Replace links with word 'URL'
            tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)

            #3 - Replace @username with 'AT_USER'
            tweet = re.sub('@[^\s]+','AT_USER',tweet)

            #4 - Repalce #word (hashtag) with just the word, witouth the '#' simble
            tweet=re.sub(r'#([^\s]+)',r'\1',tweet)

            #5 - Remove emoticons
            tweet = tweet.encode('ascii', 'ignore').decode('ascii')

            #6 - Tokenizing the tweets
            tweet = word_tokenize(tweet)

            #7 - Removing stopwords 
            return [word for word in tweet if word not in self._stopwords]        

    #Instantiating a Preprocess Class called 'tprocessor'

    tprocessor = Preprocess()

    #Creating an object that will contain the result of the TRAINING DATA after been preprocessed by the
    #'tprocessor' class

    cleanedtrainingData = tprocessor.processTweets(training_Data)

    #Creating an object that will contain the result of the TEST DATA after been preprocessed by the
    #'tprocessor' class

    cleanedtestData = tprocessor.processTweets(tweets)

    print(cleanedtrainingData[:10])

    print(cleanedtestData[:10])

## Step 5.1 - Word Frequence Visualizations

* Word Frequence using a bar plot
* Word Frequence using a Word Cloud

#the result of the 'processTweets_words' is a list fo lists, so it will be necessary unify all words in
#only one list

    words = tprocessor.processTweets_words(tweets)

    unif_words =[]
    for i in words:
        for j in i:
            unif_words.append(j)

    unif_words[:5]

#### word frequence using 'Counter' class from the 'collection' package

    from collections import Counter

    word_counter = Counter(unif_words)

    most_common_words = word_counter.most_common()[:15]

    print(most_common_words)

    #Word distribution present a large vocabulary
    sorted_word_counts = sorted(list(word_counter.values()), reverse=True)

    plt.loglog(sorted_word_counts)
    plt.ylabel("Freq")
    plt.xlabel("Word Rank")
    plt.title('Word Frequence - Log x Log')
    plt.grid()
    plt.show()

    #Word distribution that allowed detect the count of words in a specific range

    plt.hist(sorted_word_counts, bins=50, log=True);
    plt.ylabel("Freq")
    plt.xlabel("Word Rank")
    plt.title('Word Frequence - semilog')
    plt.grid()
    plt.show()

    #Converting the dictionary to a pandas DataFrame
    df = pd.DataFrame(most_common_words, columns=['word','freq'])

    #sorting the words based on their frequence
    df = df.sort_values(by=['freq'], ascending=False)

    #getting the top 20 more frequent words
    top_20 = df.head(20)
    top_20

    #Bar ploting from the top 20 more frequent words

    axes = top_20.plot.bar(x='word',y='freq',legend=False)
    plt.gcf().tight_layout()
    plt.ylabel('Freq')
    plt.grid()
    plt.title('Word Frequence')
    plt.show()

    dic = {}
    for item in unif_words:
        dic[item] = dic.get(item, 0) + 1

    #Bulding a World Cloud



    wordcloud = WordCloud(width=1600,   height=900,
                          prefer_horizontal=0.5,
                          min_font_size=10,
                          colormap='prism')

    wordcloud = wordcloud.fit_words(dic)

    wordcloud = wordcloud.to_file('TrendingTwitter.png')


### IMPORTANT!!!

#### The result will create a png file in the folder where the jupyter notebook are running. In order to see the result you will need to go check this file

    ![TrendingTwitter.png](attachment:TrendingTwitter.png)

# Step 6 - Using NLTK to extract features and train the Classifier Model

**Extract features from both the test data (the 100 tweets to be classifed) and the training data, downloaded from the corpus**
    
    The NLTK model that will be used is known as NAVIE BAYES CLASSIFICATION

           1.1 - Build a vocabulary (list of all unique words in all the tweets in the training data);

           2.2 - Represent each tweet with the presence or absence of theses words;
                 Ex.: Given a *vocabulary: {'the','worst','thing,'in','the','world'} and a *tweet: {'the','worst','thing'}        

                      This tweet will be represented as a "Feature Vector" (1,1,1,0,0,0) -> indicatinting that the 
                      first three words from the vocabulary are in the tweet, and there is other 3 words in the 
                      vocabulary that is not part of the mentioned tweet.

           2.3 - Use NLTK's built in Navie Bayes Classifier to train a Classifier                                 

    #Defining the function to create a word Vocabulary or bag_of_words

    def wordVocab(cleanedtrainingData):
        training_features = []
        for (words, sentiment) in cleanedtrainingData:
            training_features.extend(words)
        return list(set(training_features))

#### Creating the word_features_vocab object

    word_features_vocab = wordVocab(cleanedtrainingData)

    #The NLTK library have a function called apply_features that takes a user-defined function to extract
    #featrues from training data. In this case the function will be called extract_features_func that will
    #take each tweet in the in the training data and repersent it with the presence or absence of a word in
    #the vocabulary, as previously explaned in the item 2.2 above.

    def extract_features_func(tweet):
        tweet_words = set(tweet)
        features = {}
        for word in word_features_vocab:
            features[f'contains {word}'] = (word in tweet_words)
            # this step will creat a dictionary with keys like 'contains word1' and 'contains word2', and values
            # as True or False. The statement that create the True or False return is the '(word in tweet_words)'

        return features

#### Creating the traning_features object

    #apply_features will take the extract_features_func defined above, and apply it to each element of
    #cleanedtrainingData. it automatically recognize that each of these elements are tuples where the
    #first element is the text and the second is the label. The apply_features apply the extract_features_func
    #only on the text element.

    trainingFeature = nltk.classify.apply_features(extract_features_func,cleanedtrainingData)



#### Creating the classifier objected, trained using the training data features 

    NBclassifier = nltk.NaiveBayesClassifier.train(trainingFeature)

# Step 7 - Run the Classifier on the 2000 downloaded tweets

    sentiment_classifier = [NBclassifier.classify(extract_features_func(tweet[0])) for tweet in cleanedtestData]

    print('-='*40)
    print('{:^80}'.format('SENTIMENT RESULTS USING NAIVE BAYES CLASSIFIER'))
    print('-='*40)
    print('')


    if sentiment_classifier.count('positive'):
        print('Positive Sentiment = {:.2f}'.format(100*sentiment_classifier.count('positive')/len(sentiment_classifier))+"%")

    if sentiment_classifier.count('negative'):
        print('Negative Sentiment = {:.2f}'.format(100*sentiment_classifier.count('negative')/len(sentiment_classifier))+"%")

    if sentiment_classifier.count('neutral'):
        print('Neutral Sentiment = {:.2f}'.format(100*sentiment_classifier.count('neutral')/len(sentiment_classifier))+"%")

    if sentiment_classifier.count('irrelevant') or sentiment_classifier.count('irrelevant') == 0:
        print('Irrelevant Sentiment = {:.2f}'.format(100*sentiment_classifier.count('irrelevant')/len(sentiment_classifier))+"%")


    print('')
    print('-='*40)
    print('{:^80}'.format('First 10 tweets'))
    print('-='*40)
    print('')

    tweets[0:10]


# A simplified approach using TextBlob library

The object of this section is to present a simple and very interest tool that I could discover on my journey to complete this project. I will also perform the same kind of sentiment analysis in a much more simplefied version using a powerful library called TextBlob.

TextBlob is an object-oriented NLP text-processing library that is built on the NLTK and pattern NLP libraries and simplifies many of their capabilities.

*During my exploration to perform this work I had the chance to learn about this library and I strongly recomend that you dedicate a time to read and understand better the main tasks that TextBlob can perform.*

## Libraries

    import pickle
    import os
    import tweepy
    from textblob import TextBlob

    #Saving/loading the credations to access the tweet API

    #This structure will save your credentials at the first time you exetute it, after that, erase your
    #credentials to avoind anyone else to use that
    if not os.path.exists('secret_twitter_credentials.pkl'):
        Twitter={}
        Twitter['Consumer Key'] = 'your_cosumer_key_here'
        Twitter['Consumer Secret'] = 'your_cosumer_secret_here'
        Twitter['Access Token'] = 'your_access_token_here'
        Twitter['Access Token Secret'] = 'your_access_token_secret_here'
        with open('secret_twitter_credentials.pkl','wb') as f:
            pickle.dump(Twitter, f)
    else:
        Twitter=pickle.load(open('secret_twitter_credentials.pkl','rb'))
    
    
#### Authenticating and creating an API object

    auth = tweepy.OAuthHandler(Twitter['Consumer Key'], Twitter['Consumer Secret'])
    auth.set_access_token(Twitter['Access Token'], Twitter['Access Token Secret'])
    api = tweepy.API(auth, wait_on_rate_limit=True,wait_on_rate_limit_notify=True)

    #Function that will access the Tweet and get tweets based on a search term, number of tweets, inicial 
    #date and final date. This function will return a list of dictionaries where the key is the tweet text
    #and the values is a label, in this casa we are considering 'None' in the labor. This format is necessary
    #for futures aplication


    def get_tweets2(word='google',number=5, since='2019-11-01', until='2019-11-28'):

        search_term = str(input('Hi! What are you looking for today? - '))
        num_of_terms = int(input('How many tweets do you want? - '))
        since = input('From which date? ex.: 2019-11-01 - ')
        until = input('Until which date? ex.: 2019-11-28 - ')

        tweets = tweepy.Cursor(api.search,
                                    #since = f'{since},
                                    #until = f'{until},
                                    q=search_term,
                                    lang='en').items(num_of_terms)                                


        positive = 0
        negative = 0
        neutral = 0

        for tweet in tweets:
            try:
                analysis = TextBlob(tweet.extended_tweet.full_text)
            except:
                analysis = TextBlob(tweet.text)

                if analysis.sentiment[0] > 0.00:
                    positive += 1
                elif analysis.sentiment[0] < 0.00:
                    negative += 1
                else:
                    neutral += 1

        print()
        print('-=' * 3, 'TOTALS', '-=' * 3)
        print('Total Positives:', positive)
        print()
        print('Total Negatives:', negative)
        print()
        print('Total Neutrals:', neutral)
        print()
        print()
        print('-=' * 3, 'PERCENTAGE', '-=' * 3)
        print(int((positive / num_of_terms) * 100), '% Positives')
        print()
        print(int((negative / num_of_terms) * 100), '% Negatives')
        print()
        print(int((neutral / num_of_terms) * 100), '% Neutrals')
        print()

        return positive, negative, neutral

    tweets2 = get_tweets2()

### Adjusting the TextBlob to perform the sentiment analysis using a Navie Bayes Analyser

By default, a TextBlob and the Sentences and Words you get from it determine sentiment using a PatternAnalyzer, which uses the same sentiment analysis techniques as in the Pattern library. The TextBlob library also comes with a NaiveBayesAnalyzer9 (module text-blob.sentiments), **which was trained on a database of movie reviews**.

Considering that we can not re-train the algorithm in the TextBlob, it is important to understand that the performance compromised since the train data was not related with the topic of the search term that will be considered on this analysis, a term realated to a tech company for the sentiment analysis

    from textblob.sentiments import NaiveBayesAnalyzer


### Recommendation!!!

#### The sentiment analysis using TextBlob NaiveBayesAnalyzer presented a low performance in terms of speed,
#### basically it takes long time to process the analysis. Request a max of 20 tweets in order to be able to
#### quickly have a return and evaluate the result. If you have more time and a powerful machine you maybe want
#### to try a larger tweet request.

    def get_tweets3(word='google',number=5, since='2019-11-01', until='2019-11-28'):

        search_term = str(input('Hi! What are you looking for today? - '))
        num_of_terms = int(input('How many tweets do you want? - '))
        since = input('From which date? ex.: 2019-11-01 - ')
        until = input('Until which date? ex.: 2019-11-28 - ')

        tweets = tweepy.Cursor(api.search,
                                    #since = f'{since},
                                    #until = f'{until},
                                    q=search_term,
                                    lang='en').items(num_of_terms)                                


        positive = 0
        negative = 0

        for tweet in tweets:

            blob = TextBlob(tweet.text, analyzer=NaiveBayesAnalyzer())

            if blob.sentiment.p_pos > blob.sentiment.p_neg:
                positive += 1
            else:
                blob.sentiment.p_pos < blob.sentiment.p_neg
                negative += 1


        print()
        print('-=' * 3, 'TOTALS', '-=' * 3)
        print('Total Positives:', positive)
        print()
        print('Total Negatives:', negative)
        print()
        print()
        print('-=' * 3, 'PERCENTAGE', '-=' * 3)
        print(int((positive / num_of_terms) * 100), '% Positives')
        print()
        print(int((negative / num_of_terms) * 100), '% Negatives')
        print()

        return positive, negative

    tweets3 = get_tweets3()
