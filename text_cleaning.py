import pandas as pd
import nltk
import string
import re

#! fetching twitter data set with 1.6 million rows
data = pd.read_csv('twitterdata.csv')
#! separating just the re quired fields
data = data[['text', 'target']]
#! creating csv of new data
data.to_csv("clean_data_1.csv", index=False)
#! reading data for cleaning
data = pd.read_csv('clean_data_1.csv')
emoticons_happy = set([
    ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
    ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
    '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
    'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
    '<3'
])
emoticons_sad = set([
    ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
    ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
    ':c', ':{', '>:\\', ';('
])
emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)

emoticons = emoticons_happy.union(emoticons_sad)


def clean_tweets(tweet):
    stop_words = nltk.corpus.stopwords.words('english')
    word_tokens = nltk.word_tokenize(tweet)
#!   after tweepy preprocessing the colon symbol left remain after      #removing mentions
    tweet = re.sub(r':', '', tweet)
    tweet = re.sub(r'‚Ä¶', '', tweet)
#!   replace consecutive non-ASCII characters with a space
    tweet = re.sub(r'[^\x00-\x7F]+', ' ', tweet)
#!   remove emojis from tweet
    tweet = emoji_pattern.sub(r'', tweet)
#!   filter using NLTK library append it to a string
    filtered_tweet = [w for w in word_tokens if not w in stop_words]
    filtered_tweet = []
#!   looping through conditions
    for w in word_tokens:
        #! check tokens against stop words , emoticons and punctuations
        if w not in stop_words and w not in emoticons and w not in string.punctuation:
            filtered_tweet.append(w)
    tt = ' '.join(filtered_tweet)
    t2 = re.sub(r"http \S+", "", tt)
    return t2


#! applying clean function to data
data['text'] = data['text'].apply(lambda x: clean_tweets(x))
#! removing fullstops & extra spaces
data.to_csv('clean_data_2.csv', index=False)
with open('clean_data_2.csv', 'r', encoding='utf8', errors='ignore') as f, open('clean_fullstop.csv', 'w') as fo:
    for line in f:
        fo.write(line.replace('.', ''))
with open('clean_fullstop.csv', 'r') as f, open('clean_final.csv', 'w') as fo:
    for line in f:
        fo.write(line.replace('  ', ' '))
