from sklearn.feature_extraction.text import CountVectorizer
import googleapiclient.discovery as gapi
from textblob import TextBlob as tb
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle

#! this function fetches comments from youtube video based on videoid


def get_comments(youtube, video_id):
    comment_text = []
    like_count_comment = []
    results = youtube.commentThreads().list(part="snippet", videoId=video_id,
                                            textFormat="plainText", maxResults=100, order="relevance").execute()
    for i in range(len(results["items"])):
        d = dict(results["items"][i])
        d1 = dict(d['snippet'])
        d2 = dict(d1['topLevelComment'])
        d3 = dict(d2['snippet'])
        comment_text.append(d3["textDisplay"])
        like_count_comment.append(d3['likeCount'])
    while(len(results) == 5):
        results = youtube.commentThreads().list(part="snippet", videoId=video_id, textFormat="plainText",
                                                maxResults=100, order="relevance", pageToken=results['nextPageToken']).execute()
        for i in range(len(results["items"])):
            d = dict(results["items"][i])
            d1 = dict(d['snippet'])
            d2 = dict(d1['topLevelComment'])
            d3 = dict(d2['snippet'])
            comment_text.append(d3["textDisplay"])
            like_count_comment.append(d3['likeCount'])
    return comment_text, like_count_comment

#! this function fetches video stats


def get_video_stats(youtube, video_id):
    viewCount = 0
    likeCount = 0
    dislikeCount = 0
    commentCount = 0
    stats = youtube.videos().list(part="statistics", id=video_id).execute()
    s1 = dict(stats["items"][0])
    s2 = dict(s1['statistics'])
    viewCount = int(s2["viewCount"])
    likeCount = int(s2["likeCount"])
    dislikeCount = int(s2["dislikeCount"])
    commentCount = int(s2["commentCount"])
    return viewCount, likeCount, dislikeCount, commentCount

#! calculating sentiment from extracted comments


def calculating_sentiment(comment_text, m, v):
    model = m
    vect = v
    comments = pd.Series(comment_text)
    comm = vect.transform(comments)
    res = model.predict(comm)
    pos = np.count_nonzero(res == '4')
    neg = np.count_nonzero(res == '0')
    return pos, neg


#! api key
api_key = "AIzaSyDU0RH-d-669gkzB5UWgW_2H6wpD-GULxg"
#! youtube api
youtube = gapi.build('youtube', 'v3', developerKey=api_key)

# vid_id="6fYRJtKF3dU"
# vid_id="4Z6lxfglvUU"
# vid_id="YbJOTdZBX1g"
vid_id = "EeF3UTkCoxY"
# vid_id="8HslUzw35mc"
# vid_id="qlbAQArT7YQ"
# vid_id="1VrWaED18_g"
#! opening model and count vectorizer
with open("model1.pkl", 'rb') as f:
    model = pickle.load(f)
with open("vect.pkl", 'rb') as v:
    vect = pickle.load(v)
#! calling functions
comment_text, like_count_comment = get_comments(youtube, vid_id)
viewCount, likeCount, dislikeCount, commentCount = get_video_stats(
    youtube, vid_id)

pos, neg = calculating_sentiment(comment_text, model, vect)

#! plotting pi chart
lables = ["Positive", "Negative"]
data = [pos, neg]
explode = (0.1, 0.2)
plt.pie(data, explode=explode, labels=lables,
        autopct="%.2f%%", shadow=True, startangle=90)
plt.show()
#! displaying stats
print("\nTotal Views :- ", viewCount, "\nTotal Likes:-",
      likeCount, "\nTotal Dislikes:- ", dislikeCount, "\nTotal Comments:- ", commentCount)
