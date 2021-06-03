#Importing Libraries

from tweepy import OAuthHandler
import pandas as pd
import tweepy
import pre
import warnings
warnings.filterwarnings('ignore')

#Getting the secret keys for Twitter API
f=open('E:/My/SRM UNIVERSITY/3rd year/Sem 6/Seminar/Model/secret_keys.txt','r')
k=f.readlines()
access_token = k[0][:-1]
access_secret = k[1][:-1]
consumer_key = k[2][:-1]
consumer_secret = k[3]

#Srapping Tweets from Twitter using Twitter API

def scrape(words, numtweet):
    db = pd.DataFrame(columns=['user', 'text'])

    tweets = tweepy.Cursor(api.search, q=words, lang="en", tweet_mode='extended').items(numtweet)

    list_tweets = [tweet for tweet in tweets]
    i = 1

    #Getting Username, Followers and Hashtags used in the Tweet Parsed

    for tweet in list_tweets:
        username = tweet.user.screen_name
        followers = tweet.user.followers_count
        hashtags = tweet.entities['hashtags']

        try:
            text = tweet.retweeted_status.full_text
        except AttributeError:
            text = tweet.full_text
        hashtext = list()
        for j in range(0, len(hashtags)):
            hashtext.append(hashtags[j]['text'])

        #Storing tweets of those who have followers more than 10000

        if followers>10000:
            ith_tweet = [username, text]
            db.loc[len(db)] = ith_tweet

            i = i+1

    #Calling prediction

    p=pre.prediction(db)
    print(p)
    filename = 'scraped_tweets.csv'
    print('\nStoring output to ', filename)
    p.to_csv(filename)

#Authorizing Twitter API

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
api = tweepy.API(auth)

#Getting user input

words = input("Enter Twitter HashTag to search for: ")

numtweet = 1000
print("\nScrapping Tweets....")
scrape(words, numtweet)