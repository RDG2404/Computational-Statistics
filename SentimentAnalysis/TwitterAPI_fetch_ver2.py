import tweepy
import pandas as pd
import csv
import re 
import string
import preprocessor as p
import io
import re
import datetime


# Twitter API access tokens
consumer_key = 'tpzOojAJ6KxsoNTnansWU6665'
consumer_secret = 'G7aIqSMfrELwzptV4G9pgOBAm3tr1mvIML7Hl8kBivDvXwUkyc'
access_token = '987583707346358272-y5cPBhuOpHStAdy2lmYGJEfZHNAqvoZ'
access_token_secret = 'JN151eja8TFbGYTyIUJD3KmDwbN2vjJ2yGPbXhWkvpe4c'


# create OAuthHandler object
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
# set access token and secret
auth.set_access_token(access_token, access_token_secret)
# create tweepy API object to fetch tweets
api = tweepy.API(auth,wait_on_rate_limit=True)


# twitter handles
user1 = 'BloombergNRG'
user2 = 'ftenergy'
user3 = 'financialtimes'
user4 = 'EnergyLiveNews'
user5 = 'huffpost'
# keywords
k1 = 'oil'
k2 = 'energy'
k3 = 'gas'
k4 = 'brent'
k5 = 'crude'
k6 = 'WTI'
limit=50000 # total number of tweets fetched

#dates
start_date = datetime.datetime(2021, 11, 1, 0, 00, 00)
end_date = datetime.datetime(2021, 11, 15, 0, 00, 00)

# for users
tweets = tweepy.Cursor(api.user_timeline, screen_name=user5 ,count=200, tweet_mode='extended').items(limit) # cahneg users here

# # for hashtags
# tweets = tweepy.Cursor(api.search_tweets, q=keywords, count=100, tweet_mode='extended').items(limit)

# create dataframe
columns = ['User', 'Tweet', 'Created At']
data=[]

for tweet in tweets:
    if k1 in tweet.full_text or k2 in tweet.full_text or k3 in tweet.full_text or k4 in tweet.full_text or k5 in tweet.full_text or k6 in tweet.full_text: # checking if keywords exist in the tweets
        data.append([tweet.user.screen_name, tweet.full_text, str(tweet.created_at)[0:10]]) # removed time from date for compatibility with excel

# writing to excel

df=pd.DataFrame(data, columns=columns)
df.to_excel(r"C:\Users\rishi\OneDrive\Desktop\Georgia Tech\Projects\Comp Stat Project\Data\Tweets2.xlsx", index=False ) # set excel file path here 
print(df)