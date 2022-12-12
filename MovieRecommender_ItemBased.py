import numpy as np
from openpyxl import Workbook
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import csv
df=pd.read_excel(r"C:\Users\rishi\OneDrive\Desktop\ratings.xlsx", index_col=0)
#df=pd.DataFrame(df)
titles=pd.read_excel(r"C:\Users\rishi\OneDrive\Desktop\movietitle.xlsx")
titles=np.asarray(titles)
df1=df.copy()
metric='hamming'



def recommend_movies(user, num_recommended_movies):
  recommended_movies = []
  rec=[]
  for m in df[df[user] == 0].index.tolist():

    index_df = df.index.tolist().index(m)
    predicted_rating = df1.iloc[index_df, df1.columns.tolist().index(user)]
    recommended_movies.append((m, predicted_rating))
  sorted_rm = sorted(recommended_movies, key=lambda x:x[1], reverse=True)

  rank = 1
  for recommended_movie in sorted_rm[:num_recommended_movies]:
    
    print('{}: {} - predicted rating:{}'.format(rank, str(titles[recommended_movie[0]-1]), recommended_movie[1])) #for item-based
    #print('{}: {} - predicted rating:{}'.format(rank, recommended_movie[0], recommended_movie[1]))#for user-based
    rank = rank + 1
    rec.append((str(titles[recommended_movie[0]-1]),  recommended_movie[1])) #rec.append((1,2))
  #writetocsv(rank,rec)
  #writetocsv(rank, str(titles[recommended_movie[0]-1]), recommended_movie[1])




def movie_recommender(user, num_neighbors, num_recommendation):
  
  number_neighbors = num_neighbors

  knn = NearestNeighbors(metric=metric, algorithm='brute')
  knn.fit(df.values)
  distances, indices = knn.kneighbors(df.values, n_neighbors=number_neighbors)

  user_index = df.columns.tolist().index(user)

  for m,t in list(enumerate(df.index)):
    if df.iloc[m, user_index] == 0:
      sim_movies = indices[m].tolist()
      movie_distances = distances[m].tolist()
    
      if m in sim_movies:
        id_movie = sim_movies.index(m)
        sim_movies.remove(m)
        movie_distances.pop(id_movie) 

      else:
        sim_movies = sim_movies[:num_neighbors-1]
        movie_distances = movie_distances[:num_neighbors-1]
           
      movie_similarity = [1-x for x in movie_distances]
      movie_similarity_copy = movie_similarity.copy()
      nominator = 0

      for s in range(0, len(movie_similarity)):
        if df.iloc[sim_movies[s], user_index] == 0:
          if len(movie_similarity_copy) == (number_neighbors - 1):
            movie_similarity_copy.pop(s)
          
          else:
            movie_similarity_copy.pop(s-(len(movie_similarity)-len(movie_similarity_copy)))
            
        else:
          nominator = nominator + movie_similarity[s]*df.iloc[sim_movies[s],user_index]
          
      if len(movie_similarity_copy) > 0:
        if sum(movie_similarity_copy) > 0:
          predicted_r = nominator/sum(movie_similarity_copy)
        
        else:
          predicted_r = 0

      else:
        predicted_r = 0
        
      df1.iloc[m,user_index] = predicted_r
  recommend_movies(user,num_recommendation)


def writetocsv(user, rec):
  with open(r"C:\Users\rishi\OneDrive\Desktop\ItemBased.csv", 'w', newline='') as csvfile:
    fieldnames=['User No.', 'Movie1','Score1','Movie2','Score2','Movie3','Score3','Movie4','Score4','Movie5','Score5']
    thewriter=csv.DictWriter(csvfile, fieldnames=fieldnames)
    thewriter.writeheader()
    thewriter.writerow({'User No.':user, 'Movie1':rec[0][0],'Score1':rec[0][1],'Movie2':rec[1][0],'Score2':rec[1][1], 'Movie3':rec[2][0], 'Score3':rec[2][1],'Movie4':rec[3][0],'Score4':rec[3][1], 'Movie5':rec[4][0],'Score5':rec[4][1] })
  
  # print('Shape: ', np.shape(rec))
  # print(rec[0])

  print(rec)

for i in range(46):
  movie_recommender(i+1, 3, 5) #user number, num_neighbours, num_recommendations
  print("\n")

#print(titles[0])
#print(df)