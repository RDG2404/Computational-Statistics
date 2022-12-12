import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
df=pd.read_excel(r"C:\Users\rishi\OneDrive\Desktop\ratings.xlsx", index_col=0)
df=pd.DataFrame(df).T #Transpose of original dataframe
titles=pd.read_excel(r"C:\Users\rishi\OneDrive\Desktop\movietitle.xlsx")
titles=np.asarray(titles)
df1=df.copy()
metric='hamming'


def recommend_movies(user, num_recommended_movies):
  recommended_movies = []

  for m in df[df[user] == 0].index.tolist():

    index_df = df.index.tolist().index(m)
    predicted_rating = df1.iloc[index_df, df1.columns.tolist().index(user)]
    recommended_movies.append((m, predicted_rating))

  sorted_rm = sorted(recommended_movies, key=lambda x:x[1], reverse=True)
  
  rank = 1
  for recommended_movie in sorted_rm[:num_recommended_movies]:
    
    #print('{}: {} - predicted rating:{}'.format(rank, str(titles[recommended_movie[0]-1]), recommended_movie[1])) #for item-based
    print('{}: User {} - predicted rating:{}'.format(rank, recommended_movie[0], recommended_movie[1])) #for user-based
    rank = rank + 1


def movie_recommender(movie, num_neighbors, num_recommendation):
  
  number_neighbors = num_neighbors

  knn = NearestNeighbors(metric=metric, algorithm='brute')
  knn.fit(df.values)
  distances, indices = knn.kneighbors(df.values, n_neighbors=number_neighbors)

  movie_index = df.columns.tolist().index(movie)

  for m,t in list(enumerate(df.index)):
    if df.iloc[m, movie_index] == 0:
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
        if df.iloc[sim_movies[s], movie_index] == 0:
          if len(movie_similarity_copy) == (number_neighbors - 1):
            movie_similarity_copy.pop(s)
          
          else:
            movie_similarity_copy.pop(s-(len(movie_similarity)-len(movie_similarity_copy)))
            
        else:
          nominator = nominator + movie_similarity[s]*df.iloc[sim_movies[s],movie_index]
          
      if len(movie_similarity_copy) > 0:
        if sum(movie_similarity_copy) > 0:
          predicted_r = nominator/sum(movie_similarity_copy)
        
        else:
          predicted_r = 0

      else:
        predicted_r = 0
        
      df1.iloc[m,movie_index] = predicted_r
  recommend_movies(movie,num_recommendation)

#For all movies
for i in range(20):
  movie_recommender(i+1, 5, 5) #movie number, num_neighbours, num_recommendations
  print('\n')
#print(titles[0])