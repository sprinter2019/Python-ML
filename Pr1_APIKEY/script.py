import http.client
import json
import time
import sys
import collections

start_time = time.time()
#Parse API key
api_key=sys.argv[1]

base_url= 'api.themoviedb.org'

# Keeps track of total number of requests
req_counter=0

# Search Criteria
movie_genre='Drama'
earliest_date='2004'
total_movies=350
similar_movies=5
popularity='desc'

#initiate connection

conn = http.client.HTTPSConnection(base_url)
payload = "{}"


# Get Genre Id
def get_GenreID(query_genre):
    global req_counter
    conn.request("GET", query_genre, payload)
    req_counter+=1
    json_genre= conn.getresponse().read()
    genre_data= json.loads(json_genre)
    
    for item in genre_data['genres']:
        if item['name']==movie_genre:
            genre_id=item['id']
            
    return genre_id

query_genre="/3/genre/movie/list?api_key="+api_key
genre_id= get_GenreID(query_genre)


def popularMovies(movie_limit):
    
    global req_counter
    movie_ids=[]
    mv_counter=1
    
    f=open("movie_ID_name.csv","w+")
    
    for page in range(1000):
  
        query_data= "/3/discover/movie?with_genres="+str(genre_id)+"&page="+str(page+1)+"&primary_release_date.gte="+str(earliest_date)+"&sort_by=popularity."+str(popularity)+"&api_key="+api_key
        
        conn.request("GET", query_data, payload)
        req_counter+=1
        
        json_data = conn.getresponse().read()
        movie_data=json.loads(json_data)
               
        
        for item in movie_data['results']:
            movie_ids.append(item['id'])
            if ',' in item['title']:
                file_data=str(item['id'])+','+'"'+str(item['title']+'"'+'\n')
            else:
                file_data=str(item['id'])+','+str(item['title']+'\n')
            print(file_data)
            f.write(file_data)
            mv_counter+=1
            if mv_counter>movie_limit:        
                break
        
        if mv_counter>movie_limit:
            f.close()
            break
    
    return movie_ids
        

popular_movie_ids=popularMovies(total_movies)

# Retrieve similar movie ids
def similarMovies(movie_id):
    global req_counter
    save_data=[]
    for i in range(len(movie_id)):
        if req_counter%40==0:
            t_wait=11+1-(time.time()-start_time)%11
            time.sleep(t_wait)
            
        query_datax= "/3/movie/"+str(movie_id[i])+"/similar?api_key="+api_key
        conn.request("GET", query_datax, payload)
        req_counter+=1
        
        json_datax = conn.getresponse().read()
        similar_movie_data=json.loads(json_datax)
        j=0
        for item in similar_movie_data['results']:
            j+=1
            if movie_id[i]<item['id']:
                save_data.append([movie_id[i],item['id']])
            else:
                save_data.append([item['id'],movie_id[i]])
            if j==similar_movies:
                break
            
    return save_data   
        #print(similar_movie_data)

similar_mv=similarMovies(popular_movie_ids)

# Refine similar movie data to remove duplicate entries
def refineData(mvData):
    f=open("movie_ID_sim_movie_ID.csv","w+")
    dataX=[]
    for item in mvData:
        if item not in dataX:
            dataX.append(str(item[0])+','+str(item[1])+'\n')
            f.write(str(item[0])+','+str(item[1])+'\n')   
    f.close()     


refineData(similar_mv)


print("Runtime= %s seconds" % (time.time() - start_time))
#print("Total API requests= %s" % req_counter)
    
conn.close()

