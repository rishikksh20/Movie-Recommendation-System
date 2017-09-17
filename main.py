import pandas as pd
import numpy as np
from run import prediction
import tensorflow as tf
import time
import os

np.random.seed(42)

def top_k_movies(users,ratings_df,k):
    """
    Returns top k movies for respective user
    INPUTS :
        users      : list of numbers or number , list of user ids
        ratings_df : rating dataframe, store all users rating for respective movies
        k          : natural number
    OUTPUT:
        Dictionary conatining user id as key and list of top k movies for that user as value
    """
    # Extract unseen movies
    dicts={}
    if type(users) is not list:
        users=[users]
    for user in users:
        rated_movies=ratings_df[ratings_df['user']==user].drop(['st', 'user'], axis=1)
        rated_movie=list(rated_movies['item'].values)
        total_movies=list(ratings_df.item.unique())
        unseen_movies=list(set(total_movies) - set(rated_movie))
        rated_list = []
        rated_list=prediction(np.full(len(unseen_movies),user),np.array(unseen_movies))
        useen_movies_df=pd.DataFrame({'item': unseen_movies,'rate':rated_list})
        top_k=list(useen_movies_df.sort(['rate','item'], ascending=[0, 0])['item'].head(k).values)
        dicts.update({user:top_k})
    result=pd.DataFrame(dicts)
    result.to_csv("user_top_k.csv")
    return dicts


def user_rating(users,movies):
    """
    Returns user rating for respective user
    INPUTS :
        users      : list of numbers or number, list of user ids or just user id
        movies : list of numbers or number, list of movie ids or just movie id
    OUTPUT:
        list of predicted movies
    """
    if type(users) is not list:
        users=np.array([users])
    if type(movies) is not list:
        movies=np.array([movies])
    return prediction(users,movies)

def top_k_similar_items(movies,ratings_df,k,TRAINED=False):
    """
    Returns k similar movies for respective movie
    INPUTS :
        movies : list of numbers or number, list of movie ids
        ratings_df : rating dataframe, store all users rating for respective movies
        k          : natural number
        TRAINED    : TRUE or FALSE, weather use trained user vs movie table or untrained
    OUTPUT:
        list of k similar movies for respected movie
    """
    if TRAINED:
        df=pd.read_pickle("user_item_table_train.pkl")
    else:
        df=pd.read_pickle("user_item_table.pkl")

    corr_matrix=item_item_correlation(df,TRAINED)
    if type(movies) is not list:
        return corr_matrix[movies].sort_values(ascending=False).drop(movies).index.values[0:k]
    else:
        dict={}
        for movie in movies:
            dict.update({movie:corr_matrix[movie].sort_values(ascending=False).drop(movie).index.values[0:k]})
        pd.DataFrame(dict).to_csv("movie_top_k.csv")
        return dict

def user_similarity(user_1,user_2,ratings_df,TRAINED=False):
    """
    Return the similarity between two users
    INPUTS :
        user_1,user_2 : number, respective user ids
        ratings_df : rating dataframe, store all users rating for respective movies
        TRAINED    : TRUE or FALSE, weather use trained user vs movie table or untrained
    OUTPUT:
        Pearson cofficient between users [value in between -1 to 1]
    """
    corr_matrix=user_user_pearson_corr(ratings_df,TRAINED)

    return corr_matrix[user_1][user_2]


def item_item_correlation(df,TRAINED):
    if TRAINED:
        if os.path.isfile("./item_item_corr_train.pkl"):
            df_corr=pd.read_pickle("item_item_corr_train.pkl")
        else:
            df_corr=df.corr()
            df_corr.to_pickle("item_item_corr_train.pkl")
    else:
        if os.path.isfile("./item_item_corr.pkl"):
            df_corr=pd.read_pickle("item_item_corr.pkl")
        else:
            df_corr=df.corr()
            df_corr.to_pickle("item_item_corr.pkl")
    return df_corr


def user_user_pearson_corr(ratings_df,TRAINED):
    if TRAINED:
        if os.path.isfile("./user_user_corr_train.pkl"):
            df_corr=pd.read_pickle("user_user_corr_train.pkl")
        else:
            df =pd.read_pickle("user_item_table_train.pkl")
            df=df.T
            df_corr=df.corr()
            df_corr.to_pickle("user_user_corr_train.pkl")
    else:
        if os.path.isfile("./user_user_corr.pkl"):
            df_corr=pd.read_pickle("user_user_corr.pkl")
        else:
            df = pd.read_pickle("user_item_table.pkl")
            df=df.T
            df_corr=df.corr()
            df_corr.to_pickle("user_user_corr.pkl")
    return df_corr

def bucket_mean(data, bucket_ids, num_buckets):
    total = tf.unsorted_segment_sum(data, bucket_ids, num_buckets)
    count = tf.unsorted_segment_sum(tf.ones_like(data), bucket_ids, num_buckets)
    return total / count

def k_mean_clustering(ratings_df,K=4,MAX_ITERS = 1000,TRAINED=False):
    """
    Return movies alongwith there respective clusters
    INPUTS :
        ratings_df : rating dataframe, store all users rating for respective movies
        K          : number of clusters
        MAX_ITERS  : maximum number of recommendation
        TRAINED    : TRUE or FALSE, weather use trained user vs movie table or untrained
    OUTPUT:
        List of movies/items and list of clusters
    """
    if TRAINED:
        df=pd.read_pickle("user_item_table_train.pkl")
    else:
        df=pd.read_pickle("user_item_table.pkl")
    df = df.T

    start = time.time()
    N=df.shape[0]

    points = tf.Variable(df.as_matrix())
    cluster_assignments = tf.Variable(tf.zeros([N], dtype=tf.int64))

    # Silly initialization:  Use the first K points as the starting
    # centroids.  In the real world, do this better.
    centroids = tf.Variable(tf.slice(points.initialized_value(), [0,0], [K,df.shape[1]]))

    # Replicate to N copies of each centroid and K copies of each
    # point, then subtract and compute the sum of squared distances.
    rep_centroids = tf.reshape(tf.tile(centroids, [N, 1]), [N, K, df.shape[1]])
    rep_points = tf.reshape(tf.tile(points, [1, K]), [N, K, df.shape[1]])
    sum_squares = tf.reduce_sum(tf.square(rep_points - rep_centroids),
                                reduction_indices=2)

    # Use argmin to select the lowest-distance point
    best_centroids = tf.argmin(sum_squares, 1)
    did_assignments_change = tf.reduce_any(tf.not_equal(best_centroids,
                                                        cluster_assignments))



    means = bucket_mean(points, best_centroids, K)

    # Do not write to the assigned clusters variable until after
    # computing whether the assignments have changed - hence with_dependencies
    with tf.control_dependencies([did_assignments_change]):
        do_updates = tf.group(
            centroids.assign(means),
            cluster_assignments.assign(best_centroids))

    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)

    changed = True
    iters = 0

    while changed and iters < MAX_ITERS:
        iters += 1
        [changed, _] = sess.run([did_assignments_change, do_updates])

    [centers, assignments] = sess.run([centroids, cluster_assignments])
    end = time.time()
    print (("Found in %.2f seconds" % (end-start)), iters, "iterations")
    return assignments,df.index.values;
