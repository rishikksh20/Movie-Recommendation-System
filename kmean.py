import tensorflow as tf
import numpy as np
import pandas as pd
import time
import main
import readers


# Data Parameters
tf.flags.DEFINE_string("data_file", "./ratings.dat", "Data source for the positive data.")
tf.flags.DEFINE_string("K", 4, "Number of clusters")
tf.flags.DEFINE_string("MAX_ITERS", 1000, "Maximum number of iterations")
tf.flags.DEFINE_string("TRAINED", False, "Use TRAINED user vs item matrix")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")
# Read the main file i.e. ratings.dat
ratings_df = readers.read_file(FLAGS.data_file, sep="::")
clusters,movies=main.k_mean_clustering(ratings_df,K=FLAGS.K,MAX_ITERS = FLAGS.MAX_ITERS,TRAINED=FLAGS.TRAINED)
cluster_df=pd.DataFrame({'movies':movies,'clusters':clusters})
cluster_df.to_csv("clusters.csv")
