## Recommendation Engine
<b> Using SVD algorithm for collaborative filtering.</b>

* First train a model by using available ratings
* Use that trained model to predict missing rating in users vs items/movies matrix
* With all pridicted rating now users vs items/movies matrix become trained users vs items/movies matrix and we save both in formof .pkl file.
* Later we use any of  users vs items/movies matrix or trained users vs items/movies matrix by TRAINED argument,for further processing.

### First you need to train the model

train.py \
> Other options are following :\
  --data_file       : Input user-movie-rating information file (default: './ratings.dat')\
  --batch_size      : Batch Size (default: 100)\
  --dims"           : Dimensions of SVD (default: 15)\
  --max_epochs      : Dimensions of SVD (default: 25)\
  --checkpoint_dir  : Checkpoint directory from training run (default: '/save/')\
  --val             : True if Folders with files and False if single file\
  --is_gpu          : Want to train model at GPU (default=True)\
  ** Misc Parameters**\
  --allow_soft_placement    :Allow device soft device placement\
  --log_device_placement   :Log placement of ops on devices\

* After training finish, we need to save trained model as well as trained user vs item matrix
  for later use for recommendation.


### Run.py file predict the rating for a given user and movie pair
run.py\
>  --user            : User (default: 1696)")\
  --item            :Movie (default: 3113)")\
  --checkpoint_dir  : Checkpoint directory from training run (default: '/save/')\
  --is_gpu          : Want to train model at GPU (default=True)\
  ** Misc Parameters**\
  --allow_soft_placement    :Allow device soft device placement\
  --log_device_placement   :Log placement of ops on devices\

### For find the K-mean clusters
 kmean.py\
>  --data_file       : Input user-movie-rating information file (default: './ratings.dat')\
  --K               : Number of clusters (default=4)\
  --MAX_ITERS       : Maximum number of iterations (default=1000)\
  --TRAINED         : Use TRAINED user vs item matrix (default=False)\

output of kmean.py saved in clusters.csv file\

Note: Rest How to use this please go through poc.ipynb file
