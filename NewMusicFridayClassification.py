import spotipy
import spotipy.util as util
import random
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt


client_id = '' # TODO: enter client id here
client_secret = '' # TODO: enter client secret here
redirect_url = 'http://localhost/'

username = '' # TODO: enter you Spotify URI username here
scope = 'playlist-modify-private, playlist-modify-public'
token = util.prompt_for_user_token(username, scope, client_id=client_id,
                                   client_secret=client_secret, redirect_uri=redirect_url)
sp = spotipy.Spotify(auth=token)

############################################################################################################

# Define functions that will help throughout the investigation


def get_playlist_tracks(un, pl_id):
    ''' Returns all the tracks from a specified playlist id '''

    results = sp.user_playlist_tracks(un, pl_id)
    tracks = results['items']
    while results['next']:
        results = sp.next(results)
        tracks.extend(results['items'])
    return tracks


def get_track_info(tracks):
    '''Returns a dataframe with all track information from specified track ids '''

    ids = []
    for track in tracks:
        ids.append(track['track']['id'])
    num_tracks = len(ids)

    track_info = []
    for i in range(0, num_tracks, 50):  # get audio features of each track, only do 50 at a time
        track_info += sp.audio_features(ids[i:i+50])

    features_list = []
    for features in track_info:
        features_list.append([features['energy'], features['liveness'],
                              features['tempo'], features['speechiness'],
                              features['acousticness'], features['instrumentalness'],
                              features['time_signature'], features['danceability'],
                              features['key'], features['duration_ms'],
                              features['loudness'], features['valence'],
                              features['mode'], features['type'],
                              features['uri']])

    fin_df = pd.DataFrame(features_list, columns=['energy', 'liveness',
                                              'tempo', 'speechiness',
                                              'acousticness', 'instrumentalness',
                                              'time_signature', 'danceability',
                                              'key', 'duration_ms', 'loudness',
                                              'valence', 'mode', 'type', 'uri'])
    return fin_df


def model_metrics(actual, pred):
    ''' Returns some diagnostics of the suitability of a model based on predicted values
    compared to the predicted values '''
    roc_score = metrics.roc_auc_score(actual, pred)
    acc_score = metrics.accuracy_score(actual, pred)
    rec_score = metrics.recall_score(actual, pred)
    prec_score = metrics.precision_score(actual, pred)

    print('ROC: ', roc_score)
    print('Accuracy: ', acc_score)
    print('Recall: ', rec_score)
    print('Precision:', prec_score)

############################################################################################################

# Initialise data frame with labelled like and disliked songs from playlists manually created on Spotify
# And create training and test splits on the data

# Get all songs in the good and bad playlist and put their features into a pandas data frame
# start only with the song features, could also add in the genre, artist info etc. later


good_playlist_tracks = get_playlist_tracks('21du7dtyidfjcjba4c2jqlgca', '4G4esLd7cj3muK5LaAHvlm')
bad_playlist_tracks = get_playlist_tracks('21du7dtyidfjcjba4c2jqlgca', '6d169744l7umlAG1skFH8D')

good_bad_ind = [1]*len(good_playlist_tracks) + [0]*len(bad_playlist_tracks)

df = get_track_info(good_playlist_tracks)
df = df.append(get_track_info(bad_playlist_tracks))
df = df.reset_index()
df = df.drop(columns=['index', 'time_signature', 'key', 'uri', 'type', 'mode'])
df['rate'] = good_bad_ind

# split into train and test data
train, test = train_test_split(df, train_size = 0.8, random_state = 100)

############################################################################################################
# Exploratory Data Analysis

# Only do visualisations on training data so as to not bias the model creation
train_like = train[train['rate'] == 1]
train_dislike = train[train['rate'] == 0]

# plot the distribution for different features to show difference between liked songs and non liked songs
plt.hist(train_like['danceability'], alpha = 0.4, density = True, color = 'green')
plt.hist(train_dislike['danceability'], alpha = 0.4, density = True, color = 'red')
plt.show()

plt.hist(train_like['energy'], alpha = 0.4, density = True, color = 'green')
plt.hist(train_dislike['energy'], alpha = 0.4, density = True, color = 'red')
plt.show()

plt.hist(train_like['liveness'], alpha = 0.4, density = True, color = 'green')
plt.hist(train_dislike['liveness'], alpha = 0.4, density = True, color = 'red')
plt.show()

plt.hist(train_like['tempo'], alpha = 0.4, density = True, color = 'green')
plt.hist(train_dislike['tempo'], alpha = 0.4, density = True, color = 'red')
plt.show()

plt.hist(train_like['speechiness'], alpha = 0.4, density = True, color = 'green')
plt.hist(train_dislike['speechiness'], alpha = 0.4, density = True, color = 'red')
plt.show()

plt.hist(train_like['acousticness'], alpha = 0.4, density = True, color = 'green')
plt.hist(train_dislike['acousticness'], alpha = 0.4, density = True, color = 'red')
plt.show()

plt.hist(train_like['instrumentalness'], alpha = 0.4, density = True, color = 'green')
plt.hist(train_dislike['instrumentalness'], alpha = 0.4, density = True, color = 'red')
plt.show()

plt.hist(train_like['duration_ms'], alpha = 0.4, density = True, color = 'green')
plt.hist(train_dislike['duration_ms'], alpha = 0.4, density = True, color = 'red')
plt.show()

plt.hist(train_like['loudness'], alpha = 0.4, density = True, color = 'green')
plt.hist(train_dislike['loudness'], alpha = 0.4, density = True, color = 'red')
plt.show()

plt.hist(train_like['valence'], alpha = 0.4, density = True, color = 'green')
plt.hist(train_dislike['valence'], alpha = 0.4, density = True, color = 'red')
plt.show()

############################################################################################################
# Model creation

# Create the features array and target variable for train and test data sets
x_train = train.drop('rate', axis=1)
y_train = train['rate']
x_test = test.drop('rate', axis=1)
y_test = test['rate']

# What is the base classification rate that we need to compare to?
max((sum(y_train)/len(y_train)), 1-(sum(y_train)/len(y_train)))
# we get 56% by classifying all as 0's

# We look over 3 classification algorithms to begin with

# First try k-nearest neighbours

for k in [20, 30, 40, 50, 100]:  # try for different values of k
    knn_train = KNeighborsClassifier(n_neighbors=k)
    knn_train.fit(x_train, y_train)

    k_pred = knn_train.predict(x_test)
    k_act = y_test

    print()
    print('For k = ', k, 'we get:')
    model_metrics(k_act,k_pred)

# From this we can see that k=40 is a good parameter to use, but 63% is not much better than 56% at all


# Random Forest with basic parameters, we can then tune these later
rf_train = RandomForestClassifier()
rf_train.fit(x_train, y_train)

rf_pred = rf_train.predict(x_test)
rf_act = y_test

model_metrics(rf_act, rf_pred)

# This has much better accuracy than k-nn so now we tune the hyper parameters performing grid search
# Basis for grid search adapted from article by Will Koehrsen
# https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features to consider at every split
max_features = ['auto', 'sqrt']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree
bootstrap = [True, False]
grid = {'n_estimators': n_estimators,
           'max_features': max_features,
           'max_depth': max_depth,
           'min_samples_split': min_samples_split,
           'min_samples_leaf': min_samples_leaf,
           'bootstrap': bootstrap}
print(grid)

# perform a randomised search on the grid of parameters to find the best hyper parameters
rf_random = RandomizedSearchCV(estimator = rf_train, param_distributions = grid,
                               n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# fit the model on the training data
rf_random.fit(x_train, y_train)
print(rf_random.best_params_)
print(rf_random.best_estimator_)

# test the accuracy on the test data
rf_pred2 = rf_random.predict(x_test)

model_metrics(rf_act, rf_pred2)

# Now we narrow the search based on the findings from the above grid search
# later versions can adapt this into a function

n_estimators2 = [int(x) for x in np.linspace(start = 1100, stop = 2000, num = 5)]

# Number of features to consider at every split
max_features2 = ['auto', 'sqrt']

# Maximum number of levels in tree
max_depth2 = [int(x) for x in np.linspace(5, 20, num = 10)]
max_depth2.append(None)

# Minimum number of samples required to split a node
min_samples_split2 = [2, 5, 10]

# Minimum number of samples required at each leaf node
min_samples_leaf2 = [1, 2, 4]

# Method of selecting samples for training each tree
bootstrap2 = [True, False]

grid2 = {'n_estimators': n_estimators2,
           'max_features': max_features2,
           'max_depth': max_depth2,
           'min_samples_split': min_samples_split2,
           'min_samples_leaf': min_samples_leaf2,
           'bootstrap': bootstrap2}

# Perform the randomised search on the new grid
rf_random2 = RandomizedSearchCV(estimator = rf_train, param_distributions = grid2, n_iter = 100,
                                cv = 3, verbose=2, random_state=42, n_jobs = -1)

# Fit the model on the training data
rf_random2.fit(x_train, y_train)
print(rf_random2.best_params_)
print(rf_random2.best_estimator_)

# Predict the values of the test data
rf_pred3 = rf_random2.predict(x_test)

model_metrics(rf_act, rf_pred3)


# Better than k-nn, now try SVC and see if it's better than that as well

# SVC Model
# Train on default parameters initially

svc_train = SVC()
svc_train.fit(x_train, y_train)

svc_pred = svc_train.predict(x_test)
svc_act = y_test

model_metrics(svc_act, svc_pred)

# Now narrow the search using a grid search
# defining parameter range
param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']}

grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)

# Fitting the model for grid search
grid.fit(x_train, y_train)

print(grid.best_params_)
print(grid.best_estimator_)

grid_predictions = grid.predict(x_test)
print(metrics.classification_report(y_test, grid_predictions))

# Here we see that the random forests is by far the best model
# so we set the final model to the parameters discovered in the grid search above

final_model = rf_random2

############################################################################################################
# Choose songs from NMF that I will enjoy
# Now we use this final model to predict the songs in New Music Friday that I will enjoy


def nmf_tracks():
    ''' This adapts my Spotify playlist called My NMF based on the most recent New Music Friday '''
    tracks = get_playlist_tracks('21du7dtyidfjcjba4c2jqlgca','37i9dQZF1DX4W3aJJYCDfV')
    df = get_track_info(tracks)
    df = df.drop(columns=['time_signature', 'key', 'uri', 'type', 'mode'])
    pred = final_model.predict(df)
    s = sp.user_playlist_tracks('21du7dtyidfjcjba4c2jqlgca','37i9dQZF1DX4W3aJJYCDfV')
    ids = [t['track']['id'] for t in s['items']]
    to_listen = [ids[i] for i in range(len(ids)) if pred[i] == 1]
    sp.user_playlist_replace_tracks('21du7dtyidfjcjba4c2jqlgca','1J72Ewzf5GR1itw74umyGL', to_listen)


# Run this function to create the playlist in Spotify
nmf_tracks()

# Next project steps is to adapt it for any user rather than just myself
# This would involve comparing algorithm performances across users
# as well as generalising functions for an inputted user
# Could also adapt how the liked and disliked songs are found rather than manually created playlists
