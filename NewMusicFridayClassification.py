import spotipy
import spotipy.util as util
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt


client_id = '' # TODO: enter client id here
client_secret = '' # TODO: enter client secret here
redirect_url = 'http://localhost/'

username = '' # TODO: enter you Spotify URI username here
scope = 'playlist-modify-private, playlist-modify-public'
token = util.prompt_for_user_token(username, scope, client_id=client_id, client_secret=client_secret, redirect_uri=redirect_url)
sp = spotipy.Spotify(auth=token)


def get_playlist_tracks(un, plid):
    results = sp.user_playlist_tracks(un, plid)
    tracks = results['items']
    while results['next']:
        results = sp.next(results)
        tracks.extend(results['items'])
    return tracks


def get_track_playlists(tracks):
    ids = []
    for track in tracks:
        ids.append(track['track']['id'])
    num_tracks = len(ids)

    track_info = []
    for i in range(0, num_tracks, 50):
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

    df = pd.DataFrame(features_list, columns=['energy', 'liveness',
                                              'tempo', 'speechiness',
                                              'acousticness', 'instrumentalness',
                                              'time_signature', 'danceability',
                                              'key', 'duration_ms', 'loudness',
                                              'valence', 'mode', 'type', 'uri'])
    return df


# get all songs in the good and bad playlist and put their features into a pandas data frame
# start only with the song features, also add in the genre, artist info etc. later
good_playlist_tracks = get_playlist_tracks('21du7dtyidfjcjba4c2jqlgca', '4G4esLd7cj3muK5LaAHvlm')
bad_playlist_tracks = get_playlist_tracks('21du7dtyidfjcjba4c2jqlgca', '6d169744l7umlAG1skFH8D')

good_bad_ind = [1]*len(good_playlist_tracks) + [0]*len(bad_playlist_tracks)

df = get_track_playlists(good_playlist_tracks)
df = df.append(get_track_playlists(bad_playlist_tracks))
df = df.reset_index()
df = df.drop(columns=['index', 'time_signature', 'key', 'uri', 'type', 'mode'])
df['rate'] = good_bad_ind


train, test = train_test_split(df, train_size = 0.8, random_state = 100)

train_like = train[train['rate'] == 1]
train_dislike = train[train['rate'] == 0]

#EDA
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

# create the features array and target variable
x_train = train.drop('rate', axis=1)
y_train = train['rate']
x_test = test.drop('rate', axis=1)
y_test = test['rate']

# what is the base classification rate that we need to beat?
max((sum(y_train)/len(y_train)),1-(sum(y_train)/len(y_train)))
# we get 56% by classifying all as 0's

# look over 3 main classification algorithms to begin with
# first try k-nearest neighbours
for k in [20, 30, 40, 50, 100]:
    knn_train = KNeighborsClassifier(n_neighbors=k)
    knn_train.fit(x_train, y_train)

    k_pred = knn_train.predict(x_test)
    k_act = y_test

    roc_score = metrics.roc_auc_score(k_act, k_pred)
    acc_score = metrics.accuracy_score(k_act, k_pred)
    rec_score = metrics.recall_score(k_act, k_pred)
    prec_score = metrics.precision_score(k_act, k_pred)

    print()
    print('For k = ', k, 'we get:')
    print('ROC: ', roc_score)
    print('Accuracy: ', acc_score)
    print('Recall: ', rec_score)
    print('Precision:', prec_score)

# From this we can see that k=40 is a good parameter to use, but 63% is not much better than 56% at all

# Now we try random forests first with basic parameters, we can then tune these later
rf_train = RandomForestClassifier()
rf_train.fit(x_train, y_train)

rf_pred = rf_train.predict(x_test)
rf_act = y_test

roc_score = metrics.roc_auc_score(rf_act, rf_pred)
acc_score = metrics.accuracy_score(rf_act, rf_pred)
rec_score = metrics.recall_score(rf_act, rf_pred)
prec_score = metrics.precision_score(rf_act, rf_pred)

print('ROC: ', roc_score)
print('Accuracy: ', acc_score)
print('Recall: ', rec_score)
print('Precision:', prec_score)

# much better accuracy than k-nn so now we tune the hyper parameters
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
#perform a randomised search on the grid of parameters to find the best hyper parameters
rf_random = RandomizedSearchCV(estimator = rf_train, param_distributions = grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
rf_random.fit(x_train, y_train)
rf_random.best_params_
rf_random.best_estimator_

rf_pred2 = rf_random.predict(x_test)

roc_score = metrics.roc_auc_score(rf_act, rf_pred2)
acc_score = metrics.accuracy_score(rf_act, rf_pred2)
rec_score = metrics.recall_score(rf_act, rf_pred2)
prec_score = metrics.precision_score(rf_act, rf_pred2)
print('ROC: ', roc_score)
print('Accuracy: ', acc_score)
print('Recall: ', rec_score)
print('Precision:', prec_score)

# Now we narrow the search
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
rf_random2 = RandomizedSearchCV(estimator = rf_train, param_distributions = grid2, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
rf_random2.fit(x_train, y_train)
rf_random2.best_params_
rf_random2.best_estimator_

rf_pred3 = rf_random2.predict(x_test)

roc_score = metrics.roc_auc_score(rf_act, rf_pred3)
acc_score = metrics.accuracy_score(rf_act, rf_pred3)
rec_score = metrics.recall_score(rf_act, rf_pred3)
prec_score = metrics.precision_score(rf_act, rf_pred3)
print('ROC: ', roc_score)
print('Accuracy: ', acc_score)
print('Recall: ', rec_score)
print('Precision:', prec_score)

final_model = rf_random2

# Better than k-nn, now try SVC and see if it's better than that as well

svc_train = SVC()
svc_train.fit(x_train, y_train)

svc_pred = svc_train.predict(x_test)
svc_act = y_test

roc_score = metrics.roc_auc_score(svc_act, svc_pred)
acc_score = metrics.accuracy_score(svc_act, svc_pred)
rec_score = metrics.recall_score(svc_act, svc_pred)
prec_score = metrics.precision_score(svc_act, svc_pred)

print('ROC: ', roc_score)
print('Accuracy: ', acc_score)
print('Recall: ', rec_score)
print('Precision:', prec_score)



# defining parameter range
param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']}

grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)

# fitting the model for grid search
grid.fit(x_train, y_train)

print(grid.best_params_)
print(grid.best_estimator_)

grid_predictions = grid.predict(x_test)
print(metrics.classification_report(y_test, grid_predictions))


test_pl = get_playlist_tracks('21du7dtyidfjcjba4c2jqlgca', '2Q13388Htr4F9rMuUuv7La')

df_test = get_track_playlists(test_pl)
df_test = df_test.drop(columns=['time_signature', 'key', 'uri', 'type', 'mode'])

final_model.predict(df_test)


def nmf_tracks():
    tracks = get_playlist_tracks('21du7dtyidfjcjba4c2jqlgca','37i9dQZF1DX4W3aJJYCDfV')
    df = get_track_playlists(tracks)
    df = df.drop(columns=['time_signature', 'key', 'uri', 'type', 'mode'])
    pred = final_model.predict(df)
    s = sp.user_playlist_tracks('21du7dtyidfjcjba4c2jqlgca','37i9dQZF1DX4W3aJJYCDfV')
    ids = [t['track']['id'] for t in s['items']]
    names = [t['track']['name'] for t in s['items']]
    names = [names[i] for i in range(len(names)) if pred[i] == 1]
    to_listen = [ids[i] for i in range(len(ids)) if pred[i] == 1]
    sp.user_playlist_replace_tracks('21du7dtyidfjcjba4c2jqlgca','1J72Ewzf5GR1itw74umyGL', to_listen)


nmf_tracks()

