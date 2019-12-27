import spotipy
import spotipy.util as util
import random
import pandas as pd

client_id = '' # TODO: enter client id here
client_secret = '' # # TODO: enter client secret here
redirect_url = 'http://localhost/'

username = '' # TODO: enter you Spotify URI username here
scope = 'playlist-modify-private, playlist-modify-public'
token = util.prompt_for_user_token(username, scope, client_id=client_id, client_secret=client_secret, redirect_uri=redirect_url)
sp = spotipy.Spotify(auth=token)

print()
print("Hello and welcome to this custom playlist creator!")
print()
print("Can you enter three different artists that you would like to be on the playlist?")
print()
search1 = input("Who is the first artist you would like to be on the list?")
art1results = sp.search(search1,1,0,'artist')
artist1 = art1results['artists']['items'][0]

search2 = input("Who is the second artist you would like to be on the list?")
art2results = sp.search(search2,1,0,'artist')
artist2 = art2results['artists']['items'][0]

search3 = input("Who is the third artist you would like to be on the list?")
art3results = sp.search(search3,1,0,'artist')
artist3 = art3results['artists']['items'][0]

print()
print("Brilliant! So the three artists you have chosen are: " + artist1["name"] + ", " + artist2["name"] + ", " + artist3["name"])

# Input number of songs wanted on the playlist
print()
songnum = int(input("Now, how many songs do you want on the playlist?"))
print()

#List of artist objects
ArtistList = [artist1, artist2, artist3]
print("ArtistListDone")

#figure out split of number of songs per artist
remainder = songnum % 3
integer = int(songnum/3)
splits = []
for i in range(3):
    splits.append(integer)
for i in range(remainder):
    splits[i] += 1

# gather tracks for each artist
playlist_tracks = []
s = 0
for artist in ArtistList:
    print("Artist ", s)

    # get all the albums from artist 1
    sp_albums = sp.artist_albums(artist['id'], album_type='album')

    # Store artist's albums' names' and uris in separate lists
    album_names = []
    album_uris = []

    # save each album id to the list
    for i in range(len(sp_albums['items'])):
        album_names.append(sp_albums['items'][i]['name'])
        album_uris.append(sp_albums['items'][i]['uri'])

    # create list object for the song ids
    song_list_ids = []

    # add each track from each album to the song list id list
    for i in album_uris:
        album_tracks = sp.album_tracks(i)['items']
        for track in album_tracks:
            song_list_ids.append(track['id'])

    # get the track info for each of the song ids created
    tracks = []
    for p in range(0, len(song_list_ids), 50):  # can only do 50 at a time
        song_list = sp.tracks(song_list_ids[p:p+50])
        for song in song_list['tracks']:
            song_info = {'id': song['id'], 'name': song['name'], 'popularity': song['popularity']}
            tracks.append(song_info)

    # create data frame with the track info list and sort by popularity
    artist_tracks = pd.DataFrame(tracks).sort_values(by='popularity',ascending=False)

    # pick a proportion of the top 20 songs of the artist based on artists popularity
    thresh = round(0.2 * int(artist['popularity']))

    # create a list of the tracks and then add a random "split" number of the "thresh" top tracks to the playlist list
    ids = artist_tracks['id'].to_list()
    playlist_tracks.extend(random.sample(ids[0:thresh], splits[s]))
    s += 1

# create the playlist and then save the id of the new playlist
sp.user_playlist_create(username, 'Random ' + artist1['name'] + ', ' + artist2['name'] + ', and ' + artist3['name'] + ' songs')
new_playlist = sp.user_playlists(username)['items'][0]['id']
random.shuffle(playlist_tracks)


# add each of the tracks to the playlist
for i in playlist_tracks:
    sp.user_playlist_add_tracks(username, new_playlist, [i])
