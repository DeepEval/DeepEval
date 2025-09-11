import random

class MusicPlayer:
    """
    This is a class as a music player that provides to play, stop, add songs, remove songs, set volume, shuffle, and switch to the next or previous song.
    """

    def __init__(self):
        """
        Initializes the music player with an empty playlist, no current song, and a default volume of 50.
        """
        self.playlist = []
        self.current_song_index = -1
        self.volume = 50

    def add_song(self, song):
        """
        Adds a song to the playlist.
        :param song: The song to add to the playlist, str.
        >>> musicPlayer = MusicPlayer()
        >>> musicPlayer.add_song("song1")
        >>> musicPlayer.playlist
        ['song1']

        """
        self.playlist.append(song)

    def remove_song(self, song):
        """
        Removes a song from the playlist.
        :param song: The song to remove from the playlist, str.
        >>> musicPlayer = MusicPlayer()
        >>> musicPlayer.playlist = ["song1", "song2"]
        >>> musicPlayer.remove_song("song1")
        >>> musicPlayer.playlist
        ['song2']

        """
        if song in self.playlist:
            self.playlist.remove(song)
            if self.current_song == song:
                self.current_song = None
                self.current_song_index = -1

    def play(self):
        """
        Plays the current song in the playlist.
        :return: The current song in the playlist, or False if there is no current song.
        >>> musicPlayer = MusicPlayer()
        >>> musicPlayer.playlist = ["song1", "song2"]
        >>> musicPlayer.current_song = "song1"
        >>> musicPlayer.play()
       'song1'

        """
        if self.current_song is not None:
            return self.current_song
        else:
            return False

    def stop(self):
        """
        Stops the current song in the playlist.
        :return: True if the current song was stopped, False if there was no current song.
        >>> musicPlayer = MusicPlayer()
        >>> musicPlayer.playlist = ["song1", "song2"]
        >>> musicPlayer.current_song = "song1"
        >>> musicPlayer.stop()
        True

        """
        self.current_song_index = -1
        return True

    def switch_song(self):
        """
        Switches to the next song in the playlist.
        :return: True if the next song was switched to, False if there was no next song.
        >>> musicPlayer = MusicPlayer()
        >>> musicPlayer.playlist = ["song1", "song2"]
        >>> musicPlayer.current_song = "song1"
        >>> musicPlayer.switch_song()
        True

        """
        if self.playlist:
            if self.current_song_index < len(self.playlist) - 1:
                self.current_song_index += 1
                self.current_song = self.playlist[self.current_song_index]
                return True
        return False

    def previous_song(self):
        """
        Switches to the previous song in the playlist.
        :return: True if the previous song was switched to, False if there was no previous song.
        >>> musicPlayer = MusicPlayer()
        >>> musicPlayer.playlist = ["song1", "song2"]
        >>> musicPlayer.current_song = "song2"
        >>> musicPlayer.previous_song()
        True

        """
        if self.playlist:
            if self.current_song_index > 0:
                self.current_song_index -= 1
                self.current_song = self.playlist[self.current_song_index]
                return True
        return False

    def set_volume(self, volume):
        """
        Sets the volume of the music player, if the volume is between 0 and 100 is valid.
        :param volume: The volume to set the music player to, int.
        :return: True if the volume was set, False if the volume was invalid.
        >>> musicPlayer = MusicPlayer()
        >>> musicPlayer.set_volume(50)
        >>> musicPlayer.volume
        50

        """
        if 0 <= volume <= 100:
            self.volume = volume
            return True
        return False

    def shuffle(self):
        """
        Shuffles the playlist.
        :return: True if the playlist was shuffled, False if the playlist was empty.
        >>> musicPlayer = MusicPlayer()
        >>> musicPlayer.playlist = ["song1", "song2"]
        >>> musicPlayer.shuffle()
        True

        """
        if self.playlist:
            random.shuffle(self.playlist)
            self.current_song_index = -1
            return True
        return False

if __name__ == "__main__":
    musicPlayer = MusicPlayer()

    # Test case for add_song method
    musicPlayer.add_song("song1")
    print(musicPlayer.playlist)  # Output: ['song1']
    musicPlayer.add_song("song2")
    print(musicPlayer.playlist)  # Output: ['song1','song2']

    # Test case for remove_song method
    musicPlayer.remove_song("song1")
    print(musicPlayer.playlist)  # Output: ['song2']

    # Test case for play method
    musicPlayer.current_song = "song1"
    print(musicPlayer.play())  # Output:'song1'
    musicPlayer.play()  # Output: False

    # Test case for stop method
    musicPlayer.current_song = "song1"
    print(musicPlayer.stop())  # Output: True

    # Test case for switch_song method
    musicPlayer.current_song = "song1"
    musicPlayer.playlist = ["song1", "song2", "song3"]
    print(musicPlayer.switch_song())  # Output: True
    print(musicPlayer.current_song)  # Output:'song2'

    # Test case for previous_song method
    musicPlayer.current_song = "song2"
    print(musicPlayer.previous_song())  # Output: True
    print(musicPlayer.current_song)  # Output:'song1'

    # Test case for set_volume method
    print(musicPlayer.set_volume(50))  # Output: True
    print(musicPlayer.volume)  # Output: 50

    # Test case for shuffle method
    musicPlayer.playlist = ["song1", "song2", "song3"]
    print(musicPlayer.shuffle())  # Output: True
    print(musicPlayer.playlist)  # Output: a shuffled list