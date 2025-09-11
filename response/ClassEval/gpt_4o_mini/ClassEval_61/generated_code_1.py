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
        self.current_song_index = -1  # Index of the current song
        self.volume = 50

    def add_song(self, song):
        """
        Adds a song to the playlist.
        :param song: The song to add to the playlist, str.
        """
        self.playlist.append(song)

    def remove_song(self, song):
        """
        Removes a song from the playlist.
        :param song: The song to remove from the playlist, str.
        """
        if song in self.playlist:
            self.playlist.remove(song)

    def play(self):
        """
        Plays the current song in the playlist.
        :return: The current song in the playlist, or False if there is no current song.
        """
        if self.current_song_index != -1:
            return self.playlist[self.current_song_index]
        return False

    def stop(self):
        """
        Stops the current song in the playlist.
        :return: True if the current song was stopped, False if there was no current song.
        """
        if self.current_song_index != -1:
            self.current_song_index = -1
            return True
        return False

    def switch_song(self):
        """
        Switches to the next song in the playlist.
        :return: True if the next song was switched to, False if there was no next song.
        """
        if self.playlist:
            self.current_song_index = (self.current_song_index + 1) % len(self.playlist)
            return True
        return False

    def previous_song(self):
        """
        Switches to the previous song in the playlist.
        :return: True if the previous song was switched to, False if there was no previous song.
        """
        if self.playlist:
            self.current_song_index = (self.current_song_index - 1) % len(self.playlist)
            return True
        return False

    def set_volume(self, volume):
        """
        Sets the volume of the music player, if the volume is between 0 and 100 is valid.
        :param volume: The volume to set the music player to, int.
        :return: True if the volume was set, False if the volume was invalid.
        """
        if 0 <= volume <= 100:
            self.volume = volume
            return True
        return False

    def shuffle(self):
        """
        Shuffles the playlist.
        :return: True if the playlist was shuffled, False if the playlist was empty.
        """
        if self.playlist:
            random.shuffle(self.playlist)
            return True
        return False

# Test cases for each method
if __name__ == "__main__":
    musicPlayer = MusicPlayer()
    
    # Test add_song
    musicPlayer.add_song("song1")
    print(musicPlayer.playlist)  # ['song1']

    # Test remove_song
    musicPlayer.add_song("song2")
    musicPlayer.remove_song("song1")
    print(musicPlayer.playlist)  # ['song2']

    # Test play
    musicPlayer.current_song_index = 0
    print(musicPlayer.play())  # 'song2'

    # Test stop
    print(musicPlayer.stop())  # True
    print(musicPlayer.play())  # False

    # Test switch_song
    musicPlayer.add_song("song3")
    musicPlayer.switch_song()
    print(musicPlayer.current_song_index)  # 0 (first song)
    print(musicPlayer.play())  # 'song2'
    musicPlayer.switch_song()
    print(musicPlayer.current_song_index)  # 1 (second song)
    print(musicPlayer.play())  # 'song3'

    # Test previous_song
    musicPlayer.previous_song()
    print(musicPlayer.current_song_index)  # 0
    print(musicPlayer.play())  # 'song2'

    # Test set_volume
    musicPlayer.set_volume(75)
    print(musicPlayer.volume)  # 75
    print(musicPlayer.set_volume(150))  # False

    # Test shuffle
    musicPlayer.add_song("song4")
    print(musicPlayer.shuffle())  # True
    print(musicPlayer.playlist)  # Random order of songs in playlist