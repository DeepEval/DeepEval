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
        self.current_song = None
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
        if self.current_song is not None:
            return self.current_song
        return False

    def stop(self):
        """
        Stops the current song in the playlist.
        :return: True if the current song was stopped, False if there was no current song.
        """
        if self.current_song is not None:
            self.current_song = None
            return True
        return False

    def switch_song(self):
        """
        Switches to the next song in the playlist.
        :return: True if the next song was switched to, False if there was no next song.
        """
        if self.playlist:
            if self.current_song is None:
                self.current_song = self.playlist[0]
            else:
                current_index = self.playlist.index(self.current_song)
                next_index = (current_index + 1) % len(self.playlist)
                self.current_song = self.playlist[next_index]
            return True
        return False

    def previous_song(self):
        """
        Switches to the previous song in the playlist.
        :return: True if the previous song was switched to, False if there was no previous song.
        """
        if self.playlist:
            if self.current_song is None:
                self.current_song = self.playlist[-1]
            else:
                current_index = self.playlist.index(self.current_song)
                prev_index = (current_index - 1) % len(self.playlist)
                self.current_song = self.playlist[prev_index]
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

# Test cases
if __name__ == "__main__":
    musicPlayer = MusicPlayer()

    # Test add_song
    musicPlayer.add_song("song1")
    musicPlayer.add_song("song2")
    print(musicPlayer.playlist)  # Output: ['song1', 'song2']

    # Test remove_song
    musicPlayer.remove_song("song1")
    print(musicPlayer.playlist)  # Output: ['song2']

    # Test play
    musicPlayer.current_song = "song2"
    print(musicPlayer.play())  # Output: 'song2'

    # Test stop
    print(musicPlayer.stop())  # Output: True
    print(musicPlayer.current_song)  # Output: None

    # Test switch_song
    musicPlayer.playlist = ["song1", "song2"]
    musicPlayer.current_song = "song1"
    print(musicPlayer.switch_song())  # Output: True
    print(musicPlayer.current_song)  # Output: 'song2'

    # Test previous_song
    musicPlayer.current_song = "song2"
    print(musicPlayer.previous_song())  # Output: True
    print(musicPlayer.current_song)  # Output: 'song1'

    # Test set_volume
    print(musicPlayer.set_volume(75))  # Output: True
    print(musicPlayer.volume)  # Output: 75

    # Test shuffle
    musicPlayer.playlist = ["song1", "song2", "song3"]
    print(musicPlayer.shuffle())  # Output: True
    print(musicPlayer.playlist)  # Output might be ['song2', 'song1', 'song3'] or any other shuffled order