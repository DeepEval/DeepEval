import random

class MusicPlayer:
    def __init__(self):
        self.playlist = []
        self.current_song = None
        self.volume = 50

    def add_song(self, song):
        self.playlist.append(song)

    def remove_song(self, song):
        self.playlist.remove(song)

    def play(self):
        if self.current_song is None:
            return False
        return self.current_song

    def stop(self):
        self.current_song = None
        return True

    def switch_song(self):
        if len(self.playlist) == 0:
            return False
        self.current_song = random.choice(self.playlist)
        return True

    def previous_song(self):
        if len(self.playlist) == 0 or self.current_song is None:
            return False
        self.current_song = self.playlist[self.playlist.index(self.current_song) - 1]
        return True

    def set_volume(self, volume):
        if volume < 0 or volume > 100:
            return False
        self.volume = volume
        return True

    def shuffle(self):
        if len(self.playlist) == 0:
            return False
        random.shuffle(self.playlist)
        return True

if __name__ == "__main__":
    music_player = MusicPlayer()

    # Test add_song method
    music_player.add_song("song1")
    print(music_player.playlist)  # Output: ['song1']

    # Test remove_song method
    music_player.remove_song("song1")
    print(music_player.playlist)  # Output: []

    # Test play method
    music_player.add_song("song2")
    music_player.play()
    print(music_player.current_song)  # Output: 'song2'

    # Test stop method
    music_player.stop()
    print(music_player.current_song)  # Output: None

    # Test switch_song method
    music_player.add_song("song3")
    music_player.switch_song()
    print(music_player.current_song)  # Output: 'song3'

    # Test previous_song method
    music_player.add_song("song4")
    music_player.previous_song()
    print(music_player.current_song)  # Output: 'song3'

    # Test set_volume method
    music_player.set_volume(50)
    print(music_player.volume)  # Output: 50

    # Test shuffle method
    music_player.add_song("song5")
    music_player.add_song("song6")
    music_player.shuffle()
    print(music_player.playlist)  # Output: ['song6', 'song5', 'song3', 'song4', 'song2']