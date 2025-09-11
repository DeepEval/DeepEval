import random

class MusicPlayer:
    def __init__(self):
        self.playlist = []
        self.current_song = None
        self.volume = 50

    def add_song(self, song):
        self.playlist.append(song)

    def remove_song(self, song):
        if song in self.playlist:
            self.playlist.remove(song)

    def play(self):
        if self.current_song:
            return self.current_song
        else:
            return False

    def stop(self):
        current_song = self.current_song
        self.current_song = None
        return current_song is not None

    def switch_song(self):
        if self.playlist:
            self.current_song = self.playlist.pop(0)
            if not self.playlist:
                self.current_song = None
            return True
        else:
            return False

    def previous_song(self):
        if self.playlist:
            self.current_song = self.playlist.pop()
        else:
            self.current_song = None
        return self.current_song is not None

    def set_volume(self, volume):
        if 0 <= volume <= 100:
            self.volume = volume
            return True
        return False

    def shuffle(self):
        if self.playlist:
            random.shuffle(self.playlist)
            return True
        else:
            return False

# Test cases
if __name__ == "__main__":
    music_player = MusicPlayer()
    print(music_player.add_song("song1"))  # True
    print(music_player.playlist)  # ['song1']

    print(music_player.remove_song("song1"))  # True
    print(music_player.playlist)  # []

    print(music_player.play())  # 'song1'

    print(music_player.stop())  # None
    print(music_player.playlist)  # []

    print(music_player.switch_song())  # 'song1'
    print(music_player.playlist)  # []

    print(music_player.previous_song())  # 'song1'

    print(music_player.set_volume(50))  # True
    print(music_player.volume)  # 50

    print(music_player.shuffle())  # True
    print(music_player.playlist)  # List changed