import threading
import RPi.GPIO as GPIO
from gpio_config import pin_dic
import numpy as np
import time

class MusicPlayer(threading.Thread):
    """
    Music player for Raspberry Pi buzzer
    
    Well... why can't a buzzer be called as a music player...
    Plays musical notes from a file using a buzzer connected to a GPIO pin.
    Runs in a separate thread for background playback.
    """
    def __init__(self, pin):
        """
        Initialize the music player
        
        Args:
            pin: GPIO pin number (BOARD numbering) connected to buzzer
        """
        super(MusicPlayer, self).__init__() 
        
        # Set buzzer pin mode
        self.pin_buzzer = pin
        GPIO.setmode(GPIO.BOARD)
        GPIO.setwarnings(False)
        GPIO.setup(self.pin_buzzer, GPIO.OUT)
        
        # Time duration of one beat
        self.delay_beat = 0.2
        
        # Store frequencies and beat durations
        self.freqs = []
        self.beats = []
        
        # Flag to stop playback
        self.flag_stop = False
       
    def load_music_file(self, file_music):
        """
        Load music data from file
        
        Args:
            file_music: Path to music file
            
        Returns:
            bool: True if file loaded successfully, False otherwise
        """
        # Load music sheet data from file
        data = np.loadtxt(file_music, dtype='str')
        [n, d] = np.shape(data)
    
        # Must have 3 columns
        if not d == 3:
            return False
            
        # Predefined note frequencies
        CL = [0, 131, 147, 165, 175, 196, 211, 248]        
        CM = [0, 262, 294, 330, 349, 392, 440, 494]       
        CH = [0, 525, 589, 661, 700, 786, 882, 990] 
            
        # First column: pitch level, second column: note index, third column: duration
        levels = data[:, 0]
        beats = data[:, 2]
        beats = beats.astype('int32')
        notes = data[:, 1]
        notes = notes.astype('int32')

        # Generate the music sheet
        self.freqs = []
        for i in range(n):
            if levels[i] == 'H':
                self.freqs.append(CH[notes[i]])
            elif levels[i] == 'M':    
                self.freqs.append(CM[notes[i]])
            elif levels[i] == 'L':
                self.freqs.append(CL[notes[i]])

        self.beats = beats.tolist()
        
        if not len(self.freqs) == len(self.beats):
            return False
        else:
            return True
    
    def stop_playback(self):
        """
        Stop music playback
        """
        self.flag_stop = True
        
    def run(self):
        """
        Main thread method to play the music
        """
        self.flag_stop = False
        
        # Define PWM object
        buzzer = GPIO.PWM(self.pin_buzzer, 440)
        buzzer.start(50)
        
        while True:
            if self.flag_stop:
                break
                
            for freq, beat in zip(self.freqs, self.beats):
                if self.flag_stop:
                    break
                buzzer.ChangeFrequency(freq)
                time.sleep(self.delay_beat * beat)
                
        buzzer.stop()        
        GPIO.output(self.pin_buzzer, GPIO.LOW)
        
        self.flag_stop = False

# For backward compatibility
Runing_Song = MusicPlayer

if __name__ == "__main__":
    
    file = "music.txt"
    
    # Initialize music player
    music_player = MusicPlayer(pin_dic['G18'])
    
    print("Running music playback 1")
    if music_player.isAlive() == False:
        # No thread, create and start thread
        music_player = MusicPlayer(pin_dic['G18'])
        flag = music_player.load_music_file(file)
        music_player.setDaemon(True)
        music_player.start()
        
    else:
        # If playing, stop first
        music_player.stop_playback()
        time.sleep(0.1)
        music_player.join()
        
        # Reload
        music_player = MusicPlayer(pin_dic['G18'])
        flag = music_player.load_music_file(file)
        music_player.setDaemon(True)
        music_player.start()
        
    time.sleep(5)
    
    print("Running music playback 2")
    if music_player.isAlive() == False:
        # No thread, create and start thread
        music_player = MusicPlayer(pin_dic['G18'])
        flag = music_player.load_music_file(file)
        music_player.setDaemon(True)
        music_player.start()
        
    else:
        # If playing, stop first
        music_player.stop_playback()
        time.sleep(0.1)
        music_player.join()
        
        # Reload
        music_player = MusicPlayer(pin_dic['G18'])
        flag = music_player.load_music_file(file)
        music_player.setDaemon(True)
        music_player.start()
        
    time.sleep(5)
    
    print("Stop playback 1")
    if music_player.isAlive() == True:
        music_player.stop_playback()
        time.sleep(0.1)
        music_player.join()
        
    time.sleep(5)
    
    print("Stop playback 2")
    if music_player.isAlive() == True:
        music_player.stop_playback()
        time.sleep(0.1)
        music_player.join()  

    time.sleep(5)
    print("Running music playback 3")
    if music_player.isAlive() == False:
        # No thread, create and start thread
        music_player = MusicPlayer(pin_dic['G18'])
        flag = music_player.load_music_file(file)
        music_player.setDaemon(True)
        music_player.start()
        
    else:
        # If playing, stop first
        music_player.stop_playback()
        time.sleep(0.1)
        music_player.join()
        
        # Reload
        music_player = MusicPlayer(pin_dic['G18'])
        flag = music_player.load_music_file(file)
        music_player.setDaemon(True)
        music_player.start()
        
    time.sleep(5)
