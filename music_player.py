import threading
import RPi.GPIO as GPIO
from gpio_config import gpio_pins
import numpy as np
import time

class MusicPlayer(threading.Thread):
    """
    Music player for Raspberry Pi buzzer
    
    Well... why can't a buzzer be called as a music player...
    Plays musical notes from a file using a buzzer connected to a GPIO pin.
    Runs in a separate thread for background playback.
    Volume can be controlled by adjusting PWM duty cycle.
    """
    # follow active PWM 
    active_pwm = {}
    
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
        
        # Volume control (PWM duty cycle, 0-100)
        self.volume = 50
        
        # Make sure clear all the PWM in same GPIO
        self._cleanup_pwm()
       
    def _cleanup_pwm(self):
        """clear PWM on current GPIO"""
        if self.pin_buzzer in MusicPlayer.active_pwm:
            try:
                MusicPlayer.active_pwm[self.pin_buzzer].stop()
                del MusicPlayer.active_pwm[self.pin_buzzer]
            except:
                pass
    
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
    
    def set_volume(self, level):
        """
        Set volume level by adjusting PWM duty cycle
        
        Args:
            level: Volume level from 0 to 100
        """
        # Ensure volume is within valid range
        self.volume = max(0, min(100, level))
        return self.volume
    
    def get_volume(self):
        """
        Get current volume level
        
        Returns:
            int: Current volume level (0-100)
        """
        return self.volume
    
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
        
        # make sure to clear old PWM before making a new one
        self._cleanup_pwm()
        
        # Define PWM object
        buzzer = GPIO.PWM(self.pin_buzzer, 440)
        MusicPlayer.active_pwm[self.pin_buzzer] = buzzer
        buzzer.start(self.volume)
        
        try:
            while True:
                if self.flag_stop:
                    break
                    
                for freq, beat in zip(self.freqs, self.beats):
                    if self.flag_stop:
                        break
                    buzzer.ChangeFrequency(freq)
                    # Apply current volume setting
                    buzzer.ChangeDutyCycle(self.volume)
                    time.sleep(self.delay_beat * beat)
        finally:

            buzzer.stop()
            GPIO.output(self.pin_buzzer, GPIO.LOW)
            if self.pin_buzzer in MusicPlayer.active_pwm:
                del MusicPlayer.active_pwm[self.pin_buzzer]
            
        self.flag_stop = False

# For backward compatibility
Runing_Song = MusicPlayer

if __name__ == "__main__":
    
    file = "music.txt"
    
    # Initialize music player
    music_player = MusicPlayer(gpio_pins['BUZZER_PIN'])
    
    print("Running music playback 1")
    if music_player.isAlive() == False:
        # No thread, create and start thread
        music_player = MusicPlayer(gpio_pins['BUZZER_PIN'])
        flag = music_player.load_music_file(file)
        music_player.setDaemon(True)
        music_player.start()
        
    else:
        # If playing, stop first
        music_player.stop_playback()
        time.sleep(0.1)
        music_player.join()
        
        # Reload
        music_player = MusicPlayer(gpio_pins['BUZZER_PIN'])
        flag = music_player.load_music_file(file)
        music_player.setDaemon(True)
        music_player.start()
        
    time.sleep(5)
    
    print("Running music playback 2")
    if music_player.isAlive() == False:
        # No thread, create and start thread
        music_player = MusicPlayer(gpio_pins['BUZZER_PIN'])
        flag = music_player.load_music_file(file)
        music_player.setDaemon(True)
        music_player.start()
        
    else:
        # If playing, stop first
        music_player.stop_playback()
        time.sleep(0.1)
        music_player.join()
        
        # Reload
        music_player = MusicPlayer(gpio_pins['BUZZER_PIN'])
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
        music_player = MusicPlayer(gpio_pins['BUZZER_PIN'])
        flag = music_player.load_music_file(file)
        music_player.setDaemon(True)
        music_player.start()
        
    else:
        # If playing, stop first
        music_player.stop_playback()
        time.sleep(0.1)
        music_player.join()
        
        # Reload
        music_player = MusicPlayer(gpio_pins['BUZZER_PIN'])
        flag = music_player.load_music_file(file)
        music_player.setDaemon(True)
        music_player.start()
        
    time.sleep(5)
