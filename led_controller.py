import RPi.GPIO as GPIO


class LEDController:
    """
    LED Control Module
    
    Controls an LED connected to Raspberry Pi GPIO pin.
    Provides simple on/off functionality.
    """
    def __init__(self, pin):
        """
        Initialize the LED controller
        
        Args:
            pin: GPIO pin number (BOARD numbering)
        """
        # Store pin number
        self.pin = pin
        
        # Set GPIO mode to BOARD (physical pin numbering)
        GPIO.setmode(GPIO.BOARD)
        
        # Configure pin as output
        GPIO.setup(self.pin, GPIO.OUT)
        
        # Initial state: off
        GPIO.output(self.pin, GPIO.LOW)
        
    def turn_on(self):
        """
        Turn on the LED
        Sets the pin to HIGH state
        """
        GPIO.output(self.pin, GPIO.HIGH)
        
    def turn_off(self):
        """
        Turn off the LED
        Sets the pin to LOW state
        """
        GPIO.output(self.pin, GPIO.LOW)
    
    # Maintain backward compatibility with original method names
    on = turn_on
    off = turn_off