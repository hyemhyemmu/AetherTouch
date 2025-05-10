import RPi.GPIO as GPIO


class LEDController:
    """
    LED Control Module
    
    Controls an LED connected to Raspberry Pi GPIO pin.
    Provides both on/off functionality and brightness control using PWM.
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
        
        # Disable GPIO warnings
        GPIO.setwarnings(False)
        
        # Configure pin as output
        GPIO.setup(self.pin, GPIO.OUT)
        
        # Initialize PWM with 100Hz frequency (good for LED control)
        self.pwm = GPIO.PWM(self.pin, 100)
        
        # Current brightness level (0-100)
        self.brightness = 0
        
        # Start PWM with duty cycle 0 (LED off)
        self.pwm.start(0)
        
    def turn_on(self, brightness=100):
        """
        Turn on the LED with specified brightness
        
        Args:
            brightness: Brightness level from 0 to 100 (default: 100)
        """
        # Ensure brightness is within valid range
        brightness = max(0, min(100, brightness))
        self.brightness = brightness
        
        # Update PWM duty cycle
        self.pwm.ChangeDutyCycle(brightness)
        
    def turn_off(self):
        """
        Turn off the LED
        Sets the PWM duty cycle to 0
        """
        self.brightness = 0
        self.pwm.ChangeDutyCycle(0)
    
    def set_brightness(self, level):
        """
        Set LED brightness level
        
        Args:
            level: Brightness level from 0 to 100
        """
        # Ensure brightness is within valid range
        level = max(0, min(100, level))
        self.brightness = level
        
        # Update PWM duty cycle
        self.pwm.ChangeDutyCycle(level)
        
    def get_brightness(self):
        """
        Get current brightness level
        
        Returns:
            int: Current brightness level (0-100)
        """
        return self.brightness
    
    def cleanup(self):
        """
        Clean up GPIO resources for this LED.
        Stops PWM and releases the GPIO pin.
        """
        try:
            self.pwm.stop()  # Stop PWM
        except Exception as e:
            print(f"Error stopping PWM for pin {self.pin}: {e}")
        finally:
            try:
                # Only cleanup the specific pin if it was setup by this instance
                # This check might be more complex depending on how GPIO.cleanup() works
                # with multiple components. For now, we assume direct cleanup is fine.
                GPIO.cleanup(self.pin) 
                print(f"GPIO pin {self.pin} cleaned up.")
            except Exception as e:
                print(f"Error cleaning up GPIO pin {self.pin}: {e}")
    
    # Maintain backward compatibility with original method names
    on = turn_on
    off = turn_off