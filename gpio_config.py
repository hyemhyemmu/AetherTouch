"""
GPIO Pin Mapping Module

This helper module provides a mapping dictionary from GPIO pin names to
their corresponding physical pin numbers on the Raspberry Pi board.
Used for easy pin reference across the project.
"""

# GPIO pins mapped to physical board pin numbers (BOARD numbering scheme)
gpio_pins = {
    # LED pins
    'LED_PIN': 11,  # Physical pin 11 (often corresponds to GPIO17)

    # Buzzer pins
    'BUZZER_PIN': 12, # Physical pin 12 (often corresponds to GPIO18)
    
    # I2C pins
    'SDA': 3,   # I2C Data
    'SCL': 23,  # I2C Clock
    
    # SPI pins
    'MOSI': 19, # Master Out Slave In
    'MISO': 21, # Master In Slave Out
    'CE0': 24,  # Chip Enable 0
    'CE1': 26,  # Chip Enable 1
    
    # UART pins
    'TXD': 8,   # Transmit
    'RXD': 10,  # Receive
    
    # ID EEPROM pins
    'IDSD': 27, # ID EEPROM Data
    'IDSC': 28, # ID EEPROM Clock
    
    # General purpose GPIO pins
    'G4': 7,
    'G5': 29,
    'G6': 31,
    'G12': 32, # Physical pin 32
    'G13': 33,
    'G16': 36,
    'G19': 35,
    'G20': 38,
    'G21': 40,
    'G22': 15,
    'G23': 16,
    'G24': 18,
    'G26': 37,
    'G27': 13
}