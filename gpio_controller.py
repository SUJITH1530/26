import time

try:
    import RPi.GPIO as GPIO
    ON_RPI = True
except Exception:
    # Not running on Raspberry Pi; provide a mock
    ON_RPI = False


class GPIOController:
    """Simple GPIO controller for a 3-color traffic light.

    Pins configuration is: {'red': pin, 'amber': pin, 'green': pin}
    """

    def __init__(self, pins=None):
        if pins is None:
            pins = {'red': 17, 'amber': 27, 'green': 22}
        self.pins = pins
        if ON_RPI:
            GPIO.setmode(GPIO.BCM)
            for p in pins.values():
                GPIO.setup(p, GPIO.OUT)
        else:
            print('GPIOController: running in mock mode (not RPi)')

    def set_light(self, color):
        """Set a single color on, others off."""
        for name, pin in self.pins.items():
            state = (name == color)
            if ON_RPI:
                GPIO.output(pin, GPIO.HIGH if state else GPIO.LOW)
            else:
                print(f"GPIO mock: set {name} -> {'ON' if state else 'OFF'}")

    def cycle_green(self, duration):
        self.set_light('green')
        if ON_RPI:
            time.sleep(duration)
        else:
            print(f"GPIO mock: green for {duration}s")

    def emergency_mode(self, duration=15):
        """Flash green light for emergency vehicles."""
        self.set_light('green')
        if ON_RPI:
            time.sleep(duration)
        else:
            print(f"GPIO mock: EMERGENCY MODE - green for {duration}s")

    def cleanup(self):
        if ON_RPI:
            GPIO.cleanup()
        else:
            print('GPIO mock: cleanup')
