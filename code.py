# This runs on the Metro M0 Express

import time
import board
import digitalio
import pwmio

# Initialize ultrasonic sensor pins
trig = digitalio.DigitalInOut(board.D3)
trig.direction = digitalio.Direction.OUTPUT
echo = digitalio.DigitalInOut(board.D4)
echo.direction = digitalio.Direction.INPUT


# Initialize infrared sensor pin
infrared_pin = digitalio.DigitalInOut(board.D2)
infrared_pin.direction = digitalio.Direction.INPUT

servo = pwmio.PWMOut(board.D9, frequency=50)
servo.duty_cycle = 0

def get_distance():
    trig.value = False
    time.sleep(0.01)
    trig.value = True
    time.sleep(0.00001)
    trig.value = False

    pulse_start = time.monotonic()
    while not echo.value:
        pulse_start = time.monotonic()

    pulse_end = time.monotonic()
    while echo.value:
        pulse_end = time.monotonic()

    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17150
    return distance



while True:
    infrared_value = infrared_pin.value
    distance = get_distance()
    timestamp = time.time()

    # Print sensor data to serial monitor
    print(f"{timestamp},{distance}")

     # Control servo motor based on infrared sensor value
    if infrared_value:
        servo.duty_cycle = int(0xFFFF*0.9)
    else:
        servo.duty_cycle = 0       # Stop the servo

    time.sleep(0.01)  # Adjust the sleep time for more frequent readings
 # type: ignore


