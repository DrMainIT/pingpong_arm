"""Exploratory Raspberry Pi client. Motors are disabled unless explicitly enabled.

Set PINGPONG_INFERENCE_URL to the inference server. Validate model output scaling,
servo calibration, mechanical limits, and emergency-stop behavior before hardware use.
"""
import argparse
import os

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enable-motors", action="store_true", help="Send returned actions to configured GPIO servos")
    args = parser.parse_args()
    url = os.environ.get("PINGPONG_INFERENCE_URL", "http://127.0.0.1:8000/process-data")
    observation = [[0.0] * 23]  # Shape matches the current PingPongEnv observation
    response = requests.post(url, json={"observation": observation}, timeout=10)
    response.raise_for_status()
    result = response.json()
    actions = result["actions"][0]
    print("Policy actions:", actions)
    if not args.enable_motors:
        print("Dry run: no motors enabled. Pass --enable-motors only on a calibrated, secured setup.")
        return

    import RPi.GPIO as GPIO
    import time

    servo_pins = [5, 6, 13, 26]
    if len(actions) != len(servo_pins):
        raise ValueError(f"Expected {len(servo_pins)} actions, received {len(actions)}")
    GPIO.setmode(GPIO.BCM)
    pwms = []
    try:
        for pin in servo_pins:
            GPIO.setup(pin, GPIO.OUT)
            pwm = GPIO.PWM(pin, 50)
            pwm.start(0)
            pwms.append(pwm)
        for pwm, action in zip(pwms, actions):
            pwm.ChangeDutyCycle(2 + (float(action) / 18))
        time.sleep(0.5)
    finally:
        for pwm in pwms:
            pwm.stop()
        GPIO.cleanup()


if __name__ == "__main__":
    main()
