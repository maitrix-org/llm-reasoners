import signal
from utils.timeout import timeout_handler
import time

# Set the signal handler
signal.signal(signal.SIGALRM, timeout_handler)

try:
    signal.alarm(10)
    time.sleep(1)
    signal.alarm(0)  # Disable the alarm if successful
except TimeoutError:
    print("The code took too long and was terminated.")