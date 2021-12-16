import subprocess
import sys

subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
'gym==0.19.0'])

# BREAKOUT
subprocess.run("python /home/atari/atari-dqn/breakout-pfreq-50.py & python /home/atari/atari-dqn/breakout-pfreq-100.py & python /home/atari/atari-dqn/breakout-pfreq-200.py & python /home/atari/atari-dqn/breakout-pfreq-500.py", shell=True)
