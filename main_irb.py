import subprocess
import sys

subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
'gym==0.19.0'])

# BREAKOUT
subprocess.run("python /home/atari/atari-dqn/breakout-irb-1000.py & python /home/atari/atari-dqn/breakout-irb-10000.py & python /home/atari/atari-dqn/breakout-irb-50000.py & python /home/atari/atari-dqn/breakout-irb-100000.py", shell=True)
