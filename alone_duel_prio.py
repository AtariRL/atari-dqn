import subprocess
import sys

subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
'gym==0.19.0'])

subprocess.run("python /home/atari/atari-dqn/duel-prio-dqn.py", shell=True)