import subprocess
import sys

subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
'gym==0.19.0'])

subprocess.run("python /home/atari/atari-dqn/prio-dqn.py & python /home/atari/atari-dqn/duel-prio-irm.py", shell=True)
#subprocess.run("python /home/atari/atari-dqn/duel-random-irm.py & python /home/atari/atari-dqn/duel-highest-error-irm.py & python /home/atari/atari-dqn/duel-prio-irm.py", shell=True)
