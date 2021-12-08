import subprocess

subprocess.run("python /home/atari/atari-dqn/irb1000.py & python /home/atari/atari-dqn/irb10000.py & python /home/atari/atari-dqn/irb50000.py & python /home/atari/atari-dqn/irb100000.py", shell=True)