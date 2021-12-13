import subprocess

#subprocess.run("python /home/atari/atari-dqn/irb1000.py & python /home/atari/atari-dqn/irb10000.py & python /home/atari/atari-dqn/irb50000.py & python /home/atari/atari-dqn/irb100000.py", shell=True)
#subprocess.run("python /home/atari/atari-dqn/pushfreq50.py & python /home/atari/atari-dqn/pushfreq100.py & python /home/atari/atari-dqn/pushfreq200.py & python /home/atari/atari-dqn/pushfreq500.py", shell=True)
#subprocess.run("python /home/atari/atari-dqn/updatefreq50.py & python /home/atari/atari-dqn/updatefreq100.py & python /home/atari/atari-dqn/updatefreq200.py & python /home/atari/atari-dqn/updatefreq500.py", shell=True)



subprocess.run("python /home/atari/atari-dqn/assualt_duel_dqn.py & python /home/atari/atari-dqn/assualt_duel_dqn_random_irm.py & python /home/atari/atari-dqn/assualt_duel_dqn_prio_irm.py & python /home/atari/atari-dqn/assualt_duel_dqn_highest_irm.py", shell=True)