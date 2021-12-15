import subprocess

# PONG
#subprocess.run("python /home/atari/atari-dqn/irb1000.py & python /home/atari/atari-dqn/irb10000.py & python /home/atari/atari-dqn/irb50000.py & python /home/atari/atari-dqn/irb100000.py", shell=True)
#subprocess.run("python /home/atari/atari-dqn/pushfreq50.py & python /home/atari/atari-dqn/pushfreq100.py & python /home/atari/atari-dqn/pushfreq200.py & python /home/atari/atari-dqn/pushfreq500.py", shell=True)
#subprocess.run("python /home/atari/atari-dqn/updatefreq50.py & python /home/atari/atari-dqn/updatefreq100.py & python /home/atari/atari-dqn/updatefreq200.py & python /home/atari/atari-dqn/updatefreq500.py", shell=True)

# BREAKOUT
#subprocess.run("python /home/atari/atari-dqn/breakout-irb-1000.py & python /home/atari/atari-dqn/breakout-irb-10000.py & python /home/atari/atari-dqn/breakout-irb-50000.py & python /home/atari/atari-dqn/breakout-irb-100000.py", shell=True)
#subprocess.run("python /home/atari/atari-dqn/breakout-pfreq-50.py & python /home/atari/atari-dqn/breakout-pfreq-100.py & python /home/atari/atari-dqn/breakout-pfreq-200.py & python /home/atari/atari-dqn/breakout-pfreq-500.py", shell=True)
subprocess.run("python /home/atari/atari-dqn/breakout-ufreq-50.py & python /home/atari/atari-dqn/breakout-ufreq-100.py & python /home/atari/atari-dqn/breakout-ufreq-200.py & python /home/atari/atari-dqn/breakout-ufreq-500.py & python /home/atari/atari-dqn/special-breakout.py", shell=True)


#subprocess.run("python /home/atari/atari-dqn/assualt_duel_dqn.py & python /home/atari/atari-dqn/assualt_duel_dqn_random_irm.py & python /home/atari/atari-dqn/assualt_duel_dqn_prio_irm.py & python /home/atari/atari-dqn/assualt_duel_dqn_highest_irm.py", shell=True)