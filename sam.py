import subprocess
import sys

subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
'gym==0.19.0'])

# PONG
#subprocess.run("python irb1000.py & python irb10000.py & python irb50000.py & python irb100000.py", shell=True)
#subprocess.run("python pushfreq50.py & python pushfreq100.py & python pushfreq200.py & python pushfreq500.py", shell=True)
#subprocess.run("python updatefreq50.py & python updatefreq100.py & python updatefreq200.py & python updatefreq500.py", shell=True)

# BREAKOUT
#subprocess.run("python breakout-irb-1000.py & python breakout-irb-10000.py & python breakout-irb-50000.py & python breakout-irb-100000.py", shell=True)
#subprocess.run("python breakout-pfreq-50.py & python breakout-pfreq-100.py & python breakout-pfreq-200.py & python breakout-pfreq-500.py", shell=True)
subprocess.run("python breakout-ufreq-50.py & python breakout-ufreq-100.py & python breakout-ufreq-200.py & python breakout-ufreq-500.py & python breakout-pfreq-200.py & python breakout-pfreq-500.py", shell=True)


#subprocess.run("python assualt_duel_dqn.py & python assualt_duel_dqn_random_irm.py & python assualt_duel_dqn_prio_irm.py & python assualt_duel_dqn_highest_irm.py", shell=True)