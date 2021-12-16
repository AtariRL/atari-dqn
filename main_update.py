import subprocess
import sys

subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
'gym==0.19.0'])

subprocess.run("python breakout-ufreq-50.py & python breakout-ufreq-100.py & python breakout-ufreq-200.py & python breakout-ufreq-500.py", shell=True)
