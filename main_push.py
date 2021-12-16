import subprocess
import sys

subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
'gym==0.19.0'])

subprocess.run("python breakout-pfreq-50.py & python breakout-pfreq-100.py & python breakout-pfreq-200.py & python breakout-pfreq-500.py", shell=True)

