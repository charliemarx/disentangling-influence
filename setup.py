import os

path_to_here = os.path.abspath('.')

with open('DisentanglingInfluence/path.py', 'w') as f:
	f.write('def get_path():\n\treturn "{}/DisentanglingInfluence/"'.format(path_to_here))
