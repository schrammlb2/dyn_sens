import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--no_penalty', default=False, action='store_true', help='Bool type')
# parser.add_argument('--no_penalty', type=str, default='')
args = parser.parse_args()

task_list = ['walker','halfcheetah','hopper','ant', 'swimmer']#,'humanoid']

base_command = 'python td3_run.py --algorithm=sac --task='
for i in range(3):
	for task in task_list:
		command = base_command + task
		if args.no_penalty:
			command += ' --no_penalty'
		os.system(command)
