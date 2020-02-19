import xmltodict
import random
import copy
import pdb
import collections

fields_to_randomize = ['@friction', '@size', '@stiffness', '@damping']

def get_altered_field(field, scale):
	try: 
		float_val = float(field)*(1+random.uniform(-scale, scale))
		return str(float_val)
	except:
		list_field = field.split(' ')
		float_field = [float(elem)*(1+random.uniform(-scale, scale)) for elem in list_field]
		str_field = [str(elem) for elem in float_field]

		return ' '.join(str_field)


def alter_dict(diction, scale):
	#Alters dictionary representation in-place
	diction2 = copy.deepcopy(diction)
	for key, val in diction.items():
		# diction2[key] = alter(val, scale)
		if isinstance(val, (list,type(diction))):
			diction2[key] = alter(val, scale)
		elif key in fields_to_randomize:
			diction2[key] = get_altered_field(val, scale)

	return diction2

def alter(item, scale):
	if isinstance(item, list):
		return [alter(elem, scale) for elem in item]
	elif isinstance(item, collections.OrderedDict):
		return alter_dict(item, scale)


def randomize_xml(base_xml_file, scale=.1 ,count = 1):
	with open(base_xml_file) as fd:
		diction = xmltodict.parse(fd.read(), process_namespaces=True)

	for i in range(count):
		new_xml = xmltodict.unparse(alter(diction, scale), pretty=True)
		with open(base_xml_file[:-4] + '_rand_mod_' + str(i) + '.xml', 'w+') as fd:
			fd.write(new_xml)

scale = .3
randomize_xml('./mod_envs/halfcheetah/halfcheetah.xml', count = 4, scale=scale)
randomize_xml('./mod_envs/walker/walker2d.xml', count = 4, scale=scale)