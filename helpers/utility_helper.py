import yaml
import os
import shutil

def read_file(file_name):
	with open(file_name, "r", encoding="utf-8") as file:
		return file.read()
	
def write_file(file_name, content):
	with open(file_name, "w", encoding="utf-8") as file:
		file.write(content)
	
def clean_empty_lines(full_content):
	lines = full_content.split('\n')
	cleaned_lines = []
	empty_line_count = 0

	for line in lines:
		if line.strip() == '':
			empty_line_count += 1
			if empty_line_count <= 2:
				cleaned_lines.append(line)
		else:
			empty_line_count = 0
			cleaned_lines.append(line)

	# Ricostruire il contenuto con le righe vuote eliminate
	cleaned_content = '\n'.join(cleaned_lines)
	return cleaned_content

def count_all_markdown_files(folder):
	i = 0
	for dirpath, dirnames, filenames in os.walk(folder):
		for filename in filenames:
			if filename.endswith('.md'):
				i += 1
	return i


def count_all_files_with_extensions(folder, extension):
	i = 0
	for dirpath, dirnames, filenames in os.walk(folder):
		for filename in filenames:
			if filename.endswith(extension):
				i += 1
	return i

def count_all_files(folder):
	i = 0
	for dirpath, dirnames, filenames in os.walk(folder):
		for filename in filenames:
				i += 1
	return i

def count_non_markdown_files(directory):
	i = 0
	# Walk through all directories and files in the provided directory
	for dirpath, dirnames, files in os.walk(directory):
		for file in files:
			if not file.endswith('.md'):
				i += 1
	return i

def extract_yaml_front_matter(content: str):
	
	# Cerca l'inizio e la fine del front matter YAML
	start = content.find('---') + 3
	end = content.find('---', start)
	
	# Verifica se il front matter è stato trovato
	if start != -1 and end != -1:
		yaml_content = content[start:end]
		try:
			# Prova ad analizzare il contenuto YAML
			data = yaml.safe_load(yaml_content)
			return data
		except yaml.YAMLError as error:
			print(f"Errore durante l'analisi YAML: {error}")
			return {}
	else:
		# Front matter non trovato
		return {}
	

def move_files_to_folder(source_folder, destination_folder, extension:str = '.md'):
	for root, dirs, files in os.walk(source_folder):
		for file in files:
			if file.endswith(extension):
				source_path = os.path.join(root, file)
				destination_path = source_path.replace(source_folder, destination_folder)
				os.makedirs(os.path.dirname(destination_path), exist_ok=True)
				shutil.move(source_path, destination_path)