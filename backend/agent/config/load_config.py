
import os
import yaml
# Define file paths for YAML configurations
files = {
    'agents': 'agent/config/agents.yaml',
    'tasks': 'agent/config/tasks.yaml',
    'tools': 'agent/config/tools.yaml',
}

# Load configurations from YAML files
configs = {}
for config_type, file_path in files.items():
    with open(file_path, 'r') as file:
        configs[config_type] = yaml.safe_load(file)

# Assign loaded configurations to specific variables
agents_config = configs['agents']
tasks_config = configs['tasks']
tools_config = configs['tools']



     
