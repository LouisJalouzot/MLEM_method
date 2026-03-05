#!/usr/bin/env python3
import sys
import os
import subprocess
import yaml

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <yaml-file>", file=sys.stderr)
    sys.exit(1)

path = sys.argv[1]
with open(path, 'r', encoding='utf-8') as f:
    data = yaml.safe_load(f)

def extract(obj, key_suffix):
    items = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and k.endswith(key_suffix):
                if isinstance(v, list):
                    items.extend([x for x in v if isinstance(x, str)])
                elif isinstance(v, str):
                    items.append(v)
            else:
                items.extend(extract(v, key_suffix))
    elif isinstance(obj, list):
        for item in obj:
            items.extend(extract(item, key_suffix))
    return items

models = list(dict.fromkeys(extract(data, 'model_name')))
revisions = list(dict.fromkeys(extract(data, 'revision')))

if not models:
    print('No models found.')
    sys.exit(0)

env = os.environ.copy()
env['HF_HUB_OFFLINE'] = '0'

for model in models:
    if revisions:
        for rev in revisions:
            print(f"Downloading: {model} (revision: {rev})")
            subprocess.run(['hf', 'download', model, '--revision', rev], env=env)
    else:
        print(f"Downloading: {model}")
        subprocess.run(['hf', 'download', model], env=env)
