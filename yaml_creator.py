import yaml

data = {
    'train': 'custom_data/train/images',
    'val': 'custom_data/valid/images',
    'test': 'custom_data/test/images',
    'names': ['paint', 'dent', 'crack'],
    'nc': 3
}

with open('custom.yaml', 'w') as f:
    yaml.dump(data, f)

with open('custom.yaml', 'r') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
    print(data)