import os
def create_ml_project_structure(project_name):
    # Define the project structure
    # project_structure = {
    #     'data': ['raw', 'processed', 'interim', 'external'],
    #     'notebooks': ['exploratory', 'preprocessing', 'modeling'],
    #     'src': ['data', 'features', 'models', 'evaluation', 'visualization'],
    #     'models': [],
    #     'config': [],
    #     'notebooks': ['exploratory', 'preprocessing', 'modeling'],
    #     'requirements.txt': '',
    #     'README.md': '',
    #     'LICENSE': '',
    #     '.gitignore': '',
    #     '.editorconfig': ''
    # }

    project_structure = {
        'data': ['raw', 'processed', 'interim', 'external'],
        'notebooks': ['exploratory', 'preprocessing', 'modeling'],
        'src': ['data', 'features', 'models', 'evaluation', 'visualization'],
        'models': [],
        'config': [],
        'requirements.txt': '',
        'results':['train', 'test']
    }

    # Create the project directory
    if not os.path.exists(project_name):
        os.makedirs(project_name)
    os.chdir(project_name)

    # Create subdirectories and files
    for item, value in project_structure.items():
        if isinstance(value, list):
            for subitem in value:
                os.makedirs(os.path.join(item, subitem))
        else:
            with open(item, 'w') as f:
                f.write(value)

    print(f"Created the basic structure for the '{project_name}' machine learning project.")

# Replace 'your_project_name' with the desired name for your project
create_ml_project_structure("/Users/sharadc/Documents/uic/semester4/CS598/MultiModalEmotionCauseAnalysis")
