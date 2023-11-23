from setuptools import find_packages,setup
from typing import List

# Parameter to ignore
HYPEN_E_DOT = '-e .'

def get_requirements(file_path:str)->List[str]:
    '''
    this function will return the list of requirements
    '''

    requirements=[]

    with open(file_path) as file_obj: ## Temporary file object
        requirements = file_obj.readlines()
        requirements = [req.replace ("\n", "") for req in requirements]
        
        # Removing the -e . from the requirements
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements


setup(
    name='mlproject',
    version='0.0.1',
    author='Maxi',
    author_email='maxiarancibiac@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
    
)