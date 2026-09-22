#making package of project
    
from setuptools import setup, find_packages
from typing import List
def get_requirements(file_path:str)-> List[str]:
    '''
    this function will return the list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]

        if '-e .' in requirements:
            requirements.remove('-e .')
    return requirements

setup( name = 'vehicle-insurance-pred',
       version = '0.1',
       author = 'Arpita',
       author_email = 'arpitatomar45@gmail.com',
       packages = find_packages(),#connect all init files
       install_requires = get_requirements('req.txt')

 )