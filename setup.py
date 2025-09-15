from setuptools import find_packages, setup 
from typing import List

HYPEN_E_DOT = '-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    This function will return a list of required packages mentioned in the requirements.txt file 
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name =  'Diabetes_Readmission_Predcition', 
    version = '0.0.1', 
    author = 'Rubina', 
    author_email='rubina15parveen@gmail.com', 
    packages=find_packages(), 
    install_requires=get_requirements('requirements.txt')

)