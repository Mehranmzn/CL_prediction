
'''
The Setup.py file is a file that is used to create a Python package and
install it in the Python environment. It is a file that is used to install
the package in the Python environment such as the Anaconda environment, 
metaflow environment, etc. 

'''

from setuptools import setup, find_packages
from typing import List

def get_requirements() -> List[str]:

    """
    This function is used to read the requirements.txt file and return the
    """

    requirement_lst:List[str]=[]
    try:
        with open('requirements.txt','r') as file:
            #Read lines from the file
            lines=file.readlines()
            ## Process each line
            for line in lines:
                requirement=line.strip()
                ## ignore empty lines and -e .
                if requirement and requirement!= '-e .':
                    requirement_lst.append(requirement)
    except FileNotFoundError:
        print("requirements.txt file not found")

    return requirement_lst

setup(
    name="Credit Loan Prediction",
    version="0.0.1",
    author="MehranMzn",
    author_email="Mehran@mzn.com",
    packages=find_packages(),
    install_requires=get_requirements()
)