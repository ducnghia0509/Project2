from setuptools import setup, find_packages

setup(
    name='crypto_analysis_project', 
    version='0.1.0',
    author='DucNghia224883', 
    author_email='nghiabui5981@gmail.com', 
    description='Project2',
    packages=find_packages(exclude=['tests*', 'knowledge_base*', 'prepared_data_multi*', 'trained_models*', 'logs*']),

    python_requires='>=3.9',
)