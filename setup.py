from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()
with open('LICENCE') as f:
    license = f.read()

setup(
        name='project-echo-cnn',
        version='0.1.0',
        description='CNN for project echo',
        long_description=readme,
        author='Benjamin Whalley',
        author_email='benjaminjwhalley@outlook.com',
        url='',
        license=license,
        packages=find_packages(exclude=('docs', 'tests'))
)

