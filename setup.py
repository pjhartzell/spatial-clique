from setuptools import setup

setup(name='spatialclique',
      version='0.1',
      description='Maximum clique for corresponding 2D or 3D point sets',
      url='',
      author='Preston Hartzell',
      author_email='preston.hartzell@gmail.com',
      license='',
      packages=['spatialclique'],
      install_requires=[
            'numpy',
            'networkx'
      ],
      zip_safe=False)