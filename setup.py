try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'lsa',
    'author': 'Leonardo Lazzaro',
    'url': 'https://github.com/llazzaro/lsa_python',
    'download_url': '',
    'author_email': 'lazzaroleonardo@gmail.com',
    'version': '0.1',
    'install_requires': ['nose', 'nltk', 'numpy', 'scipy'],
    'packages': ['lsa'],
    'scripts': [],
    'name': 'lsa'
}

setup(**config)
