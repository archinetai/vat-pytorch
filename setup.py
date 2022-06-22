from setuptools import setup, find_packages

setup(
  name = 'vat-pytorch',
  packages = find_packages(exclude=[]),
  version = '0.0.1',
  license='MIT',
  description = 'Virtual Adversarial Training - Pytorch',
  long_description_content_type = 'text/markdown',
  author = 'Archinet',
  author_email = 'archinetai@protonmail.com',
  url = 'https://github.com/archinetai/vat-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'fine-tuning',
    'pre-trained',
  ],
  install_requires=[
    'torch>=1.6',
    'data-science-types>=0.2'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
