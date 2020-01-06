from setuptools import setup, find_packages
import os

requires = [
    "pandas",
    "scikit-image",
    "numpy",
    "scipy",
    "scikit-learn",
    "seaborn",
    "matplotlib",
]
DESCRIPTION = "Auto zoning algorithm from IF image based on vessels."

os.system(
    "pip install \
    https://files.pythonhosted.org/packages/37/86/3d0b1829c8c24eb1a4214f098a02442209f80302766203db33c99a4681ec/czifile-2019.7.2-py2.py3-none-any.whl"
)

setup(
    name="autozone",
    version=0.1,
    description=DESCRIPTION,
    url="https://github.com/yunguan-wang/liver_zone_segmentation",
    author="Yunguan Wang",
    author_email="yunguan.wang@utsouthwestern.edu",
    license="MIT",
    packages=find_packages(),
    install_requires=requires,
    zip_safe=False,
)

