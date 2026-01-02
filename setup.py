import os
from setuptools import setup, find_packages

def find_data_files(package_dir, data_dirs):
    data_files = []
    for d in data_dirs:
        # Walk through the data directory
        for root, dirs, files in os.walk(os.path.join(package_dir, d)):
            for f in files:
                # Get the path relative to the package directory
                data_files.append(os.path.join(root, f).replace(package_dir + os.sep, ''))
    return data_files

package_name = 'synriard'
if not os.path.exists(package_name):
    os.makedirs(package_name)
if not os.path.exists(os.path.join(package_name, '__init__.py')):
    with open(os.path.join(package_name, '__init__.py'), 'w') as f:
        f.write("# This file makes this a Python package.\n")

# Find all the data files
data_files = find_data_files(package_name, ['urdf', 'meshes', 'mjcf'])

setup(
    name=package_name,
    version="1.0.0",
    author="Synria Robotics",
    author_email="support@synriarobotics.ai",
    description="URDF and MJCF robot description files for Synria robotic platforms.",
    # MODIFIED: Added encoding='utf-8' to ensure cross-platform compatibility,
    # especially for Windows, preventing UnicodeDecodeError.
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Synria-Robotics/Synria-robot-descriptions",
    # Find the dummy package we created
    packages=find_packages(),
    # Tell setuptools to include the non-Python files
    package_data={
        package_name: ['urdf/**/*', 'meshes/**/*', 'mjcf/**/*']
    },
    include_package_data=True,
    keywords="robotics, urdf, mjcf, robot-description",
    python_requires=">=3.7",
)
