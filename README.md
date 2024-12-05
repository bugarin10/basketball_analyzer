



## Installation
For mediapipe it was neccessary to update the package lists and install a specific library on a Linux system that uses the APT package manager (such as Ubuntu or Debian).

sudo apt update: This command updates the package lists for packages that need upgrading, as well as new packages that have just come to the repositories.

sudo apt install -y libgl1: This command installs the libgl1 package, which is a library for OpenGL. The -y flag automatically answers "yes" to any prompts, allowing the installation to proceed without user intervention.

´´´bash
pip install -r requirements.txt
sudo apt update
sudo apt install -y libgl1
´´´

