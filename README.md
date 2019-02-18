# my_reegis

Install reegis on a Linux system (Debian stable):

```bash
sudo apt-get install python3-dev proj-bin libproj-dev libgeos-dev python3-tk libspatialindex-dev virtualenv

virtualenv -p /usr/bin/python3 your_env_name
source your_env_name/bin/activate

pip install cython descartes oemof windpowerlib reegis deflex berlin_hp
```