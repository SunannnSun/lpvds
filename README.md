# Linear Parameter Varying Dynamical Systems (LPV-DS)

Boiler plate code of LPV-DS framework, compatible with any customizing clustering and optimization methods. Providing utilies functions from loading_tools, process_tools, plot_tools, and evaluation_tool to test on any variant of LPV-DS framework.


<!-- ![Picture1](https://github.com/SunannnSun/damm_lpvds/assets/97807687/5a72467b-c771-4e8a-a0e0-7828efa59952) -->




## Usage Example

Fetch the required submodules
```
git submodule update --init --recursive
```

Compile [DAMM](https://github.com/SunannnSun/damm) submodule
```
cd src/damm/build
cmake ../src
make
```

Return to root directory and install all dependencies in a virtual environment
- Make sure to replace `/path/to/python3.8` with the correct path to the Python 3.8 executable on your system. 

```
cd -
virtualenv -p /path/to/python3.8 venv
source venv/bin/activate
pip install -r requirements.txt
```

Run 
```
python main.py
```
