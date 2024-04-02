# Linear Parameter Varying Dynamical Systems (LPV-DS)

Boiler plate code of LPV-DS framework, compatible with any customizing clustering and optimization methods. Providing utilies functions from loading_tools, process_tools, plot_tools, and evaluation_tool to test on any variant of LPV-DS framework.


<!-- ![Picture1](https://github.com/SunannnSun/damm_lpvds/assets/97807687/5a72467b-c771-4e8a-a0e0-7828efa59952) -->




## Usage Example
Using DAMM-based LPV-DS as an example, 
```
cd src
```
```
git clone https://github.com/SunannnSun/damm.git
```
```
git clone https://github.com/SunannnSun/ds_opt.git
```

Compile DAMM module
- Please refer to the **compilation** section in [damm repository](https://github.com/SunannnSun/damm).

Create a python virtual environment and install the dependencies
- Make sure to replace `/path/to/python3.8` with the correct path to the Python 3.8 executable on your system. 

```
virtualenv -p /path/to/python3.8 env3.8
source env3.8/bin/activate
pip install -r requirements.txt
```

Run
```
python main.py
```
