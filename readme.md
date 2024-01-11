
![Build,Test,Lint workflow](https://github.com/Eldeeqq/masters-thesis/actions/workflows/python-app.yml/badge.svg)

# Traced

This project contains experiment notebooks and source code for traceroute anomaly
detection using Bayesian inference with code name `traced`. This project was done as Masters Thesis at 
Faculty of Information Technology, Czech Technical University In Prague, in cooperation with CERN and University of
 Michigan.

The current version is in module `traced_v2`, `traced` module was kept for backward compatibility of notebooks, but will be removed in future.

For minimal demo, see [this notebook](./notebooks/showcase.ipynb).

# Usage

This project uses [conda](https://anaconda.org/) as a package manager. Conda environment file is present here [environment.yaml](./environment.yaml)

To install it, run:
```bash
conda env create --name traced --file=environment.yaml 
```

# Notebooks 
The notebooks are currently in read-only mode, due the re-implementation of the traced module. The notebooks have to undergo a migration to new version of the module (and also some cleaning).
 
The notebooks marked as `dirty` require further re-visition, or are meant to be deleted.

# Demo Dashboard app

The demo dashboard uses serialised models to avoid
sharing and processing the data.

Unzip the data archive into app folder, the final structure should look like: `app/data/*.dill`.
Then run the following commands.

```bash
conda env create --name traced --file=environment.yaml 
cd app
streamlit run app.py
```

After that, web browser should open with a tab containing the demo dashboard app. After each reload, please wait for the app to finish loading (animation in right top corner).

## pip alternative
You should be able to alternatively use pip requirements file. Make sure you have python 3.10 installed, then run:

```bash
pip install -r requirements.txt
```
# Remarks
The module, notebooks, and dashboard app was tested on Mac OS with M1 chip and on Ubuntu 18 LTS.  If you encounter any problems, create issue, or contact me at `perinja2@gmail.com`.