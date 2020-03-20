# stanford_cs_224n_chi_journal_club
Shared repository for Chicago journal club to work through assignments from the 2019 Stanford CS224N: NLP with Deep Learning. More info here: http://web.stanford.edu/class/cs224n/index.html#schedule

## Environment Info: 

Planning to use the environment shared with the class. The `env.yml` file can be found in `environment/` directory. Instructions opied below: 

```

Welcome to CS224N!

We'll be using Python throughout the course. If you've got a good Python setup already, great! But make sure that it is at least Python version 3.5. If not, the easiest thing to do is to make sure you have at least 3GB free on your computer and then to head over to (https://www.anaconda.com/download/) and install the Python 3 version of Anaconda. It will work on any operating system.

After you have installed conda, close any open terminals you might have. Then open a new terminal and run the following command:

# 1. Create an environment with dependencies specified in env.yml:
    
    conda env create -f env.yml

# 2. Activate the new environment:
    
    conda activate cs224n
    
# 3. Inside the new environment, instatll IPython kernel so we can use this environment in jupyter notebook: 
    
    python -m ipykernel install --user --name cs224n


# 4. Homework 1 (only) is a Jupyter Notebook. With the above done you should be able to get underway by typing:

    jupyter notebook exploring_word_vectors.ipynb
    
# 5. To make sure we are using the right environment, go to the toolbar of exploring_word_vectors.ipynb, click on Kernel -> Change kernel, you should see and select cs224n in the drop-down menu.

# To deactivate an active environment, use
    
    conda deactivate
    
```

### Potential Environmental Error for Step 3: 

I was getting the following error: 

```
ImportError: cannot import name 'constants' from 'zmq.backend.cython' (C:\Users\David\AppData\Roaming\Python\Python37\site-packages\zmq\backend\cython\__init__.py)
```

The steps listed here helped: https://github.com/jupyter/notebook/issues/3435#issuecomment-601920531

```python
pip uninstall pyzmq
pip install pyzmq
```
