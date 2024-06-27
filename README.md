## Goal
This project is meant to illustrate the use of python using machine leaening.


## Installation running in command line or via IDE - Intellij
1. Create a virtual environment - VM with the command from the source, src, directory.

```bash
  python3 -m venv venv venv
```  

2. Active the VM.
```bash
source venv/bin/activate
```  
3. Installing the required packages and installing the kernel
   1. upgrade pip
    ```bash
       pip install --upgrade pip
    ```   
   2.install from the requirements.txt
   ```bash
       pip install -r requirements.txt
   ```
   3. Activate the virtual environment
   ```bash
      source venv/bin/activate 
   ```
   4. How check the installed libs by listing them
   ```bash
      pip list
   ```
   5. kernel installation on ipython
      ```bash
        ipython kernel install --user --name=my-env-3.10
      ```
   6. ipykernel installation with python.
      ```bash
        python -m ipykernel install --user --name=my-env-3.10
      ```
    
4. If you want to run via IDE, Intellij, run the follow steps:
   1. Go to File > Project structure
   2. Select SDKs
   3. In Packages click on the plus(+) sign to add the library.
   4. Select the desired library on the desired version, e.g: pandas 1.5.3.
      Note: If you do know the correct version go to terminal, inside the IDE (here the virtual environment should be active) and type pip list to check all installed libraries and their versions.
   5. Close the project structure window.
   
5. How run an example in the command line:
   1. Open the terminal
   2. In the directory : /machine-learning-sample/src/main go to desired sub-directory. Below an example from the classification sub-directory.
   ```bash
      cd classification
   ```
   3. Run the below command : python3 <file-name>.py
   ````bash
      python3 decision-tree.py  
   ````

6. How to run a Jupiter Notebook. One browser tab will open with the jupyter notebook running.
   1. After complete the step 4 above type:
   ```bash
      jupyter notebook <notebook-file>.ipynb
   ```
   2. Another option is run the below command and navigate to the desired file(extension ipynb).
   ```bash
      jupyter notebook
   ```
   3. To stop the server just interrupt the service typing "ctrl" + c in the terminal where the service is running.


## Documentation
1. https://scikit-learn.org/stable/
2. https://www.jetbrains.com/help/idea/editing-jupyter-notebook-files.html
3. https://towardsdatascience.com/a-gentle-introduction-to-self-training-and-semi-supervised-learning-ceee73178b38

## TODO
1. Installation running with Jupiter notebook.
