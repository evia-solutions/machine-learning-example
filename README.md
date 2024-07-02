## Introduction
This project illustrates some machine learning concepts and uses python as program language. It also uses the below framework
   1. scikit-learn
   2. pandas
   3. numpy
   4. Matplot

## Covered topics:
   1. Regression problems : linear regression algorithm.
   2. Classification problems : logistic regression, random-forest and decision tree algorithms.
   3. Clustering problems from unsupervised learning : k-means clustering algorithm.
   4. Semi-supervised learning : Self-learning algorithm. 
   5. Data clean-up
      1. Outlier identification
      2. Skew data
      3. Downsampling and Upweighting strategies

## It's possible to run the examples via :
   1. command-line. 
   2. Jupyter notebook. 
   3. IDE(in this case Intellij).

**IMPORTANT:  _This project requires python 3 installed in the machine_.**

## Installation: Environment and how run the examples.
1. Create a virtual environment for python. In the src directory, machine-learning/src type:
   ```bash
   python3 -m venv venv venv
   ```  

2. Activating the python virtual environment.
   ```bash
   source venv/bin/activate
   ```  

3. Installing the required packages and kernel into the virtual environment.
   1. upgrading pip.
    ```bash
    pip install --upgrade pip
    ```   
   2.From the src folder, install the dependencies from requirements.txt file.
   ```bash
   cd ./src
   pip install -r requirements.txt
   ```
   3. Checking the installed libs by listing them. Run the below command to list the installed libraries on the terminal.
   ```bash
   pip list
   ```
   4. kernel installation with ipython
   ```bash
   ipython kernel install --user --name=my-env-3.10
   ```
   5. ipykernel installation with python.
   ```bash
   python -m ipykernel install --user --name=my-env-3.10
   ```
    
4. If you want to run via IDE, Intellij, you must follow the steps below:
   1. Go to File > Project structure
   2. Select SDKs
   3. In Packages click on the plus(+) sign to add the library.
   4. Select the desired library on the desired version, e.g: pandas 1.5.3.
      Note: If you do know the correct version go to terminal, inside the IDE (here the virtual environment should be active) and type pip list to check all installed libraries and their versions.
   5. Close the project structure window.
   
5. How do I run the examples via command line ?
   1. Open the terminal
   2. Select the directory that contains the python file that you want to execute and run the below command : python3 <file-name>.py
   Below on example:
   ````bash
   python3 decision-tree.py  
   ````

6. How do I run with Jupiter Notebook ? 
   1. After complete the step 4 above type:
   ```bash
   jupyter notebook <notebook-file>.ipynb
   ```
   2. Another option is run the below command and navigate to the desired file(extension ipynb).
   ```bash
   jupyter notebook
   ```
   3. To stop the server just interrupt the service typing "ctrl" + c in the terminal where the service is running.
   
**Notes:** 
1. When you start Jupyter one browser tab will open with the jupyter notebook server running.
2. To run Jupyter notebook inside Intellij you should install a plugin, here the instructions : https://www.jetbrains.com.cn/en-us/help/idea/jupyter-notebook-support.html#tool-windows 

## References
1. https://scikit-learn.org/stable/
2. https://www.geeksforgeeks.org/ml-linear-regression/
3. https://www.jetbrains.com/help/idea/editing-jupyter-notebook-files.html
4. https://towardsdatascience.com/a-gentle-introduction-to-self-training-and-semi-supervised-learning-ceee73178b38