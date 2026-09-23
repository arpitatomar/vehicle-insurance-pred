mlops end to end vehicle insurance
1)
git ,setup .py , req.txt, pip installreq.txt
. Write the code on setup.py  to import local packages
   >> Find more about "setup.py and pyproject.toml" at crashcourse.txt
3. Create a virtual env, activate it and install the requirements from requirements.txt
   conda create -n vehicle python=3.11 -y
   # or use python=3.12 if preferred
   conda activate vehicle
   add required modules to requirements.txt
   Do "pip install -r requirements.txt"
4. Do a "pip list" on terminal to make sure you have local packages installed.

2)
create src, its folders
-------------------------------------- logging, exception and notebooks --------------------------------------
14. Write the logger file and 
15. Write the exception file and test it python src/exception/__init__.py
testing

3)
dataingestion 
vreate train test spilt

3.30)) data versioning:even one new ecorded u know, to retrain model: data tracking

pip install dvc
dvc init
dvc add artifact/raw.csv



4) data tranformation 
 do preprocessing ->save pkl object ->from util: saveobj func
 call datatransformation class (its all fucn)in data ingestion  after saving its data

5) model traing (best model choosen)
import save object from util???
create model pickle future path
crate model fuction in util created ->saves report
call in data ingestion

6)combined hyperparameter tuning
7) pred pipeline(predict on user query)+flask dev
home html calls oredict_datapint fucn

app.py -> looks for index.html in tmeplate


