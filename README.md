THIS PROJECT IS UNDER DEVELOPMENT

create seprate env

write basic requirements.txt

download red wine quality data


git init

dvc init

dvc add data_given/winequality.csv

git add .

git commit -m "first commit"


git status

Push to your repo

one liner:- git add . && git commit -m "message" && git push origin branch

create an artifacts folder

mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 127.0.0.1 -p 2872