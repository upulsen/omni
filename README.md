This repository contains the code, data and artefacts produced as part of completing o8t case study. The structure of this repository is as below.
1. code segments and Jupyter notebooks are at the root directory.
2. Artefacts produced such as Markdown of Jupyter notebook and presentations are stored in artefacts directory. 

In addition to purely using Github as VC tool, I also wanted to demonstrate a CI/CD pipeline for machine learning models. To that extent, this repository uses DVC (Data Version Control) and CML (Continuous Machine Learning) libraries in conjunction with Github actions to automate the CI/CD pipeline as follows.
1. The main branch of this repository will always carry the best model available.
2. When we want to improve upon the current best model or experiment with other models, we can branch the main and start working on a better model (ex: alternate_model branch)
3. Once we are satisfied with the new model, we can commit it to the new branch and create a pull request to merge it to the main branch if the model is better.
4. In the backend, every time a commit is made to the repository, we will setup a docker container with the required packages, execute the model, report its performance and artefacts as a markdown report and also compare the performance of the new model to the current best model and include that in the markdown report. 
5. This comparison can then be used to either approve the pull request if the new model is better or reject and rework on another model. 
6. Ultimately, this pipeline provides a data scientist with the opportunity to iterate and build new models that will automatically get tested, evaluated and compared before getting deployed. 
