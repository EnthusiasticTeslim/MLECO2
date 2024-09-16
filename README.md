![python version](https://img.shields.io/badge/python-v.3.9-blue)
![license](https://img.shields.io/badge/license-MIT-orange)
[![author](https://img.shields.io/badge/teslim-homepage)](https://teslim404.com)
# Machine Learning in CO2 reduction
Source code and trained models for the manuscript "*Machine Learning enables the productions of valuable chemicals from CO2 electrocatalysis*". 

<!-- TABLE OF CONTENTS -->
<h2 id="table-of-contents"> Table of Contents</h2>

<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#ResearchQuestions"> ➤ Research Questions</a></li>
    <li><a href="#ProjectDescription"> ➤ Project Description</a></li>
    <li><a href="#Team"> ➤ Team</a></li>
    <li><a href="#To-Do"> ➤ To Do</a></li>
    <li><a href="#License"> ➤ License</a></li>
  </ol>
</details>

<!-- Research Questions -->
<h2 id="ResearchQuestions">Research Questions</h2>

- Determine the relationship between available parameters and target (faradaic efficiency, FE)?.
- Develop accurate predictive model for the FE response.
- Explain the impact of the features on the FE response.
- Design novel conditions with the proposed ML-based models.


<!-- Project Description -->
<h2 id="ProjectDescription">Project Description</h2>

In the project, the main helper functions housing the training, plotting, and model architecture are located in the [`src`](src) directory. The ANN model and its weights are available in the [`neuralnetwork`](neuralnetwork), while other ML models are in [`otherML`](otherML). The SHAP analysis and optimization steps are carried out in the best model folder, i.e., [`neuralnetwork`](neuralnetwork). The [`web app`](deployment/app.py), built with [streamlit](https://streamlit.io/), is available in the deployment folder, and the Docker image is available on Docker at [DockerHub](https://hub.docker.com/search?q=teslim404).

<!-- Team-->
<h2 id="Team">Team</h2>

| Name | Email | Role
| :-- | :-- | :-- | 
| Dauda Monsuru | mdauda1@lsu.edu| Experiment |
| Teslim Olayiwola | tolayi1@lsu.edu | Simulation |
| John Flake | johnflake@lsu.edu| PI |
| Jose Romagnoli | jose@lsu.edu | PI |

<!-- To Do-->
<h2 id="To-Do">To Do</h2>

- [x] Clean and perform preliminary analysis on the data. 
- [x] Train ML model using multi-output ANN model.
- [x] Use best model to perform feature analysis.
- [x] Deploy the model to [app](https://reductelectro.streamlit.app).
- [] Verify model predictions with new experiments.
- [] Extend ML modeling to basic ones like RF, XGB, etc.
- [ ] Propose novel optimized experimental conditions for assessment.
- [ ] Prepare manuscript

<!-- License -->
<h2 id="License">License</h2>
This project is a work-in-progress and kindly desist from using the information contained here without notifying the authors.
