<img src = "https://puu.sh/G4cPn/3cf1822825.png">

Repository for my Brainstation spring 2020 Capstone project.

The scope of the Capstone project is to predict what pitch a pitcher will throw during a specific scenario. It is complete with data visualizations which provide a scouting report of their pitching attributes, performance over time, frequent pitch locations and a breakdown of the proportional usage of their pitch arsenal.

Unique requirements:

`Celluloid`

Can be installed via `pip install celluloid`

Other requirements:

- `Flask`
- `TensorFlow 2.x`
- `matplotlib 3.2.0`
- `scikit-learn`
- `pandas`
- `numpy `



  
*** 
 

If running from local machine, please ensure all files in the following folders are downloaded and the folder structure is the same as here:

```
  
Main folder	
|
|___ static
|	|___ plots
|	|___ staticplots
|
|___ saved_model (containing folders for each player's model)
|
|___ templates

```

The `Notebooks` folder is a compilation of my end-to-end workflow for this project. It includes detailed documentation of the code as well as my thought process throughout. It is intended to answer any outstanding questions about the project, code or project flow.

The Files in the `Notebooks` folder are not necessary for the application to run.


Once the files and folders have been downloaded and set up, you may run the `enigma.py` script.

It will render a web page on **localhost:5000** where the web application is hosted.

***

### Enigma Walkthrough:

Once Enigma is launched, this will be the homepage that greets you when you visit localhost:5000 in a browser:

<img src = "https://puu.sh/G4cPn/3cf1822825.png">

On the page, you can then select a pitcher from the dropdown list that you wish to predict the next pitch for:

<img src = "https://puu.sh/G4cVs/bdb9e37204.png">

After you select a pitcher, below is a web form prompting you to enter in the pre-pitch circumstances:

<img src = "https://puu.sh/G4cZt/0c8d6eb20e.png">

Hitting "SUBMIT" will take you to the results page, where you will be returned a summary of the scenario chosen, the most likely pitch type as predicted by the ML model given the circumstances inputted, as well as the probabilities of the other pitch types.

<img src = "https://puu.sh/G4d0D/122a332576.png">

Below this will also be a group of data visualizations displaying a scouting report of the pitcher of interest, named PitchHUB:

<img src = "https://puu.sh/G4d2Z/629d2e2dd1.gif">
