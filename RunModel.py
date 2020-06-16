import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
import os

# Delete old plots from previous sessions to prevent pileup
folder = os.listdir('C:/Users/Boyang Qiu/Desktop/Brainstation/Capstone/static/plots/')
for fname in folder:
    if fname.startswith('prob_'):
        os.remove(os.path.join('C:/Users/Boyang Qiu/Desktop/Brainstation/Capstone/static/plots/', fname))


# Function for returning the pitch scenario
def scenario(user_input):
    situation = 'Pitch scenario: '

    # If clause for baserunners
    if user_input['on_3b'].all() == 1 and user_input['on_2b'].all() == 1 and user_input['on_1b'].all() == 1:
        situation = situation + 'bases loaded, '
    elif user_input['on_3b'].all() == 1 and user_input['on_1b'].all() == 1:
        situation = situation + 'runners on the corners, '
    elif user_input['on_2b'].all() == 1 and user_input['on_1b'].all() == 1:
        situation = situation + 'runners on first and second, '
    elif user_input['on_3b'].all() == 1 and user_input['on_2b'].all() == 1:
        situation = situation + 'runners on second and third, '
    elif user_input['on_3b'].all() == 1:
        situation = situation + 'runner on 3rd, '
    elif user_input['on_2b'].all() == 1:
        situation = situation + 'runner on 2nd, '
    elif user_input['on_1b'].all() == 1:
        situation = situation + 'runner on 1st, '
    else:
        situation = situation + 'bases empty, '

    # If clause for balls and strikes
    if user_input['b__3'].all() == 1 and user_input['s__2'].all() == 1:
        situation = situation + 'full count, '
    elif user_input['b__3'].all() == 1 and user_input['s__1'].all() == 1:
        situation = situation + '3-1 count, '
    elif user_input['b__3'].all() == 1 and user_input['s__0'].all() == 1:
        situation = situation + '3-0 count, '
    elif user_input['b__2'].all() == 1 and user_input['s__2'].all() == 1:
        situation = situation + '2-2 count, '
    elif user_input['b__2'].all() == 1 and user_input['s__1'].all() == 1:
        situation = situation + '2-1 count, '
    elif user_input['b__2'].all() == 1 and user_input['s__0'].all() == 1:
        situation = situation + '2-0 count, '
    elif user_input['b__1'].all() == 1 and user_input['s__2'].all() == 1:
        situation = situation + '1-2 count, '
    elif user_input['b__1'].all() == 1 and user_input['s__1'].all() == 1:
        situation = situation + '1-1 count, '
    elif user_input['b__1'].all() == 1 and user_input['s__0'].all() == 1:
        situation = situation + '1-0 count, '
    elif user_input['b__0'].all() == 1 and user_input['s__2'].all() == 1:
        situation = situation + '0-2 count, '
    elif user_input['b__0'].all() == 1 and user_input['s__1'].all() == 1:
        situation = situation + '0-1 count, '
    elif user_input['b__0'].all() == 1 and user_input['s__0'].all() == 1:
        situation = situation + '0-0 count, '

    # If clause for outs
    if user_input['o__0'].all() == 1:
        situation = situation + '0 outs, '
    elif user_input['o__1'].all() == 1:
        situation = situation + '1 out, '  
    else:
        situation = situation + '2 outs, '

    # If clause for inning
    if user_input['early'].all() == 1:
        situation = situation + 'early in the game, '
    elif user_input['mid'].all() == 1:
        situation = situation + 'middle of the game, '
    else:
        situation = situation + 'late in the game, '

    # If clause for batter hand
    if user_input['bat_left'].all() == 1:
        situation = situation + 'left-handed batter at the plate, '
    else:
        situation = situation + 'right-handed batter at the plate, '

    # Fill in score
    situation = situation + f'in a {user_input["fld_score"].values[0]}-{user_input["bat_score"].values[0]} '

    # If clause for game type
    if user_input['reg_season'].all() == 1:
        situation = situation + 'regular season game.'
    else:
        situation = situation + 'playoff game.'

    return situation

# Function for running the model based on user input
def run_model(name, user_input, now):
    # Get the last name from the full name in lower case
    lastname = name.split()[1].lower()

    # Load the correct model
    model = tf.keras.models.load_model(f'saved_model/{lastname}')
    # Get the predicted probabilities
    pred_proba = model.predict(user_input)
    # Pull the max probability as the predicted pitch
    pred_pitch = np.argmax(pred_proba,axis=1)

    
    # Read in the possible pitch names
    df = pd.read_csv('pitch_names.csv', index_col = 0)
    # Store the number of different pitches as a varialbe
    pitches = len(df[df.index == name].T.dropna())
    # The number of ticks on the plot based on # of different pitches
    indices = np.arange(pitches)
    # Width of bars
    width = 0.5

    # Get the row where name equals name input
    # Drop NA columns if exists
    # Put the value to list format, default is a list inside a list of lists, use pop() to get it out of the outer list
    xnames = df[df.index == name].dropna(axis = 1).values.tolist().pop()
    
    # Convert the array of integers into the pitch names using nested np.where functions
    # For 2 pitch pitchers
    if len(xnames) == 2:
        pred_pitch = np.where(pred_pitch == 0, xnames.values[0], 
                                np.where(pred_pitch == 1, xnames.values[1], pred_pitch))
        # For 3 pitch pitchers:
        elif len(xnames) == 3: 
        pred_pitch = np.where(pred_pitch == 0, xnames.values[0], 
                                np.where(pred_pitch == 1, xnames.values[1], 
                                        np.where(pred_pitch == 2, xnames.values[2], pred_pitch)))
        # For 4 pitch pitchers:  
        elif len(xnames) == 4:
        pred_pitch = np.where(pred_pitch == 0, xnames.values[0], 
                                np.where(pred_pitch == 1, xnames.values[1], 
                                        np.where(pred_pitch == 2, xnames.values[2], 
                                                np.where(pred_pitch == 3, xnames.values[3], pred_pitch))))
                                 
    # Pred proba returns an array of lists, need to get just the first list of values
    # Set a color list of all light grey which is same length as number of bars
    colors = np.array(['#f0f0f0']*len(pred_proba[0]))
    # Modify the color at the index where the bar height is at the max to dark blue to stand out
    colors[pred_proba[0] == pred_proba[0].max()] = '#0f52ba'
    
    # Do the same for the edgecolors to further highlight top probability
    edgecolors = np.array(['#f0f0f0']*len(pred_proba[0]))
    edgecolors[pred_proba[0] == pred_proba[0].max()] = '#ea3c53'
    
    fig, ax = plt.subplots()
    
    ax.bar(indices, pred_proba[0], width, edgecolor = edgecolors, linewidth='1.3', color = colors)
    
    # Add padding to improve spacing between axis, axis labels, and axis title   
    ax.set_ylabel('Predicted Probability', labelpad = 5)
    ax.set_xlabel('Pitch Type', labelpad=10)
    ax.tick_params(axis='x', which='major', pad=5)
        
    plt.xticks(indices, xnames)
    
    # Remove some of spines for aesthetics
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_bounds(indices.min()-width/2, indices.max()+width/2)
    ax.spines['left'].set_bounds(0, pred_proba[0].max())
    # Save it to a unique timestamp
    plt.savefig(f'C:/Users/Boyang Qiu/Desktop/Brainstation/Capstone/static/plots/prob_{now}.png')
    # Return a sentence summary with the probability rounded to 2 decimal places
    prediction = 'Predicted Pitch: ' + pred_pitch[0] + " ({:.2g}% predicted probability)".format(np.amax(pred_proba)*100)
    return prediction