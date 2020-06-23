from flask import Flask, flash, redirect, render_template, request, session, abort
import time
import pandas as pd
import numpy as np
# Import my custom scripts
import Animation
import prime
import RunModel as rm

# Instatiate the Flask app
app = Flask(__name__)

# Render the home page
@app.route('/')
def hello():
    return render_template('temp.html')

# Create the 'About' page
@app.route('/about')
def about():
    return render_template('about.html')

# Create the response page after submission
@app.route('/send', methods = ['POST'])
def send():
    # If something is sent via the submit button
    if request.method == 'POST':
        # Turn the inputted option in the drop down to a string and save it as a variable
        name = request.form.get('name')
        name = str(name)
        # Store the current timestamp as a variable for later plot naming
        now = time.strftime("%Y%m%d-%H%M%S")
        # Check to make sure the name is in the dataframes in the backend
        isin = Animation.isin(name)

        # Get last name by splitting the full name and pull out the second element
        # Turn it lower case to match the file name
        lastname = name.split()[1].lower()
        # Get the usage file name for everybody
        usagefile = name + '_usage_light.png'
        
        # Conditionals to fill in based on the user input
        strikes = int(request.form.get('strikes'))
        balls = int(request.form.get('balls'))
        outs = int(request.form.get('outs'))
        bat = str(request.form.get('bat'))
        inning = int(request.form.get('inning'))
        prev = str(request.form.get('pitch'))
        game = str(request.form.get('game'))
        r1 = str(request.form.get('r1'))
        r2 = str(request.form.get('r2'))
        r3 = str(request.form.get('r3'))
        # Input fields must be retrieved a bit differently than drop down/checkboxes
        pit_score = int(request.form['pscore'])
        bat_score = int(request.form['bscore'])

        # Read in the template
        df = pd.read_csv('for_inputs.csv', index_col=0)
        # Fill it out based on form values
        # Scores are filled out as is, no need for if statements
        df['bat_score'] = bat_score
        df['fld_score'] = pit_score
        # If statements for strikes
        if strikes == 0:
            df['s__0'] = 1
        elif strikes == 1:
            df['s__1'] = 1
        else:
            df['s__2'] = 1

        # If statements for balls
        if balls == 0:
            df['b__0'] = 1
        elif balls == 1:
            df['b__1'] = 1
        elif balls == 2:
            df['b__2'] = 1
        else:
            df['b__3'] = 1

        # If statements for outs
        if outs == 0:
            df['o__0'] = 1
        elif outs == 1:
            df['o__1'] = 1
        else:
            df['o__2'] = 1
            
        # If statements for innings
        if inning < 4:
            df['early'] = 1
        elif inning > 3 and inning < 7:
            df['mid'] = 1
        elif inning > 6:
            df['late'] = 1

        # If statements for batter handedness
        if bat == 'right':
            df['bat_right'] = 1
        elif bat == 'left':
            df['bat_left'] = 1

        # If statements for game type
        if game == 'reg':
            df['reg_season'] = 1
        elif game == "post":
            df['post_season'] = 1

        # If statements for previous pitch
        # Conditional only for Zack Greinke since he's the only one with 'junk' pitches
        # Its 'proper' place is before the 'prev__None' columns
        if name == 'Zack Greinke':
            # argwhere creates an array of arrays, fish out the number through indexing to get rid of layers
            # Need to make sure to turn to int instead of np.int for the pd.insert function
            loc = int(np.argwhere(df.columns == 'prev__None')[0][0])
            df.insert(loc, 'prev__Junk', 0)
        
        if prev == 'fb':
            df['prev__Fastball'] = 1
        elif prev == 'bb':
            df['prev__Breaking Ball'] = 1
        elif prev == 'os':
            df['prev__Off-speed'] = 1
        elif prev == "jk":
            df['prev__Junk'] = 1

        # If statements for runners
        # Want to consider all options so make 3 different if clauses rather than elif clauses (terminate if clause once one equals True)
        # Default is all zeroes (ie. no runners on), so no need to include that into the clause
        if r1 == 'first':
            df['on_1b'] = 1
        if r2 == 'second':
            df['on_2b'] = 1
        if r3 == 'third':
            df['on_3b'] = 1



        # If the player name does exist
        if isin == True:
            
            
            pitchfile = name +'_pitches.png'
            # Run the model with the name of player, user-input dataframe and current timestamp
            pred = rm.run_model(name, df, now)
            # Run the scenario sentence generator with the user-input dataframe
            scenario = rm.scenario(df)
            proba_plot = 'prob_' + now + '.png'
            
            # Call the custome prime.py script to calculate a player's prime years
            prime_years = prime.prime(name)
            # Run the create dynamic radar plot function
            Animation.dynamic(name, now)
            # Create the name of the animated plot to be made
            filename = 'script' + now + '.gif'
            # Run the static radar plot of average career attributes
            Animation.static(name, now)
            # Plot file name
            staticradar = 'static' + now + '.png'
            # Run the create plot of career attributes over age function
            Animation.progression(name, now)
            prog_name = 'prog' + now + '.png'
            return render_template('send.html', playername = name, file = filename, pitchfile = pitchfile, usagefile = usagefile, proba_plot = proba_plot, pred = pred, 
            situation = scenario, years = prime_years, progfile = prog_name, staticradar = staticradar)

        # If the player name does not exist in the database then send them to an error page
        else:
            return render_template('error.html')


# Run the script when it is ran on its own
if __name__ == "__main__":
    app.run()