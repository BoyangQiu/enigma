import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from math import pi
import math
from celluloid import Camera
import matplotlib.animation as manimation
import os
import matplotlib.lines as mlines

# Get current directory
cwd = os.getcwd()

# Remove previous files to prevent pileup of files in folder
# List all files in folder
files = os.listdir(cwd+ '/static/plots')
# Set folder address
folder = cwd+ '/static/plots'
# Loop through file names, if any start with prefixes below, delete them
for fname in files:
    if fname.startswith('script'):
        os.remove(os.path.join(folder, fname))
    elif fname.startswith('plot'):
        os.remove(os.path.join(folder, fname))
    elif fname.startswith('prog'):
        os.remove(os.path.join(folder, fname))
    elif fname.startswith('static'):
        os.remove(os.path.join(folder, fname))    

# Read in the dataframe to be used in the backend
df = pd.read_csv('Master_df.csv',index_col = 0)

# Function to check the input name is found in the backend df
def isin(name):
    if name not in list(df['Name']):
        return False
    else:
        return True


'''
The code below is a copy of what was in the Jupyter Notebook titled '5. Dynamic Plots.ipynb'

The main difference is it was made into a function in order to be scalable based on user input.

For details on the code and process, please view the respective Jupyter Notebook file. 

While still documented, the code here may be missing comments explaining certain sections.
'''


# Define the function to create the radar plot
# Base code from: https://matplotlib.org/examples/api/radar_chart.html
# Modifications to make it a polygonal spine from: 
# https://stackoverflow.com/questions/52910187/how-to-make-a-polygon-radar-spider-chart-in-python
def radar(num_vars, frame = 'circle'):
    '''
    num_vars = # of variables
    frame = shape of frame (default = circle)
    '''
    
    # create evenly spaced vertices starting at 0, ending at 2*pi (360deg) for 'num_vars' times
    theta = np.linspace(0, 2*np.pi, num = num_vars, endpoint = False)
    
    class RadarAxes(PolarAxes):
        
        name = 'radar'
        # variable # of args
        # kwags for variable keyworded argument list
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # set first axis @ 90 degrees (ie. at top)
            self.set_theta_zero_location('N')
        
        def fill(self, *args, closed = True, **kwargs):
            # make line closed by default
            return super().fill(closed = closed, *args, **kwargs)
        
        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)
                
        # add labels at their corresponding degrees        
        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        # how to create frame
        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                    radius=.5, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)
        
        def draw(self, renderer):
            """ Draw. If frame is polygon, make gridlines polygon-shaped """
            if frame == 'polygon':
                gridlines = self.yaxis.get_gridlines()
                for gl in gridlines:
                    gl.get_path()._interpolation_steps = num_vars
            super().draw(renderer) 
        
        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                            spine_type='circle',
                            path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)


                return {'polar': spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)
    register_projection(RadarAxes)
    return theta

# Function for creating the dynamic radar plot
def dynamic(name, now):
    # Create sub-df for the specific player
    kershaw = df[df['Name'] == name]
    kershaw.sort_values(by=['Age'], inplace = True)
    kershaw.reset_index(drop = True, inplace = True)
    # Pull only the columns to be used for the plot
    plotdf1 = kershaw[['Name', 'Strikeouts','Velocity','Pitch Diversity', 'Control', 'Stamina','Ground Balls',  'WAR', 'Age']]

    # Set N = 6 for 6 vertices (ie. a hexagon)
    N = 6
    # Calculate angles for the vertices
    x_as = [n / float(N) * 2 * pi for n in range(N)]
    # Get the angles
    theta = radar(N, frame = 'polygon')

    # Set color of axes
    plt.rc('axes', linewidth=1.5, edgecolor="black", facecolor = "lightgrey")

    # Create polar plot
    fig, ax = plt.subplots(figsize=(4.5,4.5), sharex = True, sharey= True, subplot_kw = dict(projection='radar'))

    # Set axis limits
    plt.setp(ax, yticks = [20, 40, 60, 80], yticklabels = ["20", "40", "60", "80"])

    # Set canvas colour
    fig.patch.set_facecolor('#8a8a8a94')

    # Instatiate the Celluloid Camera to capture the plot figure
    camera = Camera(fig)
    for j in range(len(plotdf1)):
        # Set data
        x = np.array(plotdf1.columns[1:-2].values)
        y = np.array(pd.to_numeric(plotdf1.iloc[j,1:-2].values))
        # Set clockwise rotation. That is:
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)

        # Set position of radial-labels
        ax.set_rlabel_position(155)

        # Set color and linestyle of grid
        ax.xaxis.grid(True, color="black", linestyle='solid', linewidth=1)
        ax.yaxis.grid(True, color="black", linestyle='solid', linewidth=1)

        # Set number of radial axes and remove labels
        plt.xticks(x_as[:], [])

        # Plot data
        line = plt.plot(x_as, y, linewidth=1.8, color = 'b', linestyle='solid', zorder=3)
        # Fill area and place it on highest layer (set zorder to 20 just in case)
        ax.fill(x_as, y, 'skyblue', alpha=0.7, zorder = 20)

        # Set axes limits
        plt.ylim(0, 100)
        
        # Adjust positioning of vertex labels based on their position from 0 - 2pi (circle)
        for i in range(N):
            # Calculate angles of the 6 vertices
            angle_rad = i / float(N) * 2 * pi
            # At top center
            if angle_rad == 0:
                ha, distance_ax = "center", 5
            # On right side
            elif 0 < angle_rad < pi:
                ha, distance_ax = "left", 3
            # On bottom
            elif angle_rad == pi:
                ha, distance_ax = "center", 5
            # On left side
            else:
                ha, distance_ax = "right", 3
            # Insert the text @ the angle position
            ax.text(angle_rad, 100 + distance_ax, x[i], size=10, color = 'black', horizontalalignment=ha, verticalalignment="center")
        
        # Celluloid can't animate title frame by frame, so use the insert text function instead
        ax.text(0.5, 1.12, f'Season Attribute Percentile (Age: {plotdf1.loc[j, "Age"]})', size = 13, color = 'black', transform=ax.transAxes,
                horizontalalignment='center', verticalalignment="center")
        camera.snap()
    plt.tight_layout()  
    # Get the animation from Celluloid camera               
    animation = camera.animate()
    # Set the writer
    Writer = manimation.PillowWriter(fps=0.65)
    # Relative path filename
    filename = cwd + f'/static/plots/script{now}.gif'

    # Save the animation, add the facecolor kwarg to keep the background color
    # Saved with a dynamic file name because Flask has bad habit of caching objects in the Static folder, meaning they don't update if overwritten
    animation.save(filename, writer = Writer, savefig_kwargs={'facecolor':'#8a8a8a94'})

# Function to create a line plot of pitcher attribute progression over time
def progression(name, now):
    player = df[df['Name'] == name]
    avg_war = pd.DataFrame(df.groupby(['Season']).mean()['WAR'])
    prog = pd.DataFrame(player['Age'])
    prog['performance'] = player['performance'].copy()
    prog['WAR'] = player['WAR'].copy()
    prog['Season'] = player['Season'].copy()
    prog.reset_index(drop = True, inplace = True)

    for i in range(len(prog)-1):
        if prog.loc[i, 'Age']+1 != prog.loc[i+1, 'Age']:
            new_row = pd.DataFrame([[prog.loc[i, 'Age']+1, 0, 0, prog.loc[i, 'Season']+1]], 
                                   columns = ['Age', 'performance', 'WAR', 'Season'])
            prog = pd.concat([prog, new_row])

            
    prog = prog.merge(avg_war, on = 'Season', how = 'inner')
    prog.sort_values(by='Age', inplace = True)
    prog.reset_index(drop = True, inplace = True)

    fig, ax = plt.subplots(figsize = (5.7, 5))
    plt.subplots_adjust(bottom = 0.4)
    fig.patch.set_facecolor('#8a8a8a')
    # Set face color
    ax.set_facecolor('#8a8a8a')

    plt.xticks(prog['Age'], prog['Age'])

    ax.plot(prog['Age'], prog['performance'], linewidth = 2, marker = '.', markersize = 10, color = 'darkblue', zorder = 2, 
            label = 'Attributes')  
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('darkblue')
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.xaxis.set_tick_params(width=1.1, labelsize = 12)
    ax.yaxis.set_tick_params(width=1.1, labelsize = 12)
    ax.tick_params(axis='y', colors = 'darkblue')
    ax.set_ylim(0, 600)
    ax.set_xlim(prog['Age'].min(), prog['Age'].max())
    
    ax.set_ylabel('Performance \n (Sum of Attribute Percentiles)', fontsize = 10, labelpad = 12, color = 'darkblue')
    ax.set_xlabel('Age', fontsize = 12, labelpad = 10)
    ax.set_title('Performance Over Career', fontsize = 14, pad = 15)
    
    for i in range(len(prog)):
        if prog.loc[i, 'performance'] == prog['performance'].max():
            att_peak = ax.scatter(prog.loc[i, 'Age'], prog.loc[i, 'performance'], marker = '*',s = 250, color = 'yellow',
                                  label = f'Attribute Peak ({prog.loc[i, "Age"]})', zorder = 10)
    
    ax.axhline(y=300, linestyle = '--', c = '#C7CEEA', label = 'League Attribute Average', zorder = 0)
    
    ax2 = ax.twinx()
    
    ax2.set_ylabel('WAR (Wins Above Replacement)', labelpad = 12, color = 'maroon')
    ax2.yaxis.set_tick_params(width=1.1, labelsize = 12)
    # Set y limits, set max to 10 because no player has WAR over 10
    # Use if statement so the axis can either start at 0, but if player has negative WAR seasons then it has to start at negative number
    if prog['WAR_x'].min() >= 0:
        ax2.set_ylim(0, 10)
        ax2.set_yticks(np.arange(0, 11, 2))
    else:
        ax2.set_ylim(math.floor(prog['WAR_x'].min()), 10)
        ax2.set_yticks(np.arange(math.floor(prog['WAR_x'].min()), 11, 2))

    ax2.tick_params(axis='y', colors='maroon')
    ax2.spines['top'].set_color('none')
    ax2.spines['left'].set_color('none')
    ax2.spines['right'].set_linewidth(1.5)
    ax2.spines['right'].set_color('maroon')
    ax2.plot(prog['Age'], prog['WAR_x'], marker = '.', markersize = 10, color = 'maroon', label = 'WAR', zorder = 1)
    ax2.plot(prog['Age'], prog['WAR_y'], color = '#FF6961', label = 'League WAR Average', linestyle = '--', zorder = 0)
    for i in range(len(prog)):
        if prog.loc[i, 'WAR_x'] == prog['WAR_x'].max():
            war_peak = ax2.scatter(prog.loc[i, 'Age'], prog.loc[i, 'WAR_x'], marker = '*', s = 200, 
                                   color = 'orange', label = f'WAR Peak ({prog.loc[i, "Age"]})', zorder = 10)  
            
    
    att = mlines.Line2D([], [], linewidth = 2, marker = '.', markersize = 10, color = 'darkblue', label = 'Attributes')
    war = mlines.Line2D([], [], marker = '.', markersize = 10, color = 'maroon', label = 'WAR')
    att_avg = mlines.Line2D([], [], linestyle = '--', c = '#C7CEEA', label = 'League Attribute Average',)
    war_avg = mlines.Line2D([], [], color = '#FF6961', label = 'League WAR Average', linestyle = '--',)
    
    
    fig.legend(handles=[att, war, att_avg, war_avg, att_peak, war_peak], 
               fontsize = 10, edgecolor = 'none', facecolor = 'none', bbox_to_anchor=(1, 0.18), ncol = 3)
    # Save as a relative path
    filename = cwd + f'/static/plots/prog{now}.png'
    # Save and add in facecolor argument otherwise the background color won't be saved
    # Saved with a dynamic file name because Flask has bad habit of caching objects in the Static folder, meaning they don't update if overwritten
    plt.savefig(filename, facecolor=fig.get_facecolor(), bbox_inches='tight')




def static(name, now):
    agg = df.groupby(['Name']).mean()
    # Get the columns of interest in a particular order (useful for plotting later)
    agg = agg[['Strikeouts','Velocity','Pitch Diversity', 'Control', 'Stamina', 'Ground Balls']]
    player = agg[agg.index == name]

    N = 6

    x_as = [n / float(N) * 2 * pi for n in range(N)]

    theta = radar(N, frame = 'polygon')
    
    # Set color of axes
    plt.rc('axes', linewidth=1.5, edgecolor="black", facecolor = "lightgrey")

    # Create polar plot
    fig, ax = plt.subplots(figsize=(4.4,4.4), sharex = True, sharey= True, subplot_kw = dict(projection='radar'))

    # Set axis limits
    plt.setp(ax, yticks = [20, 40, 60, 80], yticklabels = ["20", "40", "60", "80"])

    # Set canvas colour
    fig.patch.set_facecolor('#8a8a8a94')

    # Set data
    x = np.array(player.columns.values)
    y = np.array(player.values[0])
    # Set clockwise rotation. That is:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    plt.setp(ax, yticks = [20, 40, 60, 80], yticklabels = ["20", "40", "60", "80"])

    # Set position of radial-labels
    ax.set_rlabel_position(155)


    # Set color and linestyle of grid
    ax.xaxis.grid(True, color="black", linestyle='solid', linewidth=1)
    ax.yaxis.grid(True, color="black", linestyle='solid', linewidth=1)


    # Set number of radial axes and remove labels
    plt.xticks(x_as[:], [])


    ax.plot(x_as, y, linewidth=1.8, color = 'b', linestyle='solid', zorder=3)
    # Fill area, place it on the highest layer (put 20 just in case)
    ax.fill(x_as, y, 'skyblue', alpha=0.7, zorder = 20)

    # Set axes limits
    plt.ylim(0, 100)
    #plt.yticks([0, 20, 40, 60, 80, 100], ["0", "20", "40", "60", "80", "100"])

    # Add title
    ax.set_title('Career Average Attribute Percentiles', position=(0.5, 1.1), size = 13, 
                 horizontalalignment='center', verticalalignment='center')

    # Adjust positioning of vertex labels based on their position from 0 - 2pi (circle)
    for i in range(N):
        # Calculate angles of the 6 vertices
        angle_rad = i / float(N) * 2 * pi
        # At top center
        if angle_rad == 0:
            ha, distance_ax = "center", 5.7
        # On right side
        elif 0 < angle_rad < pi:
            ha, distance_ax = "left", 3
        # On bottom
        elif angle_rad == pi:
            ha, distance_ax = "center", 5.7
        # On left side
        else:
            ha, distance_ax = "right", 3
        # Insert the text @ the angle position
        ax.text(angle_rad, 100 + distance_ax, x[i], size=10, horizontalalignment=ha, verticalalignment="center")
        
    plt.tight_layout()

    filename = cwd + f'/static/plots/static{now}.png'

    plt.savefig(filename, facecolor=fig.get_facecolor())