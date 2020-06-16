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
from celluloid import Camera
import matplotlib.animation as manimation
import os

# Remove previous files to prevent pileup of files in folder
folder = os.listdir('C:/Users/Boyang Qiu/Desktop/Brainstation/Capstone/static/plots/')
for fname in folder:
    if fname.startswith('script'):
        os.remove(os.path.join('C:/Users/Boyang Qiu/Desktop/Brainstation/Capstone/static/plots/', fname))
    elif fname.startswith('plot'):
        os.remove(os.path.join('C:/Users/Boyang Qiu/Desktop/Brainstation/Capstone/static/plots/', fname))


df = pd.read_csv('Master_df.csv',index_col = 0)

def isin(name):
    if name not in list(df['Name']):
        return False
    else:
        return True

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

def dynamic(name, now):
        kershaw = df[df['Name'] == name]
        kershaw.sort_values(by=['Age'], inplace = True)
        kershaw.reset_index(drop = True, inplace = True)

        plotdf1 = kershaw[['Name', 'Strikeouts','Velocity','Pitch Diversity', 'Control', 'Stamina','Ground Balls',  'WAR', 'Age']]
        plotdf1.sort_values(by=['Age'], inplace = True)
        plotdf1.reset_index(drop=True, inplace=True)


        N = 6

        x_as = [n / float(N) * 2 * pi for n in range(N)]

        theta = radar(N, frame = 'polygon')

        # Set color of axes
        plt.rc('axes', linewidth=1, edgecolor="black", facecolor = "white")

        # Create polar plot
        fig, ax = plt.subplots(figsize=(5,5), sharex = True, sharey= True, subplot_kw = dict(projection='radar'))
        #ax = ax.flat

        # Set axis limits
        plt.setp(ax, yticks = [20, 40, 60, 80], yticklabels = ["20", "40", "60", "80"])

        # Set canvas colour
        #fig.patch.set_facecolor('xkcd:mint green')
        camera = Camera(fig)
        for j in range(len(plotdf1)):
            # Set data
            x = np.array(plotdf1.columns[1:-2].values)
            y = np.array(pd.to_numeric(plotdf1.iloc[j,1:-2].values))
            # Set clockwise rotation. That is:
            ax.set_theta_offset(pi / 2)
            ax.set_theta_direction(-1)


            # Set position of radial-labels
            #ax.set_rticks([0, 20, 40, 60, 80, 100])
            ax.set_rlabel_position(150)


            # Set color and linestyle of grid
            ax.xaxis.grid(True, color="black", linestyle='solid', linewidth=1)
            ax.yaxis.grid(True, color="black", linestyle='solid', linewidth=1)


            # Set number of radial axes and remove labels
            plt.xticks(x_as[:], [])


            # Plot data
            line = plt.plot(x_as, y, linewidth=1.5, color = 'b', linestyle='solid', zorder=3)
            # Fill area
            ax.fill(x_as, y, 'skyblue', alpha=0.95)

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
                ax.text(angle_rad, 100 + distance_ax, x[i], size=10, horizontalalignment=ha, verticalalignment="center")
            
            # Celluloid can't animate title frame by frame, so use the insert text function instead
            ax.text(0.5, 1.12, f'Season Attribute Percentile (Age: {plotdf1.loc[j, "Age"]})', size = 13, transform=ax.transAxes,
                    horizontalalignment='center', verticalalignment="center")
            camera.snap()
        plt.tight_layout()                 
        animation = camera.animate()
        Writer = manimation.PillowWriter(fps=0.5)
        animation.save(f'C:/Users/Boyang Qiu/Desktop/Brainstation/Capstone/static/plots/script{now}.gif', writer = Writer)

def static(name, now):

    kershaw = df[df['Name'] == name]
    kershaw.sort_values(by=['Age'], inplace = True)
    kershaw.reset_index(drop = True, inplace = True)

    plotdf1 = kershaw[['Name', 'Strikeouts','Velocity','Pitch Diversity', 'Control', 'Stamina','Ground Balls',  'WAR', 'Age']]
    plotdf1.sort_values(by=['Age'], inplace = True)
    plotdf1.reset_index(drop=True, inplace=True)


    N = 6

    x_as = [n / float(N) * 2 * pi for n in range(N)]

    theta = radar(N, frame = 'polygon')

    # Set color of axes
    plt.rc('axes', linewidth=1, edgecolor="black", facecolor = "white")

    cols = 3
    rows = (len(plotdf1)//3 + ((len(plotdf1)%3)//3)+1)

    # Create polar plot
    fig, ax = plt.subplots(rows, cols, figsize=(14.5, 16),
                        sharex = True, sharey= True, subplot_kw = dict(projection='radar'))
    ax = ax.flat


    # Set canvas colour
    #fig.patch.set_facecolor('xkcd:mint green')
    for j in range(len(plotdf1)):
        # Set data
        x = np.array(plotdf1.columns[1:-2].values)
        y = np.array(pd.to_numeric(plotdf1.iloc[j,1:-2].values))
        # Set clockwise rotation. That is:
        ax[j].set_theta_offset(pi / 2)
        ax[j].set_theta_direction(-1)
        plt.setp(ax[j], yticks = [20, 40, 60, 80], yticklabels = ["20", "40", "60", "80"])

        # Set position of radial-labels
        #ax.set_rticks([0, 20, 40, 60, 80, 100])
        ax[j].set_rlabel_position(150)


        # Set color and linestyle of grid
        ax[j].xaxis.grid(True, color="black", linestyle='solid', linewidth=1)
        ax[j].yaxis.grid(True, color="black", linestyle='solid', linewidth=1)


        # Set number of radial axes and remove labels
        plt.xticks(x_as[:], [])


        ax[j].plot(x_as, y, linewidth=1.5, color = 'b', linestyle='solid', zorder=3)
        # Fill area
        ax[j].fill(x_as, y, 'skyblue', alpha=0.95)

        # Set axes limits
        plt.ylim(0, 100)
        #plt.yticks([0, 20, 40, 60, 80, 100], ["0", "20", "40", "60", "80", "100"])
        
        # Add title
        ax[j].set_title(f'Age: {plotdf1.loc[j, "Age"]}, WAR: {plotdf1.loc[j, "WAR"]}', 
                        position=(0.5, 1.1), 
                        size = 14, horizontalalignment='center', verticalalignment='center')
        
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
            ax[j].text(angle_rad, 100 + distance_ax, x[i], size=10, horizontalalignment=ha, verticalalignment="center")
        
    # Remove any extra empty subplots
    remainders = (rows*cols) - len(plotdf1)
    if remainders > 0:
        for n in np.arange(1, remainders+1):
            fig.delaxes(ax[-n])
        
    #plt.tight_layout()
    plt.subplots_adjust(wspace = 0.6, hspace=0.4)
    plt.savefig(f'C:/Users/Boyang Qiu/Desktop/Brainstation/Capstone/static/plots/plot{now}.png')


