#import torch

from re import A
import matplotlib.pyplot as plt
import numpy as np
import sys
#print("\nsyspath\n: ",sys.path,"\n")
from pyworld3 import World3, world3

import pyworld3


import sys
sys.path.append("create_dataset")

#pyworld3.world3.


from generate_dataset_classfile import Generate_dataset


################################################### IMPORTS FROM UTILS ##########################################################

from functools import wraps

import inspect
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
from matplotlib.image import imread
from numpy import isnan, full, nan

######################################################################################################################


def generate_statevars_dict(state_matrix=np.empty([601, 12]), est_matrix=np.empty([601, 12]) ):
    #List of variable names for all variables examined
        var_str_list=np.array(["al",
                            "pal",
                            "uil",
                            "lfert",
                            "ic",
                            "sc",
                            "ppol",
                            "p1",
                            "p2",
                            "p3",
                            "p4",
                            "nr"]).T       #Transposes it to get each variable as its own column - X matrix but names
        # Zip variable names to variable vectors in the state matrix
        state_vars_dict=dict( zip( var_str_list, state_matrix.T  ) )    # Transpose state matrix for correct alignment                                                                            
                                                                        #states_estimated    #on form 601*12
                                                                        #states_estimated.T on form 12*601
        var_str_list_est=np.char.add(var_str_list,"_est")
        est_state_vars_dict=dict( zip( var_str_list_est , est_matrix.T ) )     # Transpose state matrix for correct alignment
        #print(state_vars_dict)
        #print(est_state_vars_dict)
        return state_vars_dict, est_state_vars_dict
        

def plot_state_vars(state_matrix=np.empty([601, 12]), est_matrix= np.empty([601, 12]) , name=None, variables_included=["all"], time=np.arange(0,300+.5,0.5) ):

    state_vars_dict, est_state_vars_dict = generate_statevars_dict(state_matrix=state_matrix, est_matrix=est_matrix )
    plot_auto(est_state_vars_dict=est_state_vars_dict, state_vars_dict=state_vars_dict, name=name, variables_included=variables_included, time=time )





def plot_auto(est_state_vars_dict, state_vars_dict=dict() , name=None, variables_included=["all"], time=np.arange(0,300+.5,0.5)):

    '''
    Automated plotting

    est_state_vars_dict, state_vars_dict=dict() - Takes in the dicts of variable evolutions zipped with variable names

    variables_included=["all"] - Define which variables are plotted (a list of strings - ex: ["al","pal"])

    name=None - Name of the plot
    
    '''
    
    
    #Unsure if dict() will work as default parameter

    if variables_included!=["all"]:     #Ensure the ability to chose which variables to include
        est_state_vars_dict = {key+"_est": est_state_vars_dict[key+"_est"] for key in variables_included}
        if  state_vars_dict!=dict():
            state_vars_dict = {key: state_vars_dict[key] for key in variables_included}

    ###############     Generate parameters for plotting        #################
    orig_est_data=list( state_vars_dict.values() )+list( est_state_vars_dict.values() )
    #state_vars_dict.values().extend( est_state_vars_dict.values() )

    var_names= list( state_vars_dict.keys() ) + list( est_state_vars_dict.keys() )
    #state_vars_dict.keys().extend( est_state_vars_dict.keys() )

    var_maxes= np.amax( list( state_vars_dict.values() )+list( state_vars_dict.values() ), axis=1 )*1.2

    # Convert each element in var_maxes to a list containing 0 and the current element
    var_limits = [[0, max_val] for max_val in var_maxes] 

    #np.amax( state_vars_dict.values().extend( state_vars_dict.values() )

    #Define linestyles and widths based on est and orig model
    lines=["-"]*len(state_vars_dict.values()) + [":"]*len(est_state_vars_dict.values())

    widths=[.5]*len(state_vars_dict.values()) + [1.5]*len(est_state_vars_dict.values())

    ###############     Make a dict of the parameters to be sent to plotfunction        #################

    dict_of_plotvars= {
            "time" : time, 
            "var_data" : orig_est_data ,
            "var_names" : var_names ,
            "var_lims" : var_limits ,    #Add axis=1
            "img_background" : None,
            "title" :  None,    #Add
            "figsize" : (7, 5),                                   
            "grid" : True,
            "line_styles" : lines, 
            "line_widths" : widths
            }    #"dist_spines" : float = 0.09,
    
            #######     PLOT
    alt_plot_world_variables(**dict_of_plotvars, dist_spines=0.03)
    plt.show()



#How to call:
#test_variables=["al","pal","nr","p1"]
#state_vars_dict, est_state_vars_dict=generate_statevars_dict(standard_state_matrix, states_estimated)
#print(state_vars_dict)
'''for i in range(len(state_vars_dict["al"][0])):
    #print(state_vars_dict["al"][0][i])
     
    print("i: ",i,", real: ",state_vars_dict["al"][0][i],", est: ", state_vars_dict["al"][1][i])'''


#How to call:
#plot_auto(est_state_vars_dict, state_vars_dict)# , variables_included=test_variables )




# Get a colormap
#colormap = plt.cm.get_cmap('tab20')  # Get the 'tab20' colormap with 20 distinct colors

'''
from matplotlib import colormaps
plot_color_gradients(

(greens)
(blues)
(greys)
(reds)
(purples)'''




################################################ FROM PYWORLD3 UTILS    ################################################

################################ Added line_styles ######################################   AND WIDTHS

def alt_plot_world_variables(
    time,
    var_data,
    var_names,
    var_lims,
    img_background=None,
    title=None,
    figsize=None,
    dist_spines=0.09,
    grid=False,
    line_styles=["-"],  # New parameter for line styles
    line_widths=1
):
    """
    Plots world state from an instance of World3 or any single sector.

    """
    ###################     Gets colors for state vars  ###################
    varcol_dict=create_colorcycle(var_names)
    #print("varcoldict: ", varcol_dict)
    colors=list(varcol_dict.values())
    #print("\n\nCOLORS PRE: ",colors,"\n\n")
    if len(colors)<1:
        print("\n\nNOT PLOTTING STATE VARS\n\n")
        # Get default color cycle for plot lines
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]

    #print("\n\nCOLORS POST: ",colors,"\n\n")

    ############################################################################

    # Determine the number of variables
    var_number = len(var_data)

    #Assuming plotting model and estimated:
    var_number=int(var_number/2)

    # Create subplots with shared x-axis and multiple y-axes
    fig, host = plt.subplots(figsize=figsize)
    axs = [
        host,
    ]
    for i in range(var_number - 1):
        axs.append(host.twinx())

    # Adjust spacing between subplots
    fig.subplots_adjust(left=dist_spines * 1)
    for i, ax in enumerate(axs[1:]):
        ax.spines["left"].set_position(("axes", -(i + 1) * dist_spines))
        ax.spines["left"].set_visible(True)
        ax.yaxis.set_label_position("left")
        ax.yaxis.set_ticks_position("left")

    # Add background image if provided
    if img_background is not None:
        im = imread(img_background)
        axs[0].imshow(
            im,
            aspect="auto",
            extent=[time[0], time[-1], var_lims[0][0], var_lims[0][1]],
            cmap="gray",
        )

    
    # Plot data for each variable
    ps = []
    for i, [label, ydata, color, line_style, line_width] in enumerate(zip( var_names, var_data, colors, line_styles, line_widths) ): #Added line styles+ widths
        
       


        # Calculate the index of the axis to plot on
        ax_index = i % len(axs)
        ax = axs[ax_index]

        # Plot augmented lines with decided line style and widths
        ps.append(ax.plot(time, ydata, label=label, color=color, linestyle=line_style, linewidth=line_width)[0])
    axs[0].grid(grid)
    axs[0].set_xlim(time[0], time[-1])

    '''# Plot data for each variable
    ps = []
    for ax, label, ydata, color, line_style in zip(axs, var_names, var_data, colors, line_styles): #Added line styles
        # Plot augmented lines with decided line style
        used_labels.extend(label, str(label)+"_est")
        if label in
        print(color)
        ps.append(ax.plot(time, ydata, label=label, color=color, linestyle=line_style)[0])
    axs[0].grid(grid)
    axs[0].set_xlim(time[0], time[-1])'''

    # Set y-axis limits for each subplot
    for ax, lim in zip(axs, var_lims):
        if lim is not None:
            ax.set_ylim(lim[0], lim[1])

    # Format y-axis labels and ticks
    for ax_ in axs:
        formatter_ = EngFormatter(places=0, sep="\N{THIN SPACE}")
        ax_.tick_params(axis="y", rotation=90)
        ax_.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax_.yaxis.set_major_formatter(formatter_)

    # Format x-axis labels and ticks
    tkw = dict(size=4, width=1.5)
    axs[0].set_xlabel("time [years]")
    axs[0].tick_params(axis="x", **tkw)
    for i, (ax, p) in enumerate(zip(axs, ps)):
        ax.set_ylabel(p.get_label(), rotation="horizontal")
        ax.yaxis.label.set_color(p.get_color())
        ax.tick_params(axis="y", colors=p.get_color(), **tkw)
        ax.yaxis.set_label_coords(-i * dist_spines, 1.01)

    # Add title if provided
    if title is not None:
        fig.suptitle(title, x=0.95, ha="right", fontsize=10)

    # Adjust layout for better visualization
    plt.tight_layout()


##############################################    For defining variable-colors    ######################################################

def create_colorcycle(var_names):
    '''Generates a dict of statevars as keys and corresponding colors as values'''
        
    # Define the base colors
    base_colors = [ 'green', 'royalblue', 'chocolate', 'red', 'violet']
    #sectors=[  "agriculture"  ,  "capital"  ,  "pollution"  ,  "population"  ,  "resource"  ]
    variables=[ ["al","pal","uil","lfert"]  , ["ic","sc"] , ["ppol"] , ["p1","p2","p3","p4"] , ["nr"] ]
    
    
    
    

    #sectors_colors_dict= dict(zip(sectors, list(base_colors, dark_base_colors)))

    # Initialize the colormap list
    colors = []

    var_keys=[]

    for var, color in zip(variables, base_colors):
        #print("vars: ",var)
        #print("color: ", color)

        for i in range(len(var)):   # Defines the number of shades for each color - amount of vars per sector

            #print("i: ",i)
            shade = plt.cm.colors.to_rgba(color, alpha=(i + 1) / len(var))
            colors.append(shade)
        var_keys+=var
    #print("list of colors: ", colors," len " , len(colors))
    #print("varnames: ",var_names ," len " , len(var_names))
    

    sectors_colors_dict= dict(zip(var_keys, colors ))
    #print("sectors COLORDICT  ",sectors_colors_dict)

    varcolor_dict_values=[]

    varcol_dict=dict()
    for var in var_names:
        try:
            if var in var_keys:
                varcol_dict[var]=sectors_colors_dict[var]
                varcolor_dict_values+=var
            elif var[:-4] in var_keys:
                #print("estvar: ", var) 
                varcol_dict[ var ]=sectors_colors_dict[ var[:-4] ]
                varcolor_dict_values+=var
        except:
            print("\n\nINCORRECT VARIABLE NAME INPUTED TO PLOT FUNCTION\n\n")
            #print("wrongvar: ", var) 
            #print("wrongvar - _est: ", str(var)-"_est") 

    #varcol_dict=dict(zip(var_names , varcolor_dict_values))
    
    return varcol_dict
    ######################## For darker and lighter colors #######################
    ''' #from matplotlib.colors import ListedColormap
    dark_base_colors = [ 'darkgreen', 'darkblue', 'saddlebrown', 'darkred', 'purple']      # If we diff lightness for estimated vars

    # Define the number of shades for each color
    num_shades = 4

    # Initialize the colormap list
    colors = []
    dark_colors= []

    # Generate shades for each base color
    for color, dark_color in zip(base_colors, dark_base_colors):
        # Generate shades of the base color
        for i in range(num_shades):
            shade = plt.cm.colors.to_rgba(color, alpha=(i + 1) / num_shades)
            colors.append(shade)
            dark_shade = plt.cm.colors.to_rgba(dark_color, alpha=(i + 1) / num_shades)
            dark_colors.append(dark_shade)

    # Create the colormap
    custom_cmap = ListedColormap(colors)
    custom_cmap_dark = ListedColormap(dark_colors)
    var_str_list=np.array(["al","pal","uil","lfert","ic","sc","ppol","p1","p2","p3","p4","nr"]).T #Transposes it to get each variable as its own column 
    #Assign to state_vars
    color_vars_dict=dict( zip( var_str_list, list(custom_cmap_dark.values())  ) ) 

    var_str_list_est=np.char.add(var_str_list,"_est")
    est_color_vars_dict=dict( zip( var_str_list_est, list(custom_cmap.values())  ) )
    '''
    for i, color, dark_color in enumerate(zip(custom_cmap, custom_cmap_dark)):
        pass
    # Display the colormap
    #plt.imshow(np.linspace(0, 1, 100).reshape(10, 10), cmap=custom_cmap)
    plt.imshow(np.linspace(0, 1, 100).reshape(20, 5), cmap=custom_cmap_dark)
    plt.colorbar()
    plt.show()