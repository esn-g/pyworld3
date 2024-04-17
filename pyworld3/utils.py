# -*- coding: utf-8 -*-

# © Copyright Charles Vanwynsberghe (2021)

# Pyworld3 is a computer program whose purpose is to run configurable
# simulations of the World3 model as described in the book "Dynamics
# of Growth in a Finite World".

# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software. You can use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".

# As a counterpart to the access to the source code and rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty and the software's author, the holder of the
# economic rights, and the successive licensors have only limited
# liability.

# In this respect, the user's attention is drawn to the risks associated
# with loading, using, modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean that it is complicated to manipulate, and that also
# therefore means that it is reserved for developers and experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and, more generally, to use and operate it in the
# same conditions as regards security.

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.

from functools import wraps

import inspect
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
from matplotlib.image import imread
from numpy import isnan, full, nan

verbose_debug = False


def _create_control_function(instance, default_control_functions, control_functions):
    for func_name, default_func in default_control_functions.items():
        control_function = control_functions.get(func_name, default_func)
        number_arguments_control_function = len(
            inspect.signature(control_function).parameters
        )
        if number_arguments_control_function == 3:
            # Feedback control
            refactored_function = lambda k, control_function=control_function, default_func=default_func: (
                default_func(0)
                if k <= 1
                else control_function(instance.time[k], instance, k - 1)
            )
        elif number_arguments_control_function == 1:
            # Open loop control
            refactored_function = (
                lambda k, control_function=control_function: control_function(
                    instance.time[k]
                )
            )
        else:
            raise Exception(
                f"Incorrect number of arguments in control function {func_name}. Got {number_arguments_control_function}, expected 1 or 3."
            )
        setattr(instance, func_name, refactored_function)
        setattr(instance, f"{func_name}_values", full((instance.n,), nan))


def requires(outputs=None, inputs=None, check_at_init=True, check_after_init=True):
    """
    Decorator generator to reschedule all updates of current loop, if all
    required inputs of the current update are not known.

    """

    def requires_decorator(updater):
        if verbose_debug:
            print(
                """Define the update requirements...
                  - inputs:  {}
                  - outputs: {}
                  - check at init [k=0]:    {}
                  - check after init [k>0]: {}""".format(
                    inputs, outputs, check_at_init, check_after_init
                )
            )
            print(
                "... and create a requires decorator for the update function",
                updater.__name__,
            )

        @wraps(updater)
        def requires_and_update(self, *args):
            k = args[0]
            go_grant = (k == 0) and check_at_init or (k > 0) and check_after_init
            if inputs is not None and go_grant:
                for input_ in inputs:
                    input_arr = getattr(self, input_.lower())
                    if isnan(input_arr[k]):
                        if self.verbose:
                            warn_msg = "Warning, {} unknown for current k={} -"
                            print(warn_msg.format(input_, k), updater.__name__)
                            print("Rescheduling current loop")
                        self.redo_loop = True

            return updater(self, *args)

        return requires_and_update

    return requires_decorator



################################ Added line_styles ######################################

def plot_world_variables(
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
):
    """
    Plots world state from an instance of World3 or any single sector.

    """
    ###################     Gets colors for state vars  ###################
    varcol_dict=create_colorcycle(var_names)
    colors=list(varcol_dict.values())
    print("\n\n\n\nCOLORS: ",colors,"\n\n\n\n")
    if len(colors)<1:
        print("\n\nNOT PLOTTING STATE VARS\n\n")
        # Get default color cycle for plot lines
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]

    print("\n\n\n\nCOLORS: ",colors,"\n\n\n\n")
    # Determine the number of variables
    var_number = len(var_data)

    # Create subplots with shared x-axis and multiple y-axes
    fig, host = plt.subplots(figsize=figsize)
    axs = [
        host,
    ]
    for i in range(var_number - 1):
        axs.append(host.twinx())

    # Adjust spacing between subplots
    fig.subplots_adjust(left=dist_spines * 2)
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
    for ax, label, ydata, color, line_style in zip(axs, var_names, var_data, colors, line_styles): #Added line styles
        # Plot augmented lines with decided line style
        ps.append(ax.plot(time, ydata, label=label, color=color, linestyle=line_style)[0])
    axs[0].grid(grid)
    axs[0].set_xlim(time[0], time[-1])

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

        for i in range(len(var)):   # Defines the number of shades for each color - amount of vars per sector
            shade = plt.cm.colors.to_rgba(color, alpha=(i + 1) / len(var))
            colors.append(shade)
        var_keys+=var

    

    sectors_colors_dict= dict(zip(var_keys, colors ))

    varcolor_dict_values=[]

    for var in var_names:
        try:
            if var in var_keys:
                varcolor_dict_values+=var
            elif var-"_est" in var_keys:
                print("wrongvar: ", var)
                varcolor_dict_values+=var
        except:
            print("\n\nINCORRECT VARIABLE NAME INPUTED TO PLOT FUNCTIO\n\n")

    varcol_dict=dict(zip(var_names , varcolor_dict_values))


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