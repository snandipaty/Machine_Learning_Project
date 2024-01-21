"""
--- Graphical Visualisations ---

In this section we have the code for representing Graphical Visualisations of the initial dataset, such as:

--> 1. Plotting a simple bar plot with total mushrooms cap colours
        code followed from this link, with some changes, used mainly for first graph
        https://www.kaggle.com/code/mig555/mushroom-classification

--> 2. Plotting mushrooms which are edible & poisonous based on cap color
        


"""


"""--> Data Processing"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


#read dataset into variable
dataset = pd.read_csv("mushrooms.csv")


#count total amount of each color
#answer format "{color} {amount}"
c_colors = dataset['cap-color'].value_counts()

#make an array with each color's amount;
#to be used to specify the bar height of the specific color
#answer format "[amount1, amount2, ..]"
bar_height = c_colors.values.tolist()

#returns an array of color initials
cap_color_labels = c_colors.axes[0].tolist()

#---- 2nd Plot data processing

#list of poisonous color cap mushrooms
poison_c_colors = []
#list of edible color cap mushrooms
edible_c_colors = []

for cap_color in cap_color_labels:
    edible = len(dataset[(dataset['cap-color'] == cap_color) & (dataset['class'] == 'e')])
    edible_c_colors.append(edible)
    poison = len(dataset[(dataset['cap-color'] == cap_color) & (dataset['class'] == 'p')])
    poison_c_colors.append(poison)

"""--> 1. Plotting a simple bar plot with total mushrooms cap colours"""
# the position along x-axis where a bar in the bar plot will be placed
x = np.arange(len(c_colors))

# the width of the bars
width = 0.7        

#array of colors codes to be used in the subplot
colors = ['#958375','#AEABB0','#E73636','#FDD235','#F5F7F3','#F0DC82','#B8515D','#7F6265','#5B3442','#758B41']

#fig - top level container for all elements of a plot
#ax - subplot; the region of the figure that is used to plot the data
#plt.subplot used to create a figure and one or more subplots at the same time
fig1, ax = plt.subplots(figsize=(10,6))

#array of 10 bar objects
mushroom_bars = ax.bar(x, bar_height , width, color=colors)

#Add some text for labels, title and axes ticks
ax.set_xlabel("Cap Color",fontsize=20)
ax.set_ylabel('Quantity',fontsize=20)
ax.set_title('Mushroom Cap Color Quantity',fontsize=22)
ax.set_xticks(x) #Positioning on the x axis
ax.set_xticklabels(('brown', 'gray','red','yellow','white','buff','pink','cinnamon','purple','green'),
                  fontsize = 12)

#insert bars height size for each bar
def bar_height_size(bars,fontsize=10):
    """
    Attach a text label above each bar displaying its height
    """
    for bar in bars:
        #get the height of each bar
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., 1*height,'%d' % int(height),
                ha='center', va='bottom',fontsize=fontsize)

bar_height_size(mushroom_bars)  

#save an image to current working directory
#fig1.savefig('graphs/mushroom_cap_colors.png', dpi=200)


"""--> Plotting mushrooms which are edible & poisonous based on cap color"""
width = 0.4
fig2, ax = plt.subplots(figsize = (14, 7))

#initialise array with 10 edible bar objects
edible_bars = ax.bar(x, edible_c_colors, width, color="#ff5400")

#initialise array with 10 poisonous bar objects
poison_bars = ax.bar(x + width, poison_c_colors, width, color="#184e77")

ax.set_xlabel("Cap Color", fontsize=20)
ax.set_ylabel("Quantity", fontsize=20)
ax.set_title("Mushrooms that are edible & poisonous based on cap color")
ax.set_xticks(x+width/2)
ax.set_xticklabels(('brown', 'gray','red','yellow','white','buff','pink','cinnamon','purple','green'),
                  fontsize = 12)

#add legend to indicate the edible bars and poisonous bars
ax.legend((edible_bars, poison_bars), ('edible', 'poisonous'), fontsize=17)

bar_height_size(edible_bars)
bar_height_size(poison_bars)

#save file in graphs folder
#fig2.savefig('graphs/mushroom_edible_poison_cap_color.png', dpi=200)