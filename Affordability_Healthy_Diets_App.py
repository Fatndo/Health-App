#!/usr/bin/env python
# coding: utf-8

# # Importing The necessary Libraries

# In[242]:


import streamlit as st # to build attractive user interfaces in no time.
import pandas as pd # data analysis library
import numpy as np # support arrays and matrices library
#import chart_studio.plotly as py #provides a web-service for hosting graphs
import plotly.offline as po #generate graphs offline
import plotly.graph_objs as pg #to represent figures
import matplotlib.pyplot as plt #interactive visualizations library
import plotly.express as px # create interactive plots with very little code
import seaborn as sns #data visualization library
#%matplotlib inline

import warnings # to ignore warnings
warnings.filterwarnings('ignore')


# # Data Loading

# In[243]:


# fetching data

df = pd.read_csv("health Data.csv",encoding='latin-1')


# # Data Preprocessing

# # 1- Dropping the unnecessary columns

# In[244]:


#Dropping the unnecessary columns

new_df = df.drop(['Classification Name', 'Classification Code','Time Code'], axis=1)


#st.dataframe(new_df)


# # Showing dataset columns

# In[231]:


# browsing dataset columns
#new_df.columns
#new_df.columns


# # 2- Changing the columns names

# by using their abbreviations

# In[232]:


#On the left there is position for the column and on the right there is the new name to replace with

new_df.columns.values[3] = 'CoHD_fexp'
new_df.columns.values[4] = 'CoHD_pov'
new_df.columns.values[5] = 'CoNA_fexp'
new_df.columns.values[6] = 'CoNA_pov'
new_df.columns.values[7] = 'CoCA_fexp'
new_df.columns.values[8] = 'CoCA_pov'
new_df.columns.values[9] = 'CoHD'
new_df.columns.values[10] = 'CoHD_CoCA'
new_df.columns.values[11] = 'CoNA'
new_df.columns.values[12] = 'CoCA'
new_df.columns.values[13] = 'CoHD_asf'
new_df.columns.values[14] = 'CoHD_asf_ss'
new_df.columns.values[15] = 'CoHD_f'
new_df.columns.values[16] = 'CoHD_f_ss'
new_df.columns.values[17] = 'CoHD_lns'
new_df.columns.values[18] = 'CoHD_lns_ss'
new_df.columns.values[19] = 'CoHD_of'
new_df.columns.values[20] = 'CoHD_of_ss'
new_df.columns.values[21] = 'CoHD_ss'
new_df.columns.values[22] = 'CoHD_v'
new_df.columns.values[23] = 'CoHD_v_ss'
new_df.columns.values[24] = 'CoHD_asf_prop'
new_df.columns.values[25] = 'CoHD_f_prop'
new_df.columns.values[26] = "CoHD_lns_prop"
new_df.columns.values[27] = "CoHD_of_prop"
new_df.columns.values[28] = "CoHD_v_prop"
new_df.columns.values[29] = "CoHD_ss_prop"
new_df.columns.values[30] = "CoCA_headcount"
new_df.columns.values[31] = "CoNA_headcount"
new_df.columns.values[32] = "CoHD_headcount"
new_df.columns.values[33] = 'CoCA_unafford_n'
new_df.columns.values[34] = 'CoHD_unafford_n'
new_df.columns.values[35] = 'CoNA_unafford_n'
new_df.columns.values[36] = 'Pop'


#st.dataframe(new_df)


# # 3- Handling the missing values

# In[233]:


#drop rows with missing values in 'Date' column
new_df = new_df[new_df['Time'].notna()]

# Firstly convert to string type
all_columns = list(new_df) # Creates list of all column headers
new_df[all_columns] = new_df[all_columns].astype('string')

# Replacing all value with '..' by NaN
new_df=new_df.mask(new_df == '..')


# Filling the missing value 

df1 = new_df[['Country Name','Country Code']]
df2 = new_df.drop(['Country Name','Country Code'], axis = 1).astype(float)

# Fill the missing values with the previous row value using ffill method
df2 = df2.ffill(axis = 'index' )

# Concat the dataframes

Data = pd.concat([df1, df2],axis = 1, join = 'outer', ignore_index=False, sort=False)

#st.dataframe(Data)


# In[234]:


#Allow only 2 decimal points
pd.options.display.float_format = '{:.2f}'.format

#Clean date column from extra zeros
Data['Time'] = Data['Time'].astype(int)

#st.dataframe(Data)


# # 4- Drop Duplicate Rows

# In[235]:


# Drop Duplicate Rows in dataframe

Data = Data.drop_duplicates(['Country Name','Time'],keep= 'last')

#st.dataframe(Data)

st.set_page_config(layout="wide")

st.markdown("""
<style>
.big-font {
    font-size:100px !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font"> Welcome To Healthy Diets Affordability App! \N{Broccoli} \N{Poultry Leg} \N{Glass of Milk} </p>', unsafe_allow_html=True)

# # Visualization / Dashboard

# In[236]:


# interactive bar chart

fig = px.bar(Data, x='CoHD_unafford_n',y='Country Name', 
                    animation_frame=Data['Time'],color='CoHD', height=1200,
         )

fig.update_layout(yaxis={'categoryorder':'total ascending'})
fig.update_layout(
    title=dict(text='Millions of people who cannot afford a healthy diet throught years', font=dict(size=15), yref='paper')
)
st.plotly_chart(fig)


# In[223]:


# form a dataframe where Total millions of people who cannot afford a healthy diet is calculated

Total_2017 = Data.loc[Data['Time'] == 2017, 'CoHD_unafford_n'].sum()  # get sum of all CoHD_unafford_n values in 2017
Total_2018 = Data.loc[Data['Time'] == 2018, 'CoHD_unafford_n'].sum() # get sum of all CoHD_unafford_n values in 2018
Total_2019 = Data.loc[Data['Time'] == 2019, 'CoHD_unafford_n'].sum() # get sum of all CoHD_unafford_n values in 2019
Total_2020 = Data.loc[Data['Time'] == 2020, 'CoHD_unafford_n'].sum() # get sum of all CoHD_unafford_n values in 2020

A = [2017,2018,2019,2020]
B = [Total_2017,Total_2018,Total_2019,Total_2020]

Big_insight = pd.DataFrame(
    {'Year': A,
     'Total World Milllions who cannot cannot afford a healthy diet throught years': B 
    })



st.dataframe(Big_insight)


# # insights: 
#     
# The number of people who cannot afford a healthy diet has been decreasing until 2020, when it started to increase, possibly due to the Corona pandemic.
# 
# # Recommendations:
# 
# - The world must cooperate to reduce this percentage by activating the role of organizations and following a policy of fair distribution of wealth.

# # Millions of people who cannot afford nutrient adequacy

# In[106]:


result = Data[['Country Name','Country Code','Time','CoNA_unafford_n']]

data_slider = []  # slider for time

# passing over the years
for year in result['Time'].unique():
    df_segmented =  result[(result['Time']== year)]

    for col in df_segmented.columns:
        df_segmented[col] = df_segmented[col].astype(str)
        
        # Plot the map
        data_each_yr = dict(type='choropleth',
                        locations = df_segmented['Country Code'], 
                        z = df_segmented['CoNA_unafford_n'].astype(float), 
                        text = df_segmented['Country Name'],
                           )

    data_slider.append(data_each_yr)
    
steps = []
for i in range(len(data_slider)):
    step = dict(method='restyle',
                args=['visible', [False] * len(data_slider)],
                label='Year {}'.format(i + 2017))
    step['args'][1][i] = True
    steps.append(step)

sliders = [dict(active=0, pad={"t": 1}, steps=steps)]
        
layout = dict(title = 'Millions of people who cannot afford nutrient adequacy',
              geo = dict( projection = {'type':'hammer'}, #choosing projection type
                         showlakes = True, 
                         lakecolor = 'rgb(0,191,255)'),sliders=sliders)

# plot the figure

x = pg.Figure(data = data_slider, 
              layout = layout)
#po.iplot(x)

st.plotly_chart(x)


# In[107]:


# Form a subset of dataframe

result1 = Data[['Country Name','Country Code','Time','Pop']]
#result1


# In[108]:


data_slider = [] 
for year in result1['Time'].unique():
    df_segmented =  result1[(result1['Time']== year)]

    for col in df_segmented.columns:
        df_segmented[col] = df_segmented[col].astype(str)
        
        #plot map
        data_each_yr = dict(type='choropleth',
                        locations = df_segmented['Country Code'], 
                        z = df_segmented['Pop'].astype(float), 
                        text = df_segmented['Country Name'],
                           )

    data_slider.append(data_each_yr)
    
steps = []
for i in range(len(data_slider)):
    step = dict(method='restyle',
                args=['visible', [False] * len(data_slider)],
                label='Year {}'.format(i + 2017))
    step['args'][1][i] = True
    steps.append(step)

sliders = [dict(active=0, pad={"t": 1}, steps=steps)]
        
layout = dict(title = 'Population',
              geo = dict( projection = {'type':'equirectangular'},
                         showlakes = True, 
                         lakecolor = 'rgb(0,191,255)'),sliders=sliders)
x = pg.Figure(data = data_slider, 
              layout = layout)
#po.iplot(x)

st.plotly_chart(x)


# In[214]:


# plot scatter plot

fig = px.scatter_geo(result1, locations=result1['Country Code'],
                     hover_name=result1['Country Name'], size=result1['Pop'],
                     animation_frame=result1['Time'],
                     projection="natural earth")
 
fig.update_layout(autosize = True, geo = dict(projection_scale = 6))  #update the figure 

fig.update_layout(
    title=dict(text='population in the countries', font=dict(size=25), yref='paper') #update the figure 
)

#fig.show()

st.plotly_chart(fig)


# In[110]:


# Form a subset of dataframe

result2 =  Data[['Country Name','Country Code','Time','CoHD_unafford_n']] 
#result2


# In[213]:


# Scatter plot 

result2["Colors"] = "red"

fig = px.scatter_geo(result2, locations=result2['Country Code'],
                     hover_name=result2['Country Name'], size=result2['CoHD_unafford_n'],
                     animation_frame=result2['Time'],color = 'Colors',color_discrete_map={"red": "#EF553B"},
                     projection="natural earth")
fig.update_layout(
    title=dict(text='Millions of people who cannot afford a healthy diet', font=dict(size=25), yref='paper')
)

#fig.show()
st.plotly_chart(fig)


# In[212]:


# Form a subset of dataframe

result3 = Data[['Country Name','Country Code','Time','CoCA_unafford_n']]

# for choosing color
result3["Colors"] = "purple"

fig = px.scatter_geo(result3, locations=result3['Country Code'],
                     hover_name=result3['Country Name'], size=result3['CoCA_unafford_n'],
                     animation_frame=result3['Time'],color = 'Colors',color_discrete_map={"purple": "#A020F0"},
                     projection="natural earth")

fig.update_layout(
    title=dict(text='Millions of people who cannot afford sufficient calories', font=dict(size=25), yref='paper')
)

#fig.show()
st.plotly_chart(fig)


# In[211]:


# Form a subset of dataframe

result4 = Data[['Country Name','Country Code','Time','CoHD_headcount']]

result4["Colors"] = "purple"

fig = px.scatter_geo(result4, locations=result4['Country Code'],
                     hover_name=result4['Country Name'], size=result4['CoHD_headcount'],
                     animation_frame=result4['Time'],color = 'Colors',color_discrete_map={"purple": "#A020F0"},
                     projection="natural earth")

fig.update_layout(
    title=dict(text='Percent of the population who cannot afford a healthy diet', font=dict(size=25), yref='paper')
)


#fig.show()
st.plotly_chart(fig)


# In[210]:


# Form a subset of dataframe

result5 = Data[['Country Name','Country Code','Time','CoNA_headcount']]

result5["Colors"] = "blue"

fig = px.scatter_geo(result5, locations=result5['Country Code'],
                     hover_name=result5['Country Name'], size=result5['CoNA_headcount'],
                     animation_frame=result5['Time'],color = 'Colors',color_discrete_map={"blue": "#0000FF"},
                     projection="natural earth")
fig.update_layout(
    title=dict(text='Percent of the population who cannot afford nutrient adequacy', font=dict(size=25), yref='paper')
)

#fig.show() 
st.plotly_chart(fig)


# In[209]:


# Form a subset of dataframe

result6 = Data[['Country Name','Country Code','Time','CoCA_headcount']]

result6["Colors"] = "green"
fig = px.scatter_geo(result6, locations=result6['Country Code'],
                     hover_name=result6['Country Name'], size=result6['CoCA_headcount'],
                     animation_frame=result6['Time'],color = 'Colors',color_discrete_map={"green": "#00FF00"},
                     projection="natural earth")
fig.update_layout(
    title=dict(text='Percent of the population who cannot afford sufficient calories', font=dict(size=25), yref='paper')
)

#fig.show()
st.plotly_chart(fig)


# In[208]:


# Form a subset of dataframe

result6 = Data[['Country Name','Country Code','Time','CoHD']]

result6["Colors"] = "Rosewood "
fig = px.scatter_geo(result6, locations=result6['Country Code'],
                     hover_name=result6['Country Name'], size=result6['CoHD'],
                     animation_frame=result6['Time'],color = 'Colors',color_discrete_map={"Rosewood ": "#65000b"},
                     projection="natural earth")

fig.update_layout(
    title=dict(text='Cost of a healthy diet in the studied countries', font=dict(size=25), yref='paper')
)

#fig.show()
st.plotly_chart(fig)



# In[195]:


# Form a subset of dataframe


result7 = Data[['Country Name','Country Code','Time','CoHD_pov']]

result7["Colors"] = "Black "
fig = px.scatter_geo(result7, locations=result7['Country Code'],
                     hover_name=result7['Country Name'], size=result7['CoHD_pov'],
                     animation_frame=result7['Time'],color = 'Colors',color_discrete_map={"Black ": "#000000"},
                     projection="natural earth")
fig.update_layout(
    title=dict(text='Affordability of a healthy diet: ratio of cost to the food poverty line', font=dict(size=25), yref='paper')
)

#fig.show()
st.plotly_chart(fig)


# In[194]:


# Form a subset of dataframe


Data = Data.iloc[: , :-1]

result8 = Data[['Country Name','Country Code','Time','CoHD_fexp']]

result8["Colors"] = "Rouge "
fig = px.scatter_geo(result8, locations=result8['Country Code'],
                     hover_name=result8['Country Name'], size=result8['CoHD_fexp'],
                    animation_frame=result8['Time'],color = 'Colors',color_discrete_map={"Rouge ": "#f00020"},
                   projection="natural earth")
fig.update_layout(
    title=dict(text='Affordability of a healthy diet: ratio of cost to food expenditures', font=dict(size=25), yref='paper')
)

#fig.show()
st.plotly_chart(fig)


# In[ ]:


#pip install streamlit==1.11.0  

# should be install this version to pass the problems


# In[160]:


datta = Data[Data['Time'] == 2020] # choose a specific year

CoHD_more_than_mean = (datta[datta['CoHD'] > datta['CoHD'].mean()])
sort_CoHD_more_than_mean = CoHD_more_than_mean.sort_values('CoHD',ascending=False) # order countries

fig, ax = plt.subplots(figsize=(60,60))
fig.tight_layout(pad=5)

# Creating a case-specific function to avoid code repetition
def plot_hor_bar(subplot, data):
    plt.subplot(1,2,subplot)
    ax = sns.barplot(y='Country Name', x='CoHD', data=data,
                     color='slateblue')
    plt.title('Countries Ranking where Cost of a healthy diet more than average', # assigne title
              fontsize=40)
    plt.xlabel('Cost', fontsize=40)
    plt.xticks(fontsize=40)
    plt.ylabel(None)
    plt.yticks(fontsize=40)
    sns.despine(left=True)
    ax.grid(False)
    ax.tick_params(bottom=True, left=False)
    return None

plot_hor_bar(2, sort_CoHD_more_than_mean) # plot
#plt.show()

st.pyplot(fig)


# # Insight:
#     
# - It seems very difficult to afford a healthy diet in Jamaica

# In[215]:


datta = Data[Data['Time'] == 2020]

CoHD_less_than_mean = (datta[datta['CoHD'] < datta['CoHD'].mean()])
sort_CoHD_less_than_mean = CoHD_less_than_mean.sort_values('CoHD',ascending=False)

fig, ax = plt.subplots(figsize=(60,60))
fig.tight_layout(pad=5)

# Creating a case-specific function to avoid code repetition
def plot_hor_bar(subplot, data):
    plt.subplot(1,2,subplot)
    ax = sns.barplot(y='Country Name', x='CoHD', data=data,
                     color='slateblue')
    plt.title('Countries Ranking where Cost of a healthy diet less than average',
              fontsize=50)
    plt.xlabel('Cost', fontsize=40)
    plt.xticks(fontsize=40)
    plt.ylabel(None)
    plt.yticks(fontsize=40)
    sns.despine(left=True)
    ax.grid(False)
    ax.tick_params(bottom=True, left=False)
    return None

plot_hor_bar(2, sort_CoHD_less_than_mean)
#plt.show()

st.pyplot(fig)


# # Insight:
# 
# - It seems very easy to afford a healthy diet in the UK

# In[193]:


# interactive bar chart

fig = px.bar(Data, x='CoHD',y='Country Name', 
                    animation_frame=Data['Time'],color='CoHD', height=1200,
         )

fig.update_layout(yaxis={'categoryorder':'total ascending'})
fig.update_layout(
    title=dict(text='Changing in Cost of a healthy diet throught time in the studied counties', font=dict(size=25), yref='paper')
)
#fig.show()
st.plotly_chart(fig)

# In[204]:


# plot bar chart

sns.set(font_scale=2)  # choose font scale

largest_five= result6.nlargest(8,'CoHD')
fig = plt.figure(figsize=(10, 4))
ax = sns.barplot(x="Country Name",y='CoHD', data=largest_five).set(title='The Highest countries in Cost of a healthy diet')
sns.set(rc={'figure.figsize':(11.7,8.27)})

st.pyplot(fig)





# In[206]:


# plot bar chart

sns.set(font_scale=2)

largest_five= result6.nsmallest(8,'CoHD')
fig = plt.figure(figsize=(10, 4))
ax = sns.barplot(x="Country Name",y='CoHD', data=largest_five).set(title='The Lowest countries in Cost of a healthy diet')
sns.set(rc={'figure.figsize':(11.7,8.27)})

st.pyplot(fig)


# In[ ]:




