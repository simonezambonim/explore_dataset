import pandas as pd 
import plotly.express as px
# import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

px.defaults.width = 800
px.defaults.height = 500

# Exploratory Analysis Class
class EDA:

    def __init__(self, dataframe):
        self.df = dataframe
        self.columns = self.df.columns
        self.num_vars = self.df.select_dtypes(include=[np.number]).columns
        self.cat_vars = self.df.select_dtypes(include=[np.object]).columns

    def box_plot(self, main_var, col_x=None, hue=None):
        return px.box(self.df, x=col_x, y=main_var, color=hue)

    def violin(self, main_var, col_x=None, hue=None, split=False):
        sns.set(style="darkgrid")
        return sns.violinplot(x=col_x, y=main_var, hue=hue,
                    data=self.df, palette="husl", split=split)

    def swarmplot(self, main_var, col_x=None, hue=None, split=False):
        sns.set(style="darkgrid")
        return sns.swarmplot(x=col_x, y=main_var, hue=hue,
                    data=self.df, palette="husl", dodge=split)
    
    def histogram_num(self, main_var, hue=None, bins = None, ranger=None):
        return  px.histogram(self.df[self.df[main_var].between(left = ranger[0], right = ranger[1])], \
            x=main_var, nbins =bins , color=hue, marginal='violin') # marginal='rug'

    def scatter_plot(self, col_x,col_y,hue=None, size=None):
        return px.scatter(self.df, x=col_x, y=col_y, color=hue,size=size)

    def bar_plot(self, col_y, col_x, hue=None):
        return px.bar(self.df, x=col_x, y=col_y,color=hue)
        
    def line_plot(self, col_y,col_x,hue=None, group=None):
        return px.line(self.df, x=col_x, y=col_y,color=hue, line_group=group)

    def line_plot_pivot(self, col_y,col_x,hue=None, group=np.sum):
        if hue != None:
            data = self.df.groupby([col_x,hue])[col_y].agg(group).reset_index()
        else:
            data = self.df.groupby([col_x])[col_y].agg(group).reset_index()
        return px.line(data, x=col_x, y=col_y,color=hue)
    
    def CountPlot(self, main_var, hue=None):
        sns.set(style="darkgrid")
        sns.set(font_scale=0.6)
        chart = sns.countplot(x=main_var, data=self.df, hue=hue, palette='pastel')
        return chart.set_xticklabels(chart.get_xticklabels(), rotation=30)

    def heatmap_vars(self,cols, func = np.mean):
        sns.set(style="darkgrid")
        sns.set(font_scale=0.6)
        chart = sns.heatmap(self.df.pivot_table(index =cols[0], columns =cols[1],  values =cols[2], aggfunc=func, fill_value=0).dropna(axis=1), annot=True, annot_kws={"size": 7}, linewidths=.5)
        return chart.set_xticklabels(chart.get_xticklabels(), rotation=30)

    def Corr(self, cols=None, method = 'pearson'):
        sns.set(style="darkgrid")
        sns.set(font_scale=0.6)
        if len(cols) != 0:
            corr = self.df[cols].corr(method = method)
        else:
            corr = self.df.corr(method = method)
        chart = sns.heatmap(corr, annot=True, annot_kws={"size": 7}, linewidths=.5)
        chart.set_xticklabels(chart.get_xticklabels(), rotation=30)
        chart.set_yticklabels(chart.get_yticklabels(), rotation=30)
        return chart
   
    def DistPlot(self, main_var):
        sns.set(style="darkgrid")
        return sns.distplot(self.df[main_var], color='c', rug=True)

    def PairPlot(self, cols, hue=None):
        if len(cols) == 0:
            try:
                cols = (self.num_vars).remove(hue)
            except:
                cols = self.num_vars
        sns.set(style="darkgrid")
        return sns.pairplot(self.df, hue=hue, vars = cols, palette="husl", corner=True, diag_kind="kde")

