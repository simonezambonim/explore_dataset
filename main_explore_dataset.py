'''Help create a preprocessing file'''
from eda import *
import pandas as pd
import streamlit as st
import base64
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import unidecodedata
sns.set()



def get_data(file):
    
    read_cache_csv = st.cache(pd.read_csv, allow_output_mutation=True)
    df = read_cache_csv(file)

    def date_string_option(format_date):
        if format_date =="%d/%m/%Y":
            return 'day/month/year'
        if format_date =="%m/%d/%Y":
            return 'month/day/year'
        if format_date =="%Y %m %d %H:%M:%S":
            return 'year/month/day hour:minutes:seconds' 
    

    if st.sidebar.checkbox('Read options'):
        user_input_dt=False
        mydateparser = lambda x: ()
        encode = st.sidebar.radio("Choose encoding method", ("utf-8", 'ISO-8859-1','us-ascii'))
        delimiter = st.sidebar.radio("Choose delimiter", (',',';',".", ' ', "|"))
        decimal = st.sidebar.radio("Choose decimal format", (".", ','))
        df = read_cache_csv(file, encoding=encode, sep = delimiter, decimal =decimal)
        
        dt_checkbox = st.sidebar.checkbox("Data time format")
        if dt_checkbox:
            # user_input_dt = [st.sidebar.selectbox("Choose the datetime column", df.columns)]
            user_input_dt = [int(st.sidebar.number_input("Insert datetime column position", format='%i', value = 0))]
            date_format = st.sidebar.selectbox('Choose date format', \
                ("%d/%m/%Y", "%m/%d/%Y", "%Y %m %d %H:%M:%S"), format_func = date_string_option)
            mydateparser = lambda x: pd.datetime.strptime(x, date_format)
            try:
                df = read_cache_csv(file, encoding=encode, sep = delimiter, decimal =decimal, \
                            parse_dates=user_input_dt, date_parser=mydateparser)
            except ValueError:
                st.error("Choose another column for the date time format")
                df = read_cache_csv(file, encoding=encode, sep = delimiter, decimal =decimal)
            
    format_header =(lambda x: str(x).strip().lower().replace(' ', '_').replace('(', '').replace(')', ''))
    df.rename(format_header, axis='columns', inplace = True)

    return df

@st.cache
def get_stats(df):
    stats_num = df.describe()

    if df.select_dtypes(np.object).empty :
        return stats_num.transpose(), None
    if df.select_dtypes(np.number).empty :
        return None, df.describe(include=np.object).transpose()
    else:
        return stats_num.transpose(), df.describe(include=np.object).transpose()

@st.cache
def get_info(df):
    return pd.DataFrame({'types': df.dtypes,
                            'nan': df.isna().sum(), 
                            'nan%': round((df.isna().sum()/len(df))*100,2), 
                            'unique':df.nunique()
                            })

def drop_null(df, percentual, drop_radio):
    df_info = get_info(df)

    if drop_radio == ('Specific columns'):
        col_drop = st.sidebar.multiselect ("Choose the columns to drop",\
        (df_info[df_info['nan%']  > percentual].index))

        if st.sidebar.checkbox('All set! Drop selected columns!'):
            df_drop = df.drop(col_drop, axis=1).copy()
            st.success("Columns dropped!")
            st.write('raw # of columns ', df.shape[1], ' || preproc # of columns ', df_drop.shape[1])
           
    elif drop_radio == (f'Drop all above {percentual}% nan'): 
        df_drop = df.drop(df_info[df_info['nan%']  > percentual].index, axis=1).copy()
        st.success("Columns dropped!")
        st.write('raw # of columns ', df.shape[1], ' || preproc # of columns ', df_drop.shape[1])
    
    return df_drop

def input_null(df, col, radio):
    df_inp = df.copy()

    if radio == 'Mean':
        st.write("Mean:", df[col].mean())
        df_inp[col] = df[col].fillna(df[col].mean())
    
    elif radio == 'Median':
        st.write("Median:", df[col].median())
        df_inp[col] = df[col].fillna(df[col].median())

    elif radio == 'Mode':
        for i in col:
            st.write(f"Mode {i}:", df[i].mode()[0])
            df_inp[i] = df[i].fillna(df[i].mode()[0])
        
    elif radio == 'Repeat last valid value':
        df_inp[col] = df[col].fillna(method = 'ffill')

    elif radio == 'Repeat next valid value':
        df_inp[col] = df[col].fillna(method = 'bfill')

    elif radio == 'Value':
        for i in col:
            number = st.number_input(f'Insert a number to fill missing values in {i}', format='%f', key=i)
            df_inp[i] = df[i].fillna(number)
    
    elif radio == 'Drop rows with missing values':
        if type(col) != list:
            col = [col]
        df_inp = df.dropna(axis=0, subset=col)
        st.markdown("Rows dropped!")
        st.write('raw # of rows ', df.shape[0], ' || preproc # of rows ', df_inp.shape[0])

    st.table(get_na_info(df_inp, df, col)) 
    
    return df_inp

def input_null_cat(df, col, radio):
    df_inp = df.copy()

    if radio == 'Text':
        for i in col:
            user_text = st.text_input(f'Replace missing values in {i} with', key=i)
            df_inp[i] = df[i].fillna(user_text)
    
    elif radio == 'Drop rows with missing values':
        if type(col) != list:
            col = [col]
        df_inp = df.dropna(axis=0, subset=col)
        st.markdown("Rows dropped!")
        st.write('raw # of rows ', df.shape[0], ' || preproc # of rows ', df_inp.shape[0])

    st.table(pd.concat([get_info(df[col]),get_info(df_inp[col])], axis=0))
    
    return df_inp

@st.cache
def get_na_info(df_preproc, df, col):
    raw_info = pd_of_stats(df, col)
    prep_info = pd_of_stats(df_preproc,col)
    return raw_info.join(prep_info, lsuffix= '_raw', rsuffix='_prep').T

@st.cache     
def pd_of_stats(df,col):
    #Descriptive Statistics
    stats = dict()
    stats['Mean']  = df[col].mean()
    stats['Std']   = df[col].std()
    stats['Var'] = df[col].var()
    stats['Kurtosis'] = df[col].kurtosis()
    stats['Skewness'] = df[col].skew()
    stats['CoefVar'] = stats['Std'] / stats['Mean']
    
    return pd.DataFrame(stats, index = col).T.round(2)

@st.cache   
def pf_of_info(df,col):
    info = dict()
    info['Type'] =  df[col].dtypes
    info['Unique'] = df[col].nunique()
    info['n_zeros'] = (len(df) - np.count_nonzero(df[col]))
    info['p_zeros'] = round(info['n_zeros'] * 100 / len(df),2)
    info['nan'] = df[col].isna().sum()
    info['p_nan'] =  (df[col].isna().sum() / df.shape[0]) * 100
    return pd.DataFrame(info, index = col).T.round(2)

@st.cache     
def pd_of_stats_quantile(df,col):
    #Quantile Statistics
    df_no_na = df[col].dropna()
    stats_q = dict()

    stats_q['Min'] = df[col].min()
    label = {0.25:"Q1", 0.5:'Median', 0.75:"Q3"}
    for percentile in np.array([0.25, 0.5, 0.75]):
        stats_q[label[percentile]] = df_no_na.quantile(percentile)
    stats_q['Max'] = df[col].max()
    stats_q['Range'] = stats_q['Max']-stats_q['Min']
    stats_q['IQR'] = stats_q['Q3']-stats_q['Q1']

    return pd.DataFrame(stats_q, index = col).T.round(2)    

@st.cache
def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'
    return href

    # def remove_accents_strip_lower(x):
    #     nfkd = unicodedata.normalize('NFKD', x)
    #     words = u"".join([c for c in nfkd if not unicodedata.combining(c)])

    #     return words.strip().lower().replace(' ', '_').replace('(', '').replace(')', '')

def plot_univariate(obj_plot, main_var, radio_plot_uni):
    
    if radio_plot_uni == 'Histogram' :
        st.subheader('Histogram')
        bins, range_ = None, None
        hue_opt = st.sidebar.selectbox("Hue (categorical) optional",obj_plot.columns.insert(0,None))
        bins_ = st.sidebar.slider('Number of bins optional', value = 50)
        range_ = st.sidebar.slider('Choose range optional', float(obj_plot.df[main_var].min()), \
            float(obj_plot.df[main_var].max()),(float(obj_plot.df[main_var].min()),float(obj_plot.df[main_var].max())))    
        if st.sidebar.button('Plot histogram chart'):
            st.plotly_chart(obj_plot.histogram_num(main_var, hue_opt, bins_, range_))
    
    if radio_plot_uni ==('Distribution Plot'):
        st.subheader('Distribution Plot')
        if st.sidebar.button('Plot distribution'):
            fig = obj_plot.DistPlot(main_var)
            st.pyplot()  

    if radio_plot_uni == 'BoxPlot' :
        st.subheader('Boxplot')
        # col_x, hue_opt = None, None
        col_x  = st.sidebar.selectbox("Choose x variable (categorical) optional", obj_plot.columns.insert(0,None), key ='boxplot')
        hue_opt = st.sidebar.selectbox("Hue (categorical) optional", obj_plot.columns.insert(0,None), key ='boxplot')
        if st.sidebar.button('Plot boxplot chart'):
            st.plotly_chart(obj_plot.box_plot(main_var,col_x, hue_opt))

def plot_multivariate(obj_plot, radio_plot):

    if radio_plot == ('Boxplot'):
        st.subheader('Boxplot')
        col_y  = st.sidebar.selectbox("Choose main variable (numerical)",obj_plot.num_vars, key ='boxplot')
        col_x  = st.sidebar.selectbox("Choose x variable (categorical) optional", obj_plot.columns.insert(0,None), key ='boxplot')
        hue_opt = st.sidebar.selectbox("Hue (categorical) optional", obj_plot.columns.insert(0,None), key ='boxplot')
        if st.sidebar.button('Plot boxplot chart'):
            st.plotly_chart(obj_plot.box_plot(col_y,col_x, hue_opt))
    
    if radio_plot == ('Violin'):
        st.subheader('Violin')
        col_y  = st.sidebar.selectbox("Choose main variable (numerical)",obj_plot.num_vars, key='violin')
        col_x  = st.sidebar.selectbox("Choose x variable (categorical) optional", obj_plot.columns.insert(0,None),key='violin')
        hue_opt = st.sidebar.selectbox("Hue (categorical) optional", obj_plot.columns.insert(0,None),key='violin')
        split = st.sidebar.checkbox("Split",key='violin')
        if st.sidebar.button('Plot violin chart'):
            fig = obj_plot.violin(col_y,col_x, hue_opt, split)
            st.pyplot()
    
    if radio_plot == ('Swarmplot'):
        st.subheader('Swarmplot')
        col_y = st.sidebar.selectbox("Choose main variable (numerical)",obj_plot.num_vars, key='swarmplot')
        col_x = st.sidebar.selectbox("Choose x variable (categorical) optional", obj_plot.columns.insert(0,None),key='swarmplot')
        hue_opt = st.sidebar.selectbox("Hue (categorical) optional", obj_plot.columns.insert(0,None),key='swarmplot')
        split = st.sidebar.checkbox("Split", key ='swarmplot')
        if st.sidebar.button('Plot swarmplot chart'):
            fig = obj_plot.swarmplot(col_y,col_x, hue_opt, split)
            st.pyplot()

    def pretty(method):
        return method.capitalize()

    if radio_plot == ('Correlation'):
        st.subheader('Heatmap Correlation Plot')
        correlation = st.sidebar.selectbox("Choose the correlation method", ('pearson', 'kendall','spearman'), format_func=pretty)
        cols_list = st.sidebar.multiselect("Select columns",obj_plot.columns)
        st.sidebar.markdown("If None selected, it will plot the correlation of all numeric variables.")
        if st.sidebar.button('Plot heatmap chart'):
            fig = obj_plot.Corr(cols_list, correlation)
            st.pyplot()

    def map_func(function):
        dic = {np.mean:'Mean', np.sum:'Sum', np.median:'Median'}
        return dic[function]
    
    if radio_plot == ('Heatmap'):
        st.subheader('Heatmap between vars')
        st.markdown(" In order to plot this chart remember that the order of the selection matters, \
            chooose in order the variables that will build the pivot table: row, column and value.")
        cols_list = st.sidebar.multiselect("Select 3 variables (2 categorical and 1 numeric)",obj_plot.columns, key= 'heatmapvars')
        agg_func = st.sidebar.selectbox("Choose one function to aggregate the data", (np.mean, np.sum, np.median), format_func=map_func)
        if st.sidebar.button('Plot heatmap between vars'):
            fig = obj_plot.heatmap_vars(cols_list, agg_func)
            st.pyplot()
    
    if radio_plot == ('Histogram'):
        st.subheader('Histogram')
        col_hist = st.sidebar.selectbox("Choose main variable", obj_plot.num_vars, key = 'hist')
        hue_opt = st.sidebar.selectbox("Hue (categorical) optional",obj_plot.columns.insert(0,None), key = 'hist')
        bins_, range_ = None, None
        bins_ = st.sidebar.slider('Number of bins optional', value = 30)
        range_ = st.sidebar.slider('Choose range optional', int(obj_plot.df[col_hist].min()), int(obj_plot.df[col_hist].max()),\
                (int(obj_plot.df[col_hist].min()),int(obj_plot.df[col_hist].max())))    
        if st.sidebar.button('Plot histogram chart'):
                st.plotly_chart(obj_plot.histogram_num(col_hist, hue_opt, bins_, range_))

    if radio_plot == ('Scatterplot'): 
        st.subheader('Scatter plot')
        col_x = st.sidebar.selectbox("Choose x variable (numerical)", obj_plot.num_vars, key = 'scatter')
        col_y = st.sidebar.selectbox("Choose y variable (numerical)", obj_plot.num_vars, key = 'scatter')
        hue_opt = st.sidebar.selectbox("Hue (categorical) optional", obj_plot.columns.insert(0,None), key = 'scatter')
        size_opt = st.sidebar.selectbox("Size (numerical) optional",obj_plot.columns.insert(0,None), key = 'scatter')
        if st.sidebar.button('Plot scatter chart'):
            st.plotly_chart(obj_plot.scatter_plot(col_x,col_y, hue_opt, size_opt))

    if radio_plot == ('Countplot'):
        st.subheader('Count Plot')
        col_count_plot = st.sidebar.selectbox("Choose main variable",obj_plot.columns, key = 'countplot')
        hue_opt = st.sidebar.selectbox("Hue (categorical) optional",obj_plot.columns.insert(0,None), key = 'countplot')
        if st.sidebar.button('Plot Countplot'):
            fig = obj_plot.CountPlot(col_count_plot, hue_opt)
            st.pyplot()

    if radio_plot == ('Pairplot'):
        st.subheader('Pairplot')
        cols_list = st.sidebar.multiselect("Select columns",obj_plot.num_vars, key = 'pairplot')
        st.sidebar.markdown("If None selected, it will plot the correlation of all numeric variables.")
        hue_opt = st.sidebar.selectbox("Hue (categorical) optional",obj_plot.columns.insert(0,None), key = 'pairplot')
        if st.sidebar.button('Plot Pairplot'):
            fig = obj_plot.PairPlot(cols_list, hue_opt)
            st.pyplot()

    if radio_plot == ('Barplot'):
        st.subheader('Barplot') 
        col_y = st.sidebar.selectbox("Choose main variable (numerical)",obj_plot.num_vars, key='barplot')
        col_x = st.sidebar.selectbox("Choose x variable (categorical)", obj_plot.columns,key='barplot')
        hue_opt = st.sidebar.selectbox("Hue (categorical/numerical) optional", obj_plot.columns.insert(0,None),key='barplot')
        if st.sidebar.button('Plot barplot chart'):
            st.plotly_chart(obj_plot.bar_plot(col_y,col_x, hue_opt))

    if radio_plot == ('Lineplot'):
        st.subheader('Lineplot') 
        col_y = st.sidebar.selectbox("Choose main variable (numerical)",obj_plot.num_vars, key='lineplot')
        col_x = st.sidebar.selectbox("Choose x variable (categorical)", obj_plot.columns,key='lineplot')
        hue_opt = st.sidebar.selectbox("Hue (categorical) optional", obj_plot.columns.insert(0,None),key='lineplot')
        group = st.sidebar.selectbox("Group color (categorical) optional", obj_plot.columns.insert(0,None),key='lineplot')
        if st.sidebar.button('Plot lineplot chart'):
            st.plotly_chart(obj_plot.line_plot(col_y,col_x, hue_opt, group))
    
    if radio_plot == ('Lineplot_pivot'):
        st.subheader('Lineplot') 
        col_y = st.sidebar.selectbox("Choose main variable (numerical)",obj_plot.num_vars, key='lineplot_p')
        col_x = st.sidebar.selectbox("Choose x variable (categorical)", obj_plot.columns,key='lineplot_p')
        hue_opt = st.sidebar.selectbox("Hue (categorical) optional", obj_plot.columns.insert(0,None),key='lineplot_p')
        agg_func = st.sidebar.selectbox("Choose one function to aggregate the data", (np.mean, np.sum, np.median), format_func=map_func, key ='lineplot_p')
        if st.sidebar.button('Plot lineplot chart'):
            st.plotly_chart(obj_plot.line_plot_pivot(col_y,col_x, hue_opt, agg_func))

def main():

    st.title('Preprocessing data :wrench:')
    st.header('Upload your file')  

    file  = st.file_uploader('Upload your file (.csv)', type = 'csv')

 
    if file is not None:
        
        df = get_data(file)

        numeric_features = df.select_dtypes(include=[np.number]).columns
        categorical_features = df.select_dtypes(include=[np.object]).columns

        def basic_info(df):
            st.header("Summary")
            st.write('Number of observations', df.shape[0], '|| Number of variables', df.shape[1]) 
            st.write('Number of missing (%)',((df.isna().sum().sum()/df.size)*100).round(2))
            aux_ = pd.DataFrame(df.dtypes, columns=['type'])
            for tp in aux_['type'].unique():
                    st.write(f'Number of {tp}:', (aux_[aux_['type'] == tp]).count()[0], tuple(aux_[aux_['type'] == tp].index))

        #Visualize data
        view_data = st.checkbox("View data")
        if view_data:
            basic_info(df)
            ndisplay = st.slider('Choose the number of rows to display',  min_value=1, max_value=100, value=5)
            st.dataframe(df.head(ndisplay).style.highlight_null(null_color='#d5d9e2'))

        #Sidebar Menu
        options = ["View statistics","Visualize partial data", "Treat or drop missing values",\
            "Univariate analysis", "Multivariate analysis"]
        menu = st.sidebar.selectbox("Menu options", options)

        #Data statistics
        df_info = get_info(df)   
        if (menu == "View statistics"):
            df_stat_num, df_stat_obj = get_stats(df)
            st.markdown('**Numerical summary**')
            st.table(df_stat_num)
            st.markdown('**Object summary**')
            st.table(df_stat_obj)
            st.markdown('**Missing Values**') # Data types and  Columns with N/A values
            st.table(df_info)
            st.bar_chart(df_info['nan%'].sort_values(ascending=False))

        # Analyze specific data
        if (menu =="Visualize partial data"):
            st.header("Analyse partial data")
            st.markdown("This section is intended to be used if your dataset has \
                too many columns and you wish to select just the meaningful ones.")
            an_cols = st.multiselect("Enter the variables you want to see", df.columns)

            if len(an_cols)!=0:
                sp_rows = st.checkbox("Do you want to filter your rows by a special feature?")
                if sp_rows:
                    remaining_variables = [x for x in df.columns if x not in an_cols]
                    sp_col = st.selectbox("Select a specific categorical variable to analyse", \
                        remaining_variables)
                    un_col = st.multiselect("Enter the specific values", df[sp_col].unique())
                    allcolumns = an_cols.append(sp_col)
                    selected_sp_data = df[(df[sp_col].isin(un_col))][an_cols]
                if not sp_rows:
                    selected_sp_data = df[an_cols]
                # Visualize    
                ndisplay_ = st.slider('Choose number of rows to display', min_value=1, max_value=len(df), value=5)
                st.dataframe(selected_sp_data.head(ndisplay_))
                # Data statistics
                stats_ = st.sidebar.checkbox("View related statistics")
                if stats_:
                    df_stat_num_p, df_stat_obj_p =get_stats(selected_sp_data)
                    st.markdown('**Numerical summary**')
                    st.table(df_stat_num_p.round(1))
                    st.markdown('**Object summary**')
                    st.table(df_stat_obj_p.round(1))
                # Download selected data
                download_data = st.sidebar.checkbox("Download selected data!")
                if download_data:
                    st.markdown('**Download new file below** :open_file_folder:')
                    with st.spinner('Working on it...'):
                        st.markdown(get_table_download_link(selected_sp_data), unsafe_allow_html=True)
                    st.warning("Remember: If you want to save these changes, download the file and upload it again.")

        # Treat missing values"
        if (menu =="Treat or drop missing values"):
            st.header('Treat or drop missing values')

            treat_drop = st.sidebar.selectbox('What do you want to do?', ('Treat variables', 'Drop variables'))
            if treat_drop == ('Drop variables'):
                st.subheader('Drop columns with missing values')

                st.warning("Important: this function drops a set of variables at once.\
                    To drop columns through different methods, \
                    please download the csv file after modifying it and \
                    reload this new file at the beginning of this page and apply further changes. ")
            
                percentual = st.sidebar.slider('Maximum nan (%) allowed', min_value=0, value=30, max_value=100)
                st.sidebar.subheader(f'Drop rows with missing values above {percentual}%:')
                drop_radio = st.sidebar.radio("Drop method", ('None',"Specific columns", f'Drop all above {percentual}% nan'), key='drop')

                try:
                    df_dropped = drop_null(df, percentual, drop_radio)
                    
                    if len(df.columns)> len(df_dropped.columns):
                        st.markdown('**Download new file below** :open_file_folder:')
                        with st.spinner('Working on it...'):
                            st.markdown(get_table_download_link(df_dropped), unsafe_allow_html=True)
                        st.warning("Remember: If you want to save these changes, download the file and upload it again.")
                        df_info = get_info(df_dropped)
            
                        if len (df_info[df_info['nan%']>=0]) !=0:
                            st.subheader("Updated DataFrame")
                            st.markdown("Columns with **missing values**")
                            st.table(df_info[df_info['nan%']!=0])
                except:
                    st.error("Choose one method to drop the variables or select the variables to drop on the sidebar.")

            if treat_drop == ('Treat variables'):
                st.subheader('Treat columns with missing values')

                st.warning("Important: this function can only input values using the same method to distinct variables.\
                    To replace missing values through different methods, \
                    please download the csv file after modifying it and \
                    reload this new file at the beginning of this page and apply further changes. ")
            
                col = st.multiselect("Choose the columns to input values with the same method",\
                    (df_info[df_info['nan']!=0].index))
            
                if all(elem in numeric_features  for elem in col):
                    
                    radio = st.sidebar.radio("Choose how to fill *missing values*:", \
                        ('None','Value','Mean','Median','Mode', \
                            'Repeat last valid value', 'Repeat next valid value', \
                                'Drop rows with missing values'),key="inp_num")

                    df_inp = input_null(df, col, radio)

                elif all(elem in categorical_features  for elem in col):
                    radio = st.sidebar.radio("Choose how to fill *missing values*:", ('None','Text',\
                        'Drop rows with missing values'), key="inp_cat")
                    df_inp = input_null_cat(df, col, radio)

                else:
                    st.error("Choose variables of the same type!")

                if  st.sidebar.button("Apply changes"):    
                    df = (df_inp.copy())
                    st.success(f"Replacement done in {col}!" )
                    st.markdown('**Download new file below** :open_file_folder:')
                    with st.spinner('Working on it...'):
                        st.markdown(get_table_download_link(df), unsafe_allow_html=True)
                    st.warning("Remember: If you want to save these changes, download the file and upload it again.")
                    df_info = get_info(df)
                    
                    if len (df_info[df_info['nan']!=0]) !=0:
                        st.subheader("Updated DataFrame")
                        st.markdown("Columns with **missing values**")
                        st.table(df_info[df_info['nan']!=0])
                
                st.sidebar.info('After clicking on the button above the data will be modified \
                        and a link will be available to download the data with missing \
                        values replaced by the chosen method in the selected variable(s)!')
        
        eda_plot = EDA(df) 

        # Visualize data

        if (menu =="Univariate analysis" ):
            st.header("Univariate analysis")
            st.markdown("Provides summary statistics of only one variable in the raw dataset.")
            main_var = st.selectbox("Choose one variable to analyze:", df.columns.insert(0,None))

            if main_var in numeric_features:
                if main_var != None:
                    st.subheader("Variable info")
                    st.table(pf_of_info(df, [main_var]).T)
                    st.subheader("Descriptive Statistics")
                    st.table((pd_of_stats(df, [main_var])).T)
                    st.subheader("Quantile Statistics") 
                    st.table((pd_of_stats_quantile(df, [main_var])).T) 
                    
                    chart_univariate = st.sidebar.radio('Chart', ('None','Histogram', 'BoxPlot', 'Distribution Plot'))
                    
                    plot_univariate(eda_plot, main_var, chart_univariate)

            if main_var in categorical_features:
                st.table(df[main_var].describe(include = np.object))
                st.bar_chart(df[main_var].value_counts().to_frame())

            st.sidebar.subheader("Explore other categorical variables!")
            var = st.sidebar.selectbox("Check its unique values and its frequency:", df.columns.insert(0,None))
            if var !=None:
                aux_chart = df[var].value_counts(dropna=False).to_frame()
                # st.sidebar.bar_chart(aux_chart )
                data = st.sidebar.table(aux_chart.style.bar(color='#3d66af'))

        if (menu =="Multivariate analysis" ):
            st.header("Multivariate analysis")

            st.markdown('Here you can visualize your data by choosing one of the chart options available on the sidebar!')
               
            st.sidebar.subheader('Data visualization options')
            radio_plot = st.sidebar.radio('Choose plot style', ('Correlation', 'Boxplot', 'Violin', 'Swarmplot', 'Heatmap', 'Histogram', \
                'Scatterplot', 'Countplot', "Pairplot", 'Barplot', 'Lineplot','Lineplot_pivot' ))

            plot_multivariate(eda_plot, radio_plot)


        st.sidebar.title('About')
        st.sidebar.info('This app is a preprocessing data available in [] \n \
        It is mantained by [Simone](https://www.linkedin.com/in/simonezambonim/). Check this code at https://github.com/simonezambonim/explore_dataset')


if __name__ == '__main__':
    main()

