import streamlit as st 
import pandas as pd
import plotly_express as px
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import base64


st.set_option('deprecation.showPyplotGlobalUse', False)

img = Image.open("data.jpg")
img1 = Image.open("graph3.jpg")

new_title = '<p style="font-family:monospace; color:Red; font-size: 42px;">Machine Learning</p>'
st.markdown(new_title, unsafe_allow_html=True)


home = st.sidebar.radio(
    label="Select",
    options=['Home', 'App'])
    
if home == 'Home':
    
    main_bg = "graph2.jpg"
    main_bg_ext = "jpg"

    st.write(
            f"""
            <style>
            .reportview-container {{
                background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
            }}

            </style>
            """,
            unsafe_allow_html=True
        )
    st.write("Machine learning is the concept that a computer program can learn and adapt to new data without human intervention. Machine learning is a field of artificial intelligence (AI) that keeps a computer’s built-in algorithms current regardless of changes in the worldwide economy.")
    
    st.image(img1,width=800)
    
    st.write("This is the simple application for uploading your data from the local machine and you can do some EDA and you can plot some plots")
else :

    # Add a sidebar
    st.sidebar.subheader("Settings")

    # Setup file upload
    uploaded_file = st.sidebar.file_uploader(
                            label="Upload your CSV or Excel file.",
                             type=['csv', 'xlsx'])

    global df

    if uploaded_file is not None:
        print(uploaded_file)
        print("hello")

        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            print(e)
            df = pd.read_excel(uploaded_file)
            
            
    st.write('Hi, *Guys!* :sunglasses:')

    st.sidebar.write("or")
    st.sidebar.write("Choose some default datasets")

    default_dataset = st.sidebar.selectbox(
        label = "select the dataset",
        options=['None', 'Campus Requirement Dataset', 'Carseat Dataset', 'Cereal Dataset', 'Insurance Dataset', 'Iris Dataset', 'Mtcars Dataset', 'Penguin Dataset', 'Pokemon Dataset','Students Dataset','Students Test Performance Dataset']
    )

    if default_dataset == 'Campus Requirement Dataset':
        try:
            df = pd.read_csv("C:/Users/Hariharan/Desktop/Project Dataset/Campus Requirement.csv")
        except Exception as e:
            print(e)   

    if default_dataset == 'Carseat Dataset':
        try:
            df = pd.read_csv("C:/Users/Hariharan/Desktop/Project Dataset/carseats.csv")
        except Exception as e:
            print(e)  

    if default_dataset == 'Cereal Dataset':
        try:
            df = pd.read_csv("C:/Users/Hariharan/Desktop/Project Dataset/cereal.csv")
        except Exception as e:
            print(e)  

    if default_dataset == 'Insurance Dataset':
        try:
            df = pd.read_csv("C:/Users/Hariharan/Desktop/Project Dataset/insurance.csv")
        except Exception as e:
            print(e)  

    if default_dataset == 'Iris Dataset':
        try:
            df = pd.read_csv("C:/Users/Hariharan/Desktop/Project Dataset/IRIS.csv")
        except Exception as e:
            print(e)         

    if default_dataset == 'Mtcars Dataset':
        try:
            df = pd.read_csv("C:/Users/Hariharan/Desktop/Project Dataset/mtcars.csv")
        except Exception as e:
            print(e)

    if default_dataset == 'Penguin Dataset':
        try:
            df = pd.read_csv("C:/Users/Hariharan/Desktop/Project Dataset/Penguins_data.csv")
        except Exception as e:
            print(e)

    if default_dataset == 'Pokemon Dataset':
        try:
            df = pd.read_csv("C:/Users/Hariharan/Desktop/Project Dataset/Pokemon.csv")
        except Exception as e:
            print(e)

    if default_dataset == 'Students Dataset':
        try:
            df = pd.read_csv("C:/Users/Hariharan/Desktop/Project Dataset/students.csv")
        except Exception as e:
            print(e)
            
    if default_dataset == 'Students Test Performance Dataset':
        try:
            df = pd.read_csv("C:/Users/Hariharan/Desktop/Project Dataset/studentsPerformance.csv")
        except Exception as e:
            print(e)

            

    st.header("Data")

    global numeric_columns
    global non_numeric_columns
    try:
        st.write(df)
        numeric_columns = list(df.select_dtypes(['int64','float64','int32','float32','int','float']).columns)
        non_numeric_columns = list(df.select_dtypes(['object','bool']).columns)
        non_numeric_columns.append(None)
        print(non_numeric_columns)
    except Exception as e:
        print(e)
        st.write("Please upload file to the application.")

    select = st.sidebar.selectbox(
    label = "Select",
    options = ['','EDA','Graphs']
    )

    if select == 'EDA':
        main_bg = "graph.jpg"
        main_bg_ext = "jpg"


        st.write(
            f"""
            <style>
            .reportview-container {{
                background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
            }}

            </style>
            """,
            unsafe_allow_html=True
        )
        
        st.header("EDA(Exploratory Data Analysis)")
        st.write("Choose what you want to see.")
        pre = st.selectbox(
        label = "Select",
        options = ['Show Head','Show Tail','Show Dimension','Show Columns','Select Columns to Show','Show Summary','Null Value Hadling'])
        st.subheader((pre))
        
        if pre == 'Show Head':
            try:
                head = df.head()
                st.write(head)
                st.write("This is the first 5 rows of your dataset")
            except Exception as e:
                print(e)
                
        if pre == 'Show Tail':
            try:
                Tail = df.tail()
                st.write(Tail)
                st.write("This is the last 5 rows of your dataset")
            except Exception as e:
                print(e)
                
        if pre == 'Show Dimension':
            try:
                st.write(df.shape)
                st.write("This is the dimension of your dataset.")
            except Exception as e:
                print(e)
        
        if pre == 'Show Columns':
            try:
                all_columns = df.columns.to_list()
                st.write(all_columns)
                st.write("This will show yo the column names in the dataset.")
            except Exception as e:
                print(e)
                
        if pre == 'Select Columns to Show':
            try:
                all_columns = df.columns.to_list()
                select_column = st.multiselect("Select Columns",all_columns)
                new_df = df[select_column]
                st.write(new_df)
            except Exception as e:
                print(e)
                
        if pre == 'Show Summary':
            try:
                st.write(df.describe())
                st.write("This will shows you the count,mean,standard deviation,minimum value,1st quartile,median,3rd quartile and maximum value.")              
            except Exception as e:
                print(e)
                
        if pre == 'Null Value Hadling':
            if st.checkbox('Contains null value or not'):
                null = df.isnull().any()
                st.write(null)
                st.write("If the dataset contains null value then it show true the we have to preprocess the data.")
                
            if st.checkbox('How many null values'):
                value = df.isnull().sum()
                st.write(value)
                st.write("It shows us what are all the column that contains how many null values.")
                
            if st.checkbox("Null Values Percentage"):
                st.write("It shows the null values percentage of each columns in the data.")
                null_percent_df = df.isnull().sum()*100/df.shape[0]
                st.write(null_percent_df)
                st.write("Now we have to preprocess the data that contains the null value.")
                
            if st.checkbox("Preprocessing"):       
                one = st.selectbox('Numeric Column', options=numeric_columns)
                df[one].fillna(df[one].astype(float).mean(skipna=True),inplace=True)
                two = st.selectbox("Object", options=non_numeric_columns)
                df[two].fillna(df[two].mode()[0],inplace=True)
                st.write("Now we complete our preprocessing")
                
            if st.checkbox("After Preprocessing"):
                value = df.isnull().sum()
                st.write(value)
                st.write("Before you see some values in 'How many null values' but now there is no null values.")

    elif select == 'Graphs':
        st.header("Graphs")
        
        st.markdown("Data visualization is the graphical representation of information and data. By using visual elements like charts,graphs, and maps, data visualization tools provide an accessible way to see and understand trends, outliers, and patterns in data.")
		
        st.image(img,width=800)
        
        
        st.write("Select the chart you want to plot")
        chart_select = st.selectbox(
        label = "Select the chart",
        options = ['Barchart','Scatterplots', 'Lineplots', 'Histogram', 'Boxplot', 'Violinplot', 'Piechart','Correlogram'])
        
        st.subheader((chart_select))
        
        main_bg = "graph1.jpg"
        main_bg_ext = "jpg"


        st.write(
            f"""
            <style>
            .reportview-container {{
                background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
            }}

            </style>
            """,
            unsafe_allow_html=True
        )

        
        if chart_select == 'Scatterplots':
            st.sidebar.subheader("Scatterplot Settings")
            try:
                x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
                y_values = st.sidebar.selectbox('Y axis', options=numeric_columns)
                color_value = st.sidebar.selectbox("Color", options=non_numeric_columns)
                plot = px.scatter(data_frame=df, x=x_values, y=y_values, color=color_value)
                # display the chart
                st.plotly_chart(plot)
                st.write("Generally Scatterplot helps to find the relation between the variables. The x axis and Y axis Of the graph is ",x_values," and ",y_values," respectively. So we can find the graph shows the releationship between ", x_values, "and", y_values, ". Each color of the graph represents the ",color_value,"'s.")
            except Exception as e:
                print(e)
            

        if chart_select == 'Lineplots':
            st.sidebar.subheader("Line Plot Settings")
            try:
                x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
                y_values = st.sidebar.selectbox('Y axis', options=numeric_columns)
                color_value = st.sidebar.selectbox("Color", options=non_numeric_columns)
                plot = px.line(data_frame=df, x=x_values, y=y_values, color=color_value)
                st.plotly_chart(plot)
                st.write("A lineplot is a graph that displays data as points or check marks above a number line, showing the frequency of each value. In the above graph we take  x axis and Y axis as is ",x_values," and ",y_values,". Each color of the graph represents the ",color_value,"'s. It shows the difference between each ",color_value,".")
            except Exception as e:
                print(e)

        if chart_select == 'Histogram':
            st.sidebar.subheader("Histogram Settings")
            try:
                x = st.sidebar.selectbox('Feature', options=numeric_columns)
                bin_size = st.sidebar.slider("Number of Bins", min_value=10,
                                             max_value=100, value=40)
                plot = px.histogram(x=x, data_frame=df, nbins = bin_size)
                st.plotly_chart(plot)
                st.write("A Histogram is a graphical representation of a grouped frequency distribution with continuous classes. Histogram is a univariate graph. In this graph we plotted a histogram for ",x,". In this we can find the frequency whether it is maximum frequency or minimum frequency in the ",x,". ")
            except Exception as e:
                print(e)

        if chart_select == 'Boxplot':
            st.sidebar.subheader("Boxplot Settings")
            try:
                y = st.sidebar.selectbox("Y axis", options=numeric_columns)
                x = st.sidebar.selectbox("X axis", options=non_numeric_columns)
                color_value = st.sidebar.selectbox("Color", options=non_numeric_columns)
                plot = px.box(data_frame=df, y=y, x=x, color=color_value)
                st.plotly_chart(plot)
                st.write("This Graph Is Box Plot. Which shows us the lower limit(min),quartile 1,quartile 2,quartile 3 and upper limit(max). Which helps to find how the data is spreaded.In this graph we can see that maximum data is spread between ",x," and ",y,".")
            except Exception as e:
                print(e)

        if chart_select == 'Violinplot':
            st.sidebar.subheader("Violinplot Settings")
            try:
                y = st.sidebar.selectbox("Y axis", options=numeric_columns)
                x = st.sidebar.selectbox("X axis", options=non_numeric_columns)
                color_value = st.sidebar.selectbox("Color", options=non_numeric_columns)
                plot = px.violin(data_frame=df, y=y, x=x, box=True, color=color_value)
                st.plotly_chart(plot)
                st.write(" Violin plots are similar to box plots, except that they also show the probability density of the data at different values, usually smoothed by a kernel density estimator. In this we plotted violin chart between ",x," and ",y,". The violin plot is plotted for ",y," with respect to ",x,". In this we find q1, q2, q3, min, max valuse and kde value.")
            except Exception as e:
                print(e)
                
        if chart_select == 'Piechart':
            st.sidebar.subheader("Piechart Settings")
            try:
                x = st.sidebar.selectbox('Feature', options=numeric_columns)
                color_value = st.sidebar.selectbox("Color", options=non_numeric_columns)
                plot = px.pie(values=x, data_frame=df, names=color_value)
                st.plotly_chart(plot)
                st.write("A pie chart (or a circle chart) is a circular statistical graphic, which is divided into slices to illustrate numerical proportion. In this graph we draw for ",x," with respect to the ",color_value,". It helps to understand how many percentage of ",x," in ",color_value," .")
            except Exception as e:
                print(e)

        if chart_select == 'Barchart':
            st.sidebar.subheader("Bar Chart Settings")
            try:
                y_values = st.sidebar.selectbox("Y axis", options=numeric_columns)
                x_values = st.sidebar.selectbox("X axis", options=non_numeric_columns)
                plot = px.bar(data_frame=df, y=y_values, x=x_values)
                st.plotly_chart(plot)
                st.write("Generally bar graph is helps to find the difference between categorical data. In this graph we can easily find the difference between ",x_values," about their ",y_values,". It helps easily  to understand that which ",x_values," have more ",y_values," and which ",x_values," have less ",y_values,".")
            except Exception as e:
                print(e)

        if chart_select == 'Correlogram':
            try:
                #st.write(sns.heatmap(df.corr(),annot=True))
                cor = sns.heatmap(df.corr(),annot=True)
                st.write(cor)
                st.pyplot()
            except Exception as e:
                print(e)