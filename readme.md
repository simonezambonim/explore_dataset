# Data exploring with Streamlit 

This app aims to help data scientists get their dataset first insights in a simple and fast way.
**Exploratory Data Analysis** is one of the most important steps in a data science project and it can be very time-consuming.
However, many times this work can be facilitated using a generalized code for EDA. 
This app helps to extract relevant variables from a dataset with hundreds of variables, treat/drop missing values, and also perform univariate and multivariate analysis through beautiful charts.    

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

The packages needed are in the requirements.txt file.

### Installing


Clone the repository and install all the packages necessary:

```
cd path
vintualenv venv
cd path\venv\Scripts\activate

pip install -r requirements.txt 
```

Use the following command to run the application:

```
streamlit run main_explore_dataset.py
```

## Deployment

You can create your own app on Heroku platform! It is easy and free!
Create your account and download the Heroku CLI.
Follow this *[tutorial](https://dev.to/hannahyan/getting-started-in-deploying-interactive-data-science-apps-with-streamlit-part-2-3ob).

You can access the app *[here](https://exploredataset.herokuapp.com/)

## Built With

* [Streamlit](https://docs.streamlit.io/api.html) - The web framework 
* [Plotly express](https://plotly.com/python/plotly-express/) - Interactive plots
* [Heroku](https://www.heroku.com/) - App host


## Author

* **Simone Rosana Zambonim**  - [Linkedin](https://www.linkedin.com/in/simonezambonim/) [Github](https://github.com/simonezambonim/)


## Acknowledgments

Using Streamlit you can create beautiful apps and it can also help data scientists translate your work to a business language.
