# Import necessary libraries 
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html

from sklearn import datasets
from PIL import Image
import pickle
import joblib

import utilities

# Set a titleback
st.title("Welcome to My Streamlit! 👋")

with st.sidebar:
	choose = option_menu("ML Classifiers", ["Home","Iris Flower"],
							icons=['house', 'flower1'],
							menu_icon="app-indicator", default_index=0,
							orientation="horizontal",
							styles={
		"container": { "orientation":"horizontal","padding": "5!important","background-color": "#3b5998"},
		"icon": {"color": "orange", "font-size": "25px"}, 
		"nav-link": {"font-size": "16px", "text-align": "left","orientation":"horizontal", "margin":"0px", "--hover-color": "#999"},
		"nav-link-selected": {"background-color": "#dfe3"},
		
	}
	) 

# Add text to the app 
if(choose=="Home"):
	html_temp = """
		<div style="background-color:#3b5998 ;padding:10px; border-radius:8px; margin:25px" >
		<h2 style="color:white;text-align:center;">Explore different classifiers</h2>
		</div>
		"""
	# Explore different classifiers
	st.markdown(html_temp, unsafe_allow_html=True)

	st.write("""
	Change the **classifer** or the **dataset** on the left to see how different models perform. 
	""")



	# Add a select box widget to the side 
	dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine"))

	classifier = st.sidebar.selectbox("Select Classifiers", ("KNN", "SVM", "Random Forest","GaussianMixture"))

	scaling = st.sidebar.checkbox("Scaling?")

	if st.checkbox('Show Raw Data'):
		st.subheader('raw data')
		dataa = utilities.get_dataset(dataset_name)
		# dataa = pd.DataFrame(utilities.get_dataset(dataset_name))
		# dataa.data
		st.write(dataa)

	# Get the data 
	X, y = utilities.get_dataset(dataset_name)
	st.write("Shape of the data:", X.shape)
	st.write("Number of Classes:", len(np.unique(y)))

	# Add parameters to the UI based on the classifier
	params = utilities.add_parameter_ui(classifier)

	# Get our classifier with the correct classifiers 
	clf = utilities.get_classifier(classifier, params)

	# Check if scaling is required 
	if scaling:
		X = utilities.scale_data(X)

	# Make predictions and get accuray 
	accuracy = utilities.classification(X, y, clf)
	st.write("**Classifer:** ", classifier)
	st.write("**Accuracy:** ", accuracy)

	# Plot the components of the data 

	graph = st.selectbox('select one',['scatter', 'plot'])
	if(graph=='scatter'):
		utilities.plot_data_Scatter(X, y)
	if(graph=='plot'):
		utilities.plot_data_plot(X, y)
	st.write("* ** More classifier will be added soon **")



# **********   second page ********

elif(choose=="Iris Flower"):

	
	setosa= Image.open('setosa.png')
	versicolor= Image.open('versicolor.png')
	virginica = Image.open('virginica.png')


	lin_model=pickle.load(open('lin_model.pkl','rb'))
	log_model=pickle.load(open('log_model.pkl','rb'))
	svm=pickle.load(open('svc_model.pkl','rb'))
	RandomForest=pickle.load(open('rF_model.pkl','rb'))

	
	def classify(num):
		if num<0.5:
			return 'Setosa' and st.image(setosa)
		elif num <1.5:
			return 'Versicolor' and st.image(versicolor)
		else:
			return 'Virginica' and st.image(virginica)

		# <div style="background-color:teal ;padding:10px">
	def main():
		html_temp = """
		<div style="background-color:#3b5998 ;padding:10px; border-radius:8px" >
		<h2 style="color:white;text-align:center;">Iris Classification</h2>
		</div>
		"""
		st.markdown(html_temp, unsafe_allow_html=True)
		activities=['Linear Regression','Logistic Regression','SVM','Random Forest']
		option=st.sidebar.selectbox('Which model would you like to use?',activities)
		st.subheader(option)
		sl=st.slider('Select Sepal Length', 0.0, 10.0)
		sw=st.slider('Select Sepal Width', 0.0, 10.0)
		pl=st.slider('Select Petal Length', 0.0, 10.0)
		pw=st.slider('Select Petal Width', 0.0, 10.0)
		inputs=[[sl,sw,pl,pw]]

		
		if st.button('Classify'):
			if option=='Linear Regression':
				prediction = st.success(classify(lin_model.predict(inputs)))
				print(prediction)
				# if(prediction=='0'):
				# 	st.image(setosa)
			elif option=='Logistic Regression':
				st.success(classify(log_model.predict(inputs)))
			
			elif option=='Random Forest':
				st.success(classify(log_model.predict(inputs)))
			
			else:
				st.success(classify(svm.predict(inputs)))


	if __name__=='__main__':
		main()
	