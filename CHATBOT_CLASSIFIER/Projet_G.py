import streamlit as st
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

nltk.download('punkt')
nltk.download('stopwords')

# Load and preprocess the data
data = pd.read_csv('/Users/sakina/Downloads/medical.csv')
data['Aliment'] = data['Aliment'].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word.lower() not in stopwords.words('english')]))

# Feature extraction
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(data['Aliment'])
y_train = data['Nutriments']

# Train the classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Initialize session state
if 'user_inputs' not in st.session_state:
    st.session_state.user_inputs = []

# Streamlit UI
st.title("Friendly Classification Chatbot 🎯")

user_input = st.text_input("Enter your text:")
if st.button("Classify"):
    preprocessed_input = ' '.join([word for word in word_tokenize(user_input) if word.lower() not in stopwords.words('english')])
    transformed_input = vectorizer.transform([preprocessed_input])
    category = classifier.predict(transformed_input)
    st.write("Predicted category:", category[0])

    # Display user input and predicted category in a DataFrame
    st.session_state.user_inputs.append({'User Input': user_input, 'Predicted Category': category[0]})
    history_df = pd.DataFrame(st.session_state.user_inputs)
    st.write("User Input History:")
    st.dataframe(history_df)

# Add a page for visualizations
st.sidebar.title("Options")
page = st.sidebar.selectbox("Select a page", ["Chatbot 🐢", "Visualizations 🐞", "Dataset 🐬", "Recommend_Food 🐝"])

if page == "Visualizations 🐞":
    st.title("Data Visualizations 🐞")
    history_df = pd.DataFrame(st.session_state.user_inputs)


    if not history_df.empty:  # Check if history_df is empty
        # Example visualization: Count of predicted categories
        st.subheader("Count of Predicted Categories 🐚")
        sns.set(style='whitegrid')
        #plt.figure(10, 10)
        colors = sns.color_palette("pastel")

        category_counts = history_df['Predicted Category'].value_counts()
        sns.barplot(x=category_counts.index, y=category_counts.values, palette=colors)
        plt.xticks(rotation=45, ha="right", fontsize=12)
        plt.yticks(fontsize=10)
        plt.title("Nutriments Classification")
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.tight_layout()
        # Adding data labels on the bars
        for index, value in enumerate(category_counts):
            plt.text(index, value + 10, str(value), ha="center", fontsize=10)

        st.pyplot()

    else:
        st.warning("No data available for visualization 🐸")

if page == "Dataset 🐬":
    st.title("Dataset")
    data = pd.read_csv('/Users/sakina/Downloads/medical.csv')
    data

if page == "Recommend_Food 🐝":
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score
    from sklearn.tree import DecisionTreeClassifier
    import xgboost as xgb
    from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
    import random
    import streamlit as st


    # Generate random data for columns

    def generate_data(n):
        gender = ['M', 'F']
        social_levels = ['Middle', 'High', 'Low']
        jobs = ['Engineer', 'Doctor', 'Driver', 'Teacher', 'Retired', 'Developer']
        yes_no = ['Yes', 'No']
        diseases = ['None', 'Hypertension', 'Diabetes type 2', 'Asthma', 'Arthritis', 'Obesity', 'Respiratory problems',
                    'Stroke', 'Osteoporosis']
        risk_degree = ['Low', 'Medium', 'High']
        symptoms = ['Headache', 'Nausea', 'Stomachache', 'Shortness of breath', 'Blurred vision', 'Dizziness',
                    'Numbness', 'Asthenia', 'Weight loss', 'Chest pain']
        country = ['Benin', 'Burkina Faso', 'Ivory Cost', 'Gambia', 'Ghana', 'Guinea', 'Guinea-Bissau', 'Cape Verde',
                   'Liberia', 'Mali', 'Niger', 'Nigeria', 'Senegal', 'Sierra Leone', 'Togo']  # Pays afrique de l'Ouest
        duration_of_symptoms = ['2 hours', '5 hours', 'More than 5 hours']
        aliments = ['Apple', 'Banana', 'Carrot', 'Spinach', 'Chicken', 'Salmon', 'Rice', 'Bread', 'Cheese', 'Chocolate',
                    'Avocados', 'Bananas', 'Bluberries', 'Strawberries', 'Eggs', 'Lean beef', 'Almonds', 'Chia seeds',
                    'Coconuts', 'Cucumber', 'Garlic', 'Tomatoes']
        nutriments = ['Vitamin C', 'Potassium', 'Vitamin A', 'Iron', 'Protein', 'Omega-3', 'Carbohydrates', 'Fiber',
                      'Calcium', 'Sugar', 'Vitamin B6', 'Antioxidants', 'Manganese', 'Iron', 'vitamin E', 'Magnesium',
                      'Fatty acids', 'Vitamin K']

        data = {
            'Age': [random.randint(5, 75) for _ in range(n)],
            'Gender': [random.choice(gender) for _ in range(n)],
            'Social level': [random.choice(social_levels) for _ in range(n)],
            'Employment': [random.choice(jobs) for _ in range(n)],
            'Chronic illness': [random.choice(diseases) for _ in range(n)],
            'Sedentary lifestyle due to illness': [random.choice(yes_no) for _ in range(n)],
            'Allergies': [random.choice(yes_no) for _ in range(n)],
            'Potential diseases caused by a sedentary lifestyle': [random.choice(diseases) for _ in range(n)],
            'Degree of risk': [random.choice(risk_degree) for _ in range(n)],
            'country': [random.choice(country) for _ in range(n)],
            'duration': [random.choice(duration_of_symptoms) for _ in range(n)],
            'Price($)': [random.randint(1, 175) for _ in range(n)],
        }

        return data


    df = pd.DataFrame(generate_data(1000))

    # Specify Nutriments:
    aliment_nutriments = {
        'Apple': ['Vitamin C', 'Fiber'],
        'Banana': ['Potassium', 'Carbohydrates'],
        'Carrot': ['Vitamin A', 'Fiber'],
        'Spinach': ['Iron', 'Fiber'],
        'Chicken': ['Protein'],
        'Salmon': ['Protein', 'Omega-3'],
        'Rice': ['Carbohydrates'],
        'Bread': ['Carbohydrates'],
        'Cheese': ['Calcium', 'Protein'],
        'Chocolate': ['Sugar'],
        'Avocados': ['fiber', 'Potassium', 'Vitamin C'],
        'Bananas': ['Vitamin B6', 'Fiber'],
        'Bluberries': ['Antioxidants'],
        'Strawberries': ['Vitamin C', 'Fiber', 'Manganese'],
        'Eggs': ['Protein'],
        'Lean beef': ['Protein', 'Iron'],
        'Almonds': ['Vitamin E', 'Antioxidants', 'Magnesium', 'Fiber'],
        'Chia seeds': ['Fiber', 'Magnesium', 'Manganese', 'Calcium'],
        'Coconuts': ['Fiber', 'Fatty acids'],
        'Cucumber': ['Vitamin K'],
        'Garlic': ['Antioxidants'],
        'Tomatoes': ['Potassium', 'Vitamin C']
    }

    data_with_aliments = generate_data(1000)  # Générer les données pour les autres colonnes comme auparavant

    # Our Aliment List
    aliments = ['Apple', 'Banana', 'Carrot', 'Spinach', 'Chicken', 'Salmon', 'Rice', 'Bread', 'Cheese', 'Chocolate',
                'Avocados', 'Bananas', 'Bluberries', 'Strawberries', 'Eggs', 'Lean beef', 'Almonds', 'Chia seeds',
                'Coconuts', 'Cucumber', 'Garlic', 'Tomatoes']

    # Add Data in the new column
    data_with_aliments['Aliment'] = [random.choice(aliments) for _ in range(1000)]


    def assign_nutrients(aliment):
        return aliment_nutriments[aliment]


    # Use of for to attribute nutriments for each aliment
    data_with_aliments['Nutriments'] = [assign_nutrients(aliment) for aliment in data_with_aliments['Aliment']]


    # Verification
    def has_nutrient(nutriments_list, nutrient):
        return nutrient in nutriments_list


    # Final DataFrame
    df_with_aliments = pd.DataFrame(data_with_aliments)

    # print(data)
    data = df_with_aliments

    # Encoding

    from sklearn.preprocessing import LabelEncoder

    label_encoder = LabelEncoder()

    data['Gender'] = label_encoder.fit_transform(data['Gender'])
    data['Social level'] = label_encoder.fit_transform(data['Social level'])
    data['Employment'] = label_encoder.fit_transform(data['Employment'])
    data['Chronic illness'] = label_encoder.fit_transform(data['Chronic illness'])
    data['Sedentary lifestyle due to illness'] = label_encoder.fit_transform(data['Sedentary lifestyle due to illness'])
    data['Allergies'] = label_encoder.fit_transform(data['Allergies'])
    data['Potential diseases caused by a sedentary lifestyle'] = label_encoder.fit_transform(
        data['Potential diseases caused by a sedentary lifestyle'])
    data['Degree of risk'] = label_encoder.fit_transform(data['Degree of risk'])
    data['duration'] = label_encoder.fit_transform(data['duration'])
    data['country'] = label_encoder.fit_transform(data['country'])
    data['Aliment'] = label_encoder.fit_transform(data['Aliment'])

    from sklearn.preprocessing import MultiLabelBinarizer

    # Use MultiLabelBinarizer to encode Nutriiments
    mlb = MultiLabelBinarizer()
    nutriments_encoded = mlb.fit_transform(data['Nutriments'])
    nutriments_df = pd.DataFrame(nutriments_encoded, columns=mlb.classes_)

    # Add Nutriments to Columns
    data_ = pd.concat([data, nutriments_df], axis=1)

    # Delete "Nutriments" Column
    # df_with_aliments = data.drop(columns=['Nutriments'])

    data = data_

    # Supprimer la colonne "Nutriments" d'origine
    df_with_aliments = data_.drop(columns=['Nutriments'])
    data_ = df_with_aliments
    y = data_.iloc[:, 5]

    X = data_.iloc[:, :5].join(data_.iloc[:, 6:])

    # Streamlit UI
    st.title("Nutrition Recommendation Chatbot 🍪")

    # User input for classification

    Age = st.number_input("Age", min_value=5, max_value=75)

    gender = st.selectbox("Gender", data['Gender'].unique())

    social_level = st.selectbox("Social_level", data['Social level'].unique())

    employment = st.selectbox("Employment", data['Employment'].unique())

    chronic_illness = st.selectbox("Chronic illness", data['Chronic illness'].unique())

    sedentary_lifestyle = st.selectbox("Sedentary lifestyle due to illness",
                                       data['Sedentary lifestyle due to illness'].unique())

    allergies = st.selectbox("Allergies", data['Allergies'].unique())

    potential_diseases = st.selectbox("Potential diseases",
                                      data['Potential diseases caused by a sedentary lifestyle'].unique())

    risk_degree = st.selectbox("Degree of risk", data['Degree of risk'].unique())

    country = st.selectbox("Country", data['country'].unique())

    duration = st.selectbox("Symptoms duration", data['duration'].unique())

    price = st.number_input("Price ($)", min_value=1, max_value=175)

    if st.button("The good one"):
        # Prediction

        # Divide to train and Test
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score

        # Define Features and Target
        y = data_.iloc[:, 12]
        X = data_.iloc[:, :12].join(data_.iloc[:, 13:])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=4)

        # USE RandomForestClassifier model
        model = RandomForestClassifier()

        # Train our model
        model.fit(X_train, y_train)

        # Make Prediction
        predicted_aliment = model.predict(X_test)[0]
        st.write("Food Prediction 🍣:", predicted_aliment)
        if predicted_aliment == 17:
            st.write("Hey wonderful person, it's better for you to eat : ", "Rice 🍙")
        elif predicted_aliment == 21:
            st.write("Hey wonderful person, it's better for you to eat : ", "Tomatoes 🍅")
        elif predicted_aliment == 11:
            st.write("Hey wonderful person, it's better for you to eat : ", "Chocolate 🍫")
        elif predicted_aliment == 19:
            st.write("Hey wonderful person, it's better for you to eat : ", "Spinach 🍲")
        elif predicted_aliment == 8:
            st.write("Hey wonderful person, it's better for you to eat : ", "Cheese 🍮")
        elif predicted_aliment == 12:
            st.write("Hey wonderful person, it's better for you to eat : ", "Coconuts 🍉")
        elif predicted_aliment == 5:
            st.write("Hey wonderful person, it's better for you to eat : ", "Bluberries 🍒")
        elif predicted_aliment == 9:
            st.write("Hey wonderful person, it's better for you to eat : ", "Chia seeds 🍯")
        elif predicted_aliment == 10:
            st.write("Hey wonderful person, it's better for you to eat : ", "Chicken 🍗")
        elif predicted_aliment == 20:
            st.write("Hey wonderful person, it's better for you to eat : ", "Strawberries 🍓")
        elif predicted_aliment == 18:
            st.write("Hey wonderful person, it's better for you to eat : ", "Salmon 🍜")
        elif predicted_aliment == 6:
            st.write("Hey wonderful person, it's better for you to eat : ", "Bread 🍞")
        elif predicted_aliment == 2:
            st.write("Hey wonderful person, it's better for you to eat : ", "Avocados 🍈")
        elif predicted_aliment == 3:
            st.write("Hey wonderful person, it's better for you to eat : ", "Banana 🍌")
        elif predicted_aliment == 15:
            st.write("Hey wonderful person, it's better for you to eat : ", "Garlic 🍇")
        elif predicted_aliment == 4:
            st.write("Hey wonderful person, it's better for you to eat : ", "Bananas 🍌")
        elif predicted_aliment == 13:
            st.write("Hey wonderful person, it's better for you to eat : ", "Cucumber 🍱")
        elif predicted_aliment == 16:
            st.write("Hey wonderful person, it's better for you to eat : ", "Lean beef 🍖")
        elif predicted_aliment == 7:
            st.write("Hey wonderful person, it's better for you to eat : ", "Carrot 🍆")
        elif predicted_aliment == 0:
            st.write("Hey wonderful person, it's better for you to eat : ", "Almonds 🍡")
        elif predicted_aliment == 1:
            st.write("Hey wonderful person, it's better for you to eat : ", "Apple 🍏")
        elif predicted_aliment == 14:
            st.write("Hey wonderful person, it's better for you to eat : ", "Eggs 🍥")
