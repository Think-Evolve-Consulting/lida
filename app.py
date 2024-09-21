import streamlit as st
import pandas as pd
from datetime import datetime
import json
from PIL import Image
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set page configuration - This must be the very first Streamlit command
st.set_page_config(page_title="BPD Dashboard", layout="wide")

st.markdown("""
    <style>
    .stButton > button {
        width: 100% !important;
    }
    </style>
    """, unsafe_allow_html=True)
# Functions to handle different pages


def show_upload_page():
    st.title("Upload Page")

    if 'uploaded_reports' not in st.session_state:
        st.session_state.uploaded_reports = {}

    # Initial document setup with default dates
    initial_date = "01-Aug-2024"
    documents = [
        {"Report File": "GRN-Report", "Uploaded Date": initial_date},
        {"Report File": "Stock-Report", "Uploaded Date": initial_date},
        {"Report File": "Purchase-Report", "Uploaded Date": initial_date},
        {"Report File": "TECO-Report", "Uploaded Date": initial_date},
        {"Report File": "Reservation-Report", "Uploaded Date": initial_date},
    ]

    # Load previous document upload state if exists
    if 'upload_dates' in st.session_state:
        saved_documents = st.session_state.upload_dates
        saved_dict = {doc['Report File']: doc for doc in saved_documents}
        for doc in documents:
            if doc['Report File'] not in saved_dict:
                saved_documents.append(doc)
        documents = saved_documents

    # Create a DataFrame to display the table
    df = pd.DataFrame(documents)

    # Function to handle report file uploads
    def upload_document(index):
        st.markdown(f"**Browse {df.iloc[index]['Report File']}:**")
        uploaded_file = st.file_uploader(
            "", type=["csv", "json", "xlsx"], key=index)
        if uploaded_file is not None:
            # Load uploaded file directly into session state
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                data = pd.read_json(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                data = pd.read_excel(uploaded_file)  # Handle Excel files

            # Update the upload date to the current date
            current_date = datetime.now().strftime('%d-%b-%Y')
            df.at[index, 'Uploaded Date'] = current_date
            documents[index]['Uploaded Date'] = current_date

            # Store the DataFrame in session state instead of the file path
            st.session_state.uploaded_reports[df.iloc[index]
                                              ['Report File']] = data
            st.success(f"{uploaded_file.name} uploaded successfully!")

            # Save the upload dates to session state
            st.session_state.upload_dates = documents

    st.write("## Document Upload Table")
    for i, row in df.iterrows():
        col1, col2, col3 = st.columns([3, 2, 2])
        col1.write(row['Report File'])
        col2.write(row['Uploaded Date'])
        with col3:
            upload_document(i)


def show_dashboard_page():
    st.title("Dashboard")
    st.write("This is the dashboard page where you can see different visualizations.")
    # Embed an external website using an iframe
    st.components.v1.iframe(
        "http://ec2-34-207-119-191.compute-1.amazonaws.com:8088/superset/welcome/", height=600, scrolling=True)


def show_gen_ai_page():
    st.title("GenAI DataQuery")

    # Step 1: Ensure that files are uploaded
    if st.session_state.uploaded_reports:
        st.write("### Uploaded Reports")
        selected_report = st.selectbox(
            "Select a report:", list(st.session_state.uploaded_reports.keys()))

        # Load the selected report
        selected_dataset = st.session_state.uploaded_reports[selected_report]
    else:
        st.warning("Please upload a report first.")
        return

    # Step 2: Set dynamic predefined goals based on the selected report
    if "GRN-Report" in selected_report:
        predefined_goals = [
            "Generate a piechart for the top 5 vendors based on total value.",
            "Who are the top 5 materials in terms of Amount in LC",
            "Pie Chart for top 5 material description invested in the month of March 2024 "
        ]
    elif "Stock-Report" in selected_report:
        predefined_goals = []
    elif "Purchase-Report" in selected_report:
        predefined_goals = []
    elif "TECO-Report" in selected_report:
        predefined_goals = []
    elif "Reservation-Report" in selected_report:
        predefined_goals = []
    else:
        predefined_goals = []

    # Step 3: Goal Selection Dropdown
    st.write("## Select and Edit Your Goal")
    selected_goal = st.selectbox("Choose a predefined goal:", predefined_goals)

    # Step 4: Editable Goal Section
    user_goal_input = st.text_area(
        "Edit your analysis goal:", value=selected_goal, height=150)

    # Step 5: Submit button to trigger visualizations
    if st.button("Generate Visualizations"):
        if user_goal_input.strip() != "":
            st.write(f"Selected and Edited Goal: {user_goal_input}")

            # Proceed to generate visualizations after goal is submitted
            # Dummy key for OpenAI (replace with a valid key)
            openai_key = "your_openai_key_here"

            from lida import Manager, TextGenerationConfig, llm
            lida = Manager(text_gen=llm("openai", api_key=openai_key))
            textgen_config = TextGenerationConfig(
                n=1,  # Number of responses generated
                temperature=0.0,  # Make the model deterministic
                model="gpt-4-turbo",  # Model to be used
                use_cache=True,  # Use caching for faster generation
                top_p=0.1,  # Limit the next-token selection to a highly probable subset
                top_k=1,  # Consider only the most likely next word
            )

            # Summarize the dataset
            summary = lida.summarize(
                selected_dataset,
                summary_method="default",
                textgen_config=textgen_config
            )

            # Generate visualizations based on the edited goal
            visualizations = lida.visualize(
                summary=summary,
                goal=user_goal_input,  # Use the user-edited goal
                textgen_config=textgen_config,
                library="matplotlib"  # Example: Use matplotlib
            )

            viz_titles = [
                f'Visualization {i+1}' for i in range(len(visualizations))]

            selected_viz_title = st.selectbox(
                'Choose a visualization', options=viz_titles, index=0)

            selected_viz = visualizations[viz_titles.index(selected_viz_title)]

            if selected_viz.raster:
                from PIL import Image
                import io
                import base64

                imgdata = base64.b64decode(selected_viz.raster)
                img = Image.open(io.BytesIO(imgdata))
                st.image(img, caption=selected_viz_title,
                         use_column_width=True)
        else:
            st.warning("Please enter a valid goal.")


# Get the query parameter from the URL using st.query_params
query_params = st.query_params

# Initialize session state for the page if not present
if 'page' not in st.session_state:
    # Use the query parameter if it's set, else default to 'dashboard'
    if 'page' in query_params:
        st.session_state.page = query_params['page'][0]
    else:
        st.session_state.page = 'upload'  # Default page

# Sidebar navigation panel
with st.sidebar:
    # Buttons for navigation
    if st.button("Upload"):
        st.session_state.page = 'upload'
        # Set query parameters in the URL to simulate routes
        st.experimental_set_query_params(page='upload')
        st.experimental_rerun()  # Rerun the app to reflect the page change

    if st.button("Dashboard"):
        st.session_state.page = 'dashboard'
        st.experimental_set_query_params(page='dashboard')
        st.experimental_rerun()  # Rerun the app to reflect the page change

    if st.button("GenAI DataQuery"):
        st.session_state.page = 'genai'
        st.experimental_set_query_params(page='genai')
        st.experimental_rerun()  # Rerun the app to reflect the page change

# Display the appropriate page based on the session state
if st.session_state.page == 'upload':
    show_upload_page()
elif st.session_state.page == 'dashboard':
    show_dashboard_page()
elif st.session_state.page == 'genai':
    show_gen_ai_page()
