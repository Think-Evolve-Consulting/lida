import streamlit as st
import pandas as pd
from datetime import datetime
import mysql.connector
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set page configuration
st.set_page_config(page_title="BPD Dashboard", layout="wide")

st.markdown("""
    <style>
    .stButton > button {
        width: 100% !important;
    }
    </style>
    """, unsafe_allow_html=True)

# MySQL connection details


def execute_query(query):
    connection = mysql.connector.connect(
        host="20.198.20.220",
        user="root",
        password="pass123",
        database="bpd"
    )
    cursor = connection.cursor()
    cursor.execute(query)
    connection.commit()
    cursor.close()
    connection.close()


def fetch_table_data(table_name):
    connection = mysql.connector.connect(
        host="20.198.20.220",
        user="root",
        password="pass123",
        database="bpd"
    )

    query = f"SELECT * FROM `{table_name}`"
    data = pd.read_sql(query, connection)
    connection.close()

    return data


def get_table_update_date(table_name):
    connection = mysql.connector.connect(
        host="20.198.20.220",
        user="root",
        password="pass123",
        database="bpd"
    )
    cursor = connection.cursor()
    query = f"""
        SELECT DATE(update_time) as last_update_date
        FROM information_schema.tables
        WHERE table_schema = 'bpd' AND table_name = '{table_name}'
    """
    cursor.execute(query)
    result = cursor.fetchone()
    cursor.close()
    connection.close()

    return result[0].strftime('%d-%b-%Y') if result and result[0] else "None"


def backup_and_insert_data(table_name, data):
    backup_table_name = f"{table_name}_backup"

    # Define the date columns that need to be converted
    # Add other date columns if needed
    date_columns = ['Document Date', 'Posting Date']

    try:
        # Step 1: Check if backup table exists
        connection = mysql.connector.connect(
            host="20.198.20.220",
            user="root",
            password="pass123",
            database="bpd"
        )
        cursor = connection.cursor()

        # Check if the backup table exists
        cursor.execute(f"""
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema = 'bpd' AND table_name = '{backup_table_name}'
        """)
        exists = cursor.fetchone()[0] > 0

        if exists:
            # Truncate the backup table if it already exists
            truncate_backup_query = f"TRUNCATE TABLE {backup_table_name}"
            execute_query(truncate_backup_query)
        else:
            # Create backup table if it doesn't exist
            create_backup_query = f"CREATE TABLE {backup_table_name} AS SELECT * FROM {table_name}"
            execute_query(create_backup_query)
            st.success(f"Backup created: {backup_table_name}")

        # Step 2: Clear current table
        clear_table_query = f"TRUNCATE TABLE {table_name}"
        execute_query(clear_table_query)

        # Step 3: Get the column names from the database table
        cursor.execute(
            f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '{table_name}'")
        columns = [row[0] for row in cursor.fetchall()]

        # Ensure data only contains the columns that exist in the database table
        # Filter the DataFrame to only include valid columns
        data = data[columns]

        # Convert specified date columns to the required format
        for date_col in date_columns:
            if date_col in data.columns:
                data[date_col] = pd.to_datetime(data[date_col], format='%d.%m.%Y', errors='coerce').dt.strftime(
                    '%Y-%m-%d')  # Convert to MySQL date format

        # Escape column names for SQL
        # Use backticks for column names
        escaped_columns = [f"`{col}`" for col in columns]

        # Build the insert query
        insert_data_query = f"INSERT INTO {table_name} ({', '.join(escaped_columns)}) VALUES "
        values_list = []

        for row in data.itertuples(index=False):
            values_tuple = []
            for value in row:
                if isinstance(value, str):
                    # Escape single quotes in strings
                    values_tuple.append("'" + value.replace("'", "''") + "'")
                elif pd.isnull(value):
                    values_tuple.append("NULL")  # Handle NaN as NULL
                else:
                    # Convert other types to string
                    values_tuple.append(str(value))

            values_list.append(f"({', '.join(values_tuple)})")

        insert_data_query += ', '.join(values_list)

        execute_query(insert_data_query)

    except Exception as e:
        st.error(f"An error occurred: {e}")


def initialize_uploaded_reports():
    if 'uploaded_reports' not in st.session_state:
        st.session_state.uploaded_reports = {}
        st.session_state.uploaded_reports['GRN-Report'] = fetch_table_data(
            "grn1")
        st.session_state.uploaded_reports['Stock-Report'] = fetch_table_data(
            "Table 1 - Master Data")
        st.session_state.uploaded_reports['Purchase-Report'] = fetch_table_data(
            "Table 2 - Purchase Report")


# Initialize session state for page
if 'page' not in st.session_state:
    st.session_state.page = 'upload'  # Default page

# Call the initialization function
initialize_uploaded_reports()


def show_upload_page():
    st.title("Upload Page")
    grn_date = get_table_update_date("grn1")
    stock_date = get_table_update_date("Table 1 - Master Data")
    purchase_date = get_table_update_date("Table 2 - Purchase Report")

    documents = [
        {"Report File": "GRN-Report", "Uploaded Date": grn_date},
        {"Report File": "Stock-Report", "Uploaded Date": stock_date},
        {"Report File": "Purchase-Report", "Uploaded Date": purchase_date},
        {"Report File": "TECO-Report", "Uploaded Date": "None"},
        {"Report File": "Reservation-Report", "Uploaded Date": "None"},
    ]

    # Mapping of report files to database table names
    table_mapping = {
        "GRN-Report": "grn1",
        "Stock-Report": "Table 1 - Master Data",
        "Purchase-Report": "Table 2 - Purchase Report",
        "TECO-Report": "teco_report",  # Add appropriate table name if exists
        # Add appropriate table name if exists
        "Reservation-Report": "reservation_report",
    }

    if 'upload_dates' in st.session_state:
        saved_documents = st.session_state.upload_dates
        saved_dict = {doc['Report File']: doc for doc in saved_documents}
        for doc in documents:
            if doc['Report File'] not in saved_dict:
                saved_documents.append(doc)
        documents = saved_documents

    df = pd.DataFrame(documents)

    def upload_document(index):
        st.markdown(f"**Browse {df.iloc[index]['Report File']}:**")
        uploaded_file = st.file_uploader(
            "", type=["csv", "json", "xlsx"], key=index)
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                data = pd.read_json(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                data = pd.read_excel(uploaded_file)

            # Step 1: Clean the "Amount in LC" column (replace 'Amount in LC' with the actual column name)
            if 'Amount in LC' in data.columns:
                # Remove the ₹ symbol and commas using basic string operations
                data['Amount in LC'] = data['Amount in LC'].apply(lambda x: str(
                    x).replace('₹', '').replace(',', '') if pd.notnull(x) else x)

                # Convert the cleaned column to numeric type
                data['Amount in LC'] = pd.to_numeric(
                    data['Amount in LC'], errors='coerce')

            current_date = datetime.now().strftime('%d-%b-%Y')
            df.at[index, 'Uploaded Date'] = current_date
            documents[index]['Uploaded Date'] = current_date
            st.session_state.uploaded_reports[df.iloc[index]
                                              ['Report File']] = data
            st.success(f"{uploaded_file.name} uploaded successfully!")
            st.session_state.upload_dates = documents

            # Backup and insert the new data
            report_name = df.iloc[index]['Report File']
            table_name = table_mapping.get(report_name)
            if table_name:
                backup_and_insert_data(table_name, data)
            else:
                st.error(f"Table name for {report_name} is not defined.")

    st.write("## Document Upload Table")
    for i, row in df.iterrows():
        col1, col2, col3 = st.columns([3, 2, 2])
        col1.write(row['Report File'])
        col2.write(row['Uploaded Date'])
        with col3:
            upload_document(i)


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


# Sidebar navigation panel
with st.sidebar:
    if st.button("Upload"):
        st.session_state.page = 'upload'
        st.experimental_rerun()
    if st.button("GenAI DataQuery"):
        st.session_state.page = 'genai'
        st.experimental_rerun()

    st.markdown('''
        <style>
        .dashboard-button {
            background-color: rgb(255,255,255);
            color: rgb(38, 39, 48);
            padding: 0.25rem 1rem;
            border-radius: 0.25rem;
            border: 1px solid rgba(49, 51, 63, 0.4);
            cursor: pointer;
            display: inline-block;
            text-align: center;
            font-size: 1rem;
            font-weight: 400;
            line-height: 1.6;
            width: 100%;
            transition: color 0.3s ease, border-color 0.3s ease;
        }
        .dashboard-button:hover {
            color: rgb(0, 153, 0);
            border-color: rgb(0, 153, 0);
        }
        </style>
        <a href="http://20.198.20.220:8088/superset/welcome/" target="_blank">
            <div class="dashboard-button">Dashboard</div>
        </a>
        ''', unsafe_allow_html=True)

# Display the appropriate page based on the session state
if st.session_state.page == 'upload':
    show_upload_page()
elif st.session_state.page == 'genai':
    show_gen_ai_page()
