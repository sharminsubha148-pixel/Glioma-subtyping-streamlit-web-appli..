PROJECT TITLE: CLINICALLY INTERPRETABLE DEEP LEARNING GENOMIC BASED FOR GLIOMA SUBTYPING USING SHAP ANALYSIS

SUBMITTED BY:

Sharmin Akter

Project Overview:

This project utilizes an advanced Deep Learning model to predict the subtype or grade of Glioma tumors based on genomic data. A key feature of this system is the integration of 'SHAP Analysis', which provides clinical interpretability by explaining why the model made a specific prediction.

File Structure: Ensure that all the following files are located in the same folder: app.py (Main Streamlit Web Interface) model.keras (Pre-trained Deep Learning Model) minmax_scaler.pkl (Data Normalization Scaler) requirements.txt (List of necessary Python libraries) clinical_glioma_grading.csv (Sample Input Dataset for testing) code 03 11.ipynb (Original Research/Source Code)

System Requirements:

Python Version: 3.8 or higher. Internet Connection: Required for initial library installation.

Installation & Execution Guide:
Step 1: Copy the project folder to your local machine and open the Command Prompt (CMD) or Terminal inside that folder.

Step 2: Install the required libraries by running the following command: pip install -r requirements.txt

Step 3: Launch the application by running: python -m streamlit run app.py

How to Use:
Once the command is executed, a web interface will open in your browser. Use the sidebar on the left to input patient data (Age, Gender) and various genomic mutations (e.g., IDH1, TP53, etc.). You can refer to the 'clinical_glioma_grading.csv' file included in the folder for sample test inputs. Click the 'Predict Grade' button to see the prediction results along with the SHAP Waterfall Plot for clinical explanation.
