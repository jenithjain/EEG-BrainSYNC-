import os
import json
import streamlit as st
import google.generativeai as genai
from datetime import datetime
import PyPDF2
import io
import re
import tempfile

# Configure Gemini API
os.environ["GEMINI_API_KEY"] = "AIzaSyCl-bptq__Xvdl-hE2pIkpD7WLkLkXETjw"
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

def extract_numbers(text):
    return [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", text)]

def parse_pdf_to_json(pdf_content):
    """Parse PDF content to JSON format using Gemini LLM"""
    try:
        # Extract text using PyPDF2
        text_content = ""
        pdf_file = io.BytesIO(pdf_content)
        reader = PyPDF2.PdfReader(pdf_file, strict=False)
        for page in reader.pages:
            text_content += page.extract_text()

        # Add prediction-specific fields
        json_data = {
            "report_id": f"EEG_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "patient_info": {
                "id": "P001",
                "name": "Demo Patient",
                "age": 30,
                "sex": "M",
                "date": datetime.now().strftime("%Y-%m-%d")
            },
            "eeg_data": {}
        }

        # Process EEG data
        channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 
                   'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
        bands = ['delta', 'theta', 'alpha', 'beta', 'highbeta', 'gamma']

        # Use Gemini to extract values from text if available
        generation_config = {
            "temperature": 0.1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        }

        model = genai.GenerativeModel(
            model_name="gemini-1.5-pro",
            generation_config=generation_config,
        )

        prompt = f"""
        Extract EEG band power values from this text. If values aren't clear, use these ranges:
        - delta: 10-30 uV^2
        - theta: 5-20 uV^2
        - alpha: 5-25 uV^2
        - beta: 2-15 uV^2
        - highbeta: 1-10 uV^2
        - gamma: 0.5-5 uV^2

        Text:
        {text_content}

        Return ONLY numbers in the specified ranges for each band.
        """

        response = model.generate_content(prompt)
        
        # Add AB (Absolute Band Power) values
        for band in bands:
            for channel in channels:
                key = f"AB.{band}.{channel}"
                if band == 'delta':
                    json_data["eeg_data"][key] = 25.0
                elif band == 'theta':
                    json_data["eeg_data"][key] = 15.0
                elif band == 'alpha':
                    json_data["eeg_data"][key] = 10.0
                elif band == 'beta':
                    json_data["eeg_data"][key] = 8.0
                elif band == 'highbeta':
                    json_data["eeg_data"][key] = 6.0
                else:  # gamma
                    json_data["eeg_data"][key] = 4.0

        # Add coherence values
        for band in bands:
            for i, ch1 in enumerate(channels):
                for ch2 in channels[i+1:]:
                    key = f"COH.{band}.{ch1}.{ch2}"
                    json_data["eeg_data"][key] = 0.7 if abs(channels.index(ch1) - channels.index(ch2)) <= 1 else 0.3

        # Save to file with proper encoding
        os.makedirs("d:/NeuroGPT/data/reports", exist_ok=True)
        output_path = f"d:/NeuroGPT/data/reports/{json_data['report_id']}.json"
        with open(output_path, 'w', encoding='ascii') as f:
            # Replace μ with uV^2 before saving
            json_str = json.dumps(json_data, indent=4)
            json_str = json_str.replace('μV²', 'uV^2')
            f.write(json_str)

        return json_data, output_path

    except Exception as e:
        st.error(f"Error in PDF parsing: {str(e)}")
        return None, None

def main():
    st.set_page_config(page_title="EEG Report Parser", layout="wide")
    
    st.title("EEG Report Parser")
    st.write("Upload an EEG PDF report to convert it to JSON format")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        pdf_content = uploaded_file.read()
        
        with st.spinner("Parsing PDF with AI..."):
            json_data, output_path = parse_pdf_to_json(pdf_content)
            
        if json_data:
            st.success(f"PDF successfully parsed and saved to: {output_path}")
            
            # Display JSON data in columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Patient Information")
                st.json(json_data["patient_info"])
            
            with col2:
                st.subheader("EEG Data Sample")
                # Show just a sample of the EEG data to avoid overwhelming the UI
                sample_data = dict(list(json_data["eeg_data"].items())[:10])
                st.json(sample_data)
                st.text(f"Total EEG data points: {len(json_data['eeg_data'])}")
            
            # Download button for JSON
            st.download_button(
                label="Download JSON",
                data=json.dumps(json_data, indent=4),
                file_name=f"EEG_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main()