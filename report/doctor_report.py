import os
from datetime import datetime
import streamlit as st
from fpdf import FPDF

def clean_text(text):
    """Remove or replace problematic characters for PDF generation"""
    if not isinstance(text, str):
        text = str(text)
    # Replace common problematic characters
    replacements = {
        "'": "'",
        """: '"',
        """: '"',
        "–": "-",
        "—": "-",
        "…": "...",
        "\u2019": "'",
        "\u201C": '"',
        "\u201D": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2026": "..."
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Final fallback - strip any remaining non-ASCII characters
    return ''.join(c if ord(c) < 128 else '?' for c in text)

def generate_pdf_report(patient_info, prediction_results, band_averages, analysis, treatment_results=None):
    """Generate PDF report with clean text handling"""
    pdf = FPDF()
    pdf.add_page()
    
    # Add title
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'EEG Analysis Report', 0, 1, 'C')
    
    # Patient info
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Patient Information', 0, 1)
    pdf.set_font('Arial', '', 12)
    
    # Only include essential patient info to avoid encoding issues
    safe_fields = ['id', 'name', 'age', 'sex', 'date']
    for key in safe_fields:
        if key in patient_info:
            pdf.cell(0, 8, f'{key}: {clean_text(patient_info[key])}', 0, 1)
    
    # Analysis results
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Analysis Results', 0, 1)
    pdf.set_font('Arial', '', 12)
    
    pdf.cell(0, 8, f"Diagnosis: {clean_text(prediction_results['primary_diagnosis'])}", 0, 1)
    pdf.cell(0, 8, f"Confidence: {prediction_results['confidence']:.1f}%", 0, 1)
    
    # Band Analysis
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'EEG Band Analysis', 0, 1)
    
    # Create table for band values
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(60, 8, 'Frequency Band', 1, 0, 'C')
    pdf.cell(40, 8, 'Value (uV^2)', 1, 0, 'C')
    pdf.cell(40, 8, 'Status', 1, 1, 'C')
    
    pdf.set_font('Arial', '', 10)
    bands = {
        'Delta (0.5-4 Hz)': {'value': band_averages['delta'], 'normal': (5, 20)},
        'Theta (4-8 Hz)': {'value': band_averages['theta'], 'normal': (5, 15)},
        'Alpha (8-13 Hz)': {'value': band_averages['alpha'], 'normal': (10, 25)},
        'Beta (13-30 Hz)': {'value': band_averages['beta'], 'normal': (5, 20)},
        'High Beta (20-30 Hz)': {'value': band_averages['highbeta'], 'normal': (5, 15)},
        'Gamma (>30 Hz)': {'value': band_averages['gamma'], 'normal': (1, 5)}
    }
    
    for band_name, info in bands.items():
        value = info['value']
        if value < info['normal'][0]:
            status = 'Low'
        elif value > info['normal'][1]:
            status = 'High'
        else:
            status = 'Normal'
        
        pdf.cell(60, 8, band_name, 1, 0)
        pdf.cell(40, 8, f"{value:.2f}", 1, 0)
        pdf.cell(40, 8, status, 1, 1)
    
    # Clinical Analysis - simplified to avoid encoding issues
    pdf.ln(5)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Clinical Analysis', 0, 1)
    pdf.set_font('Arial', '', 10)
    
    # Break analysis into short paragraphs and clean each one
    clean_paragraphs = []
    for paragraph in clean_text(analysis).split('\n'):
        if paragraph.strip():
            clean_paragraphs.append(paragraph.strip())
    
    for paragraph in clean_paragraphs:
        pdf.multi_cell(0, 6, paragraph)
        pdf.ln(2)
    
    # Save PDF
    os.makedirs('d:/NeuroGPT/outputs/reports', exist_ok=True)
    filename = f"EEG_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    filepath = f"d:/NeuroGPT/outputs/reports/{filename}"
    
    try:
        pdf.output(filepath)
        return filepath
    except Exception as e:
        # Fallback to ultra-safe mode if still having issues
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'EEG Analysis Report', 0, 1, 'C')
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 10, 'Basic Patient Information', 0, 1)
        pdf.cell(0, 8, f"ID: {patient_info.get('id', 'N/A')}", 0, 1)
        pdf.cell(0, 8, f"Diagnosis: {clean_text(prediction_results['primary_diagnosis'])}", 0, 1)
        pdf.output(filepath)
        return filepath

def display_report_section(patient_info, prediction_results, band_averages, analysis, treatment_results=None):
    """Display report section in Streamlit"""
    st.subheader("📄 Clinical Report")
    
    try:
        # Generate PDF report
        pdf_path = generate_pdf_report(patient_info, prediction_results, band_averages, analysis, treatment_results)
        
        # Display download button
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
            st.download_button(
                label="📥 Download Clinical Report (PDF)",
                data=pdf_bytes,
                file_name=os.path.basename(pdf_path),
                mime="application/pdf"
            )
    except Exception as e:
        st.error(f"Could not generate PDF report: {str(e)}")
        st.info("Displaying text report instead:")
        
        # Display text version as fallback
        st.write("### EEG Analysis Report")
        st.write("#### Patient Information")
        for key, value in patient_info.items():
            st.write(f"{key}: {value}")
        
        st.write("#### Diagnosis")
        st.write(f"Condition: {prediction_results['primary_diagnosis']}")
        st.write(f"Confidence: {prediction_results['confidence']:.1f}%")