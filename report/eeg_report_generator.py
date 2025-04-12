import json
import pandas as pd
import numpy as np
from datetime import datetime
import os
from fpdf import FPDF2 as FPDF

class EEGReportGenerator:
    def __init__(self):
        self.channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 
                        'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
        
    def generate_pdf_report(self, patient_info, eeg_data, output_dir="d:/NeuroGPT/data/reports"):
        pdf = FPDF()
        pdf.add_page()
        
        # Header with border
        pdf.set_fill_color(240, 240, 240)
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 15, 'EEG Examination Report', 1, 1, 'C', True)
        pdf.ln(10)
        
        # Patient Information in a table
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Patient Information', 0, 1)
        pdf.set_font('Arial', '', 12)
        pdf.set_fill_color(245, 245, 245)
        
        # Create info table
        for key, value in patient_info.items():
            pdf.cell(40, 8, key.upper() + ":", 1, 0, 'L', True)
            pdf.cell(0, 8, str(value), 1, 1, 'L')
        pdf.ln(10)
        
        # EEG Data Section
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'EEG Measurements', 0, 1)
        
        # Channel mapping table
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, 'Channel Mapping:', 0, 1)
        pdf.set_font('Arial', '', 10)
        
        # Print channels in a grid
        channels_per_row = 5
        for i in range(0, len(self.channels), channels_per_row):
            channel_group = self.channels[i:i+channels_per_row]
            for ch in channel_group:
                pdf.cell(38, 8, ch, 1, 0, 'C', True)
            pdf.ln()
        pdf.ln(5)
        
        # Frequency Band Data in tables
        pdf.set_font('Arial', 'B', 12)
        # Change the import
        from fpdf import FPDF2 as FPDF
        
        # In generate_pdf_report method, change μV² to uV^2:
        for band in ['delta', 'theta', 'alpha', 'beta', 'highbeta', 'gamma']:
            values = eeg_data[band]
            pdf.cell(0, 8, f"{band.upper()} Band Values (uV^2):", 0, 1)
            
            # Create table header
            pdf.set_font('Arial', 'B', 10)
            for i in range(5):
                pdf.cell(38, 8, 'Channel', 1, 0, 'C', True)
                pdf.cell(38, 8, 'Value', 1, 0, 'C', True)
            pdf.ln()
            
            # Fill data
            pdf.set_font('Arial', '', 10)
            row = []
            for i, value in enumerate(values):
                if i > 0 and i % 5 == 0:
                    for item in row:
                        pdf.cell(38, 8, item[0], 1, 0, 'C')
                        pdf.cell(38, 8, f"{item[1]:.2f}", 1, 0, 'C')
                    pdf.ln()
                    row = []
                row.append((self.channels[i], value))
            
            # Print remaining items
            if row:
                for item in row:
                    pdf.cell(38, 8, item[0], 1, 0, 'C')
                    pdf.cell(38, 8, f"{item[1]:.2f}", 1, 0, 'C')
                pdf.ln()
            pdf.ln(5)
        
        # Technical Information
        pdf.ln(5)
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Technical Information', 0, 1)
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 8, 'Sampling Rate: 256 Hz', 0, 1)
        pdf.cell(0, 8, 'Recording Duration: 5 minutes', 0, 1)
        
        # Footer with report ID
        pdf.set_y(-30)
        pdf.set_font('Arial', 'I', 10)
        report_id = f"EEG_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        pdf.cell(0, 10, f'Report ID: {report_id}', 0, 1, 'L')
        
        # Save both PDF and JSON
        pdf_path = os.path.join(output_dir, f"{report_id}.pdf")
        json_path = os.path.join(output_dir, f"{report_id}.json")
        
        pdf.output(pdf_path)
        
        # Save JSON with exact same structure as needed for ML
        with open(json_path, 'w') as f:
            json.dump({
                "report_id": report_id,
                "patient_info": patient_info,
                "eeg_data": eeg_data
            }, f, indent=4)
        
        return pdf_path, json_path

def example_usage():
    patient_info = {
        "id": "P12345",
        "age": 35,
        "sex": "M"
    }
    
    eeg_data = {
        "delta": [35.2, 34.8, 33.9, 34.3, 34.5, 34.2, 33.8, 33.5, 33.7, 34.0,
                 33.6, 33.2, 32.9, 33.3, 33.5, 33.2, 32.8, 32.5, 32.7],
        "theta": [22.3, 22.1, 21.8, 22.2, 22.4, 22.1, 21.7, 21.4, 21.6, 21.9,
                 21.5, 21.1, 20.8, 21.2, 21.4, 21.1, 20.7, 20.4, 20.6],
        "alpha": [18.7, 18.5, 18.2, 18.6, 18.8, 18.5, 18.1, 17.6, 18.0, 18.3,
                 17.9, 17.5, 17.2, 17.6, 17.8, 17.5, 17.1, 16.8, 17.0],
        "beta": [12.5, 12.3, 12.0, 12.4, 12.6, 12.3, 11.9, 11.4, 11.8, 12.1,
                11.7, 11.3, 11.0, 11.4, 11.6, 11.3, 10.9, 10.6, 10.8],
        "highbeta": [8.1, 7.9, 7.6, 8.0, 8.2, 7.9, 7.5, 7.0, 7.4, 7.7,
                    7.3, 6.9, 6.6, 7.0, 7.2, 6.9, 6.5, 6.2, 6.4],
        "gamma": [3.8, 3.6, 3.3, 3.7, 3.9, 3.6, 3.2, 2.7, 3.1, 3.4,
                 3.0, 2.6, 2.3, 2.7, 2.9, 2.6, 2.2, 1.9, 2.1]
    }
    
    generator = EEGReportGenerator()
    pdf_path, json_path = generator.generate_pdf_report(patient_info, eeg_data)
    
    print(f"Report generated successfully!")
    print(f"PDF saved to: {pdf_path}")
    print(f"Data saved to: {json_path}")

if __name__ == "__main__":
    example_usage()