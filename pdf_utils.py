from fpdf import FPDF
import tempfile
from datetime import datetime

def create_pdf_report(metrics_df, start_date, end_date, title="Portfolio Report"):
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, title, ln=1, align='C')
    
    # Dates
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"{start_date} to {end_date}", ln=1, align='C')
    pdf.ln(10)
    
    # Metrics Table
    col_width = pdf.w / (len(metrics_df.columns) + 1)
    
    # Header Row
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(col_width, 10, "Metric", border=1)
    for col in metrics_df.columns:
        pdf.cell(col_width, 10, col, border=1)
    pdf.ln()
    
    # Data Rows
    pdf.set_font("Arial", '', 10)
    for metric in metrics_df.index:
        pdf.cell(col_width, 10, metric, border=1)
        for value in metrics_df.loc[metric]:
            pdf.cell(col_width, 10, f"{value:.2%}", border=1)
        pdf.ln()
    
    # Footer
    pdf.ln(15)
    pdf.set_font("Arial", 'I', 8)
    pdf.cell(0, 10, f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}", 0, 0, 'C')
    
    # Save to temp file
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(tmp_file.name)
    return tmp_file.name