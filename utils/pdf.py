from fpdf import FPDF
import io

def create_recommendations_pdf(recommendations, profile_type):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(0, 10, f"Investment Recommendations for {profile_type} Investor", ln=True, align="C")
    pdf.ln(10)

    for stock in recommendations:
        pdf.set_font("Arial", size=12, style='B')
        pdf.cell(0, 10, f"{stock['ticker']} - {stock['name']}", ln=True)
        pdf.set_font("Arial", size=11)
        pdf.cell(0, 8, f"Sector: {stock.get('sector', 'N/A')}", ln=True)
        pdf.cell(0, 8, f"P/E Ratio: {stock.get('pe_ratio', 'N/A')}", ln=True)
        pdf.cell(0, 8, f"Dividend Yield: {stock.get('dividend_yield', 'N/A')}", ln=True)
        pdf.cell(0, 8, "Reasons it matches your profile:", ln=True)
        for reason in stock.get('match_reasons', []):
            pdf.cell(0, 8, f"- {reason}", ln=True)
        pdf.ln(5)

    # Output PDF to bytes buffer
    pdf_output = io.BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)
    return pdf_output.read()