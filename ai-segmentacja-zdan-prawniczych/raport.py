import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

def register_polish_font():
    """Próbuje znaleźć i zarejestrować czcionkę z polskimi znakami."""
    font_paths = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/arial.ttf",
        "C:\\Windows\\Fonts\\arial.ttf"
    ]
    
    selected_font = 'Helvetica' 
    
    for path in font_paths:
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont('Arial', path))
                return 'Arial'
            except Exception:
                continue
                
    return selected_font

def format_sentences_pdf(sent_list):
    formatted = ""
    for i, sent in enumerate(sent_list, 1):
        formatted += f"<b>[{i}]</b> {sent}<br/>"
    return formatted

def generate_pdf(results, filename="raport.pdf"):
    doc = SimpleDocTemplate(filename, pagesize=landscape(A4), rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    
    font_name = register_polish_font()
    elements = []
    styles = getSampleStyleSheet()
    
    title_style = styles['Title']
    title_style.fontName = font_name

    elements.append(Spacer(1, 20))

    cell_style = ParagraphStyle(
        'CellStyle',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=8,
        leading=10
    )
    
    headers = ["NLTK (Default)", "NLTK + Reguły", "Spacy", "Stanza"]
    
    for i, res in enumerate(results, 1):
        # Nagłówek sekcji (Przykład #X)
        elements.append(Paragraph(f"<b>PRZYKŁAD #{i}</b>", ParagraphStyle('H3', parent=styles['Normal'], fontName=font_name, fontSize=12, spaceAfter=10)))
        
        # Przygotowanie danych do tabeli
        nltk_std_txt = format_sentences_pdf(res['nltk_std'])
        nltk_imp_txt = format_sentences_pdf(res['nltk_imp'])
        spacy_txt = format_sentences_pdf(res['spacy'])
        stanza_txt = format_sentences_pdf(res['stanza'])
        
        data = [
            headers,
            [Paragraph(nltk_std_txt, cell_style), 
             Paragraph(nltk_imp_txt, cell_style), 
             Paragraph(spacy_txt, cell_style), 
             Paragraph(stanza_txt, cell_style)]
        ]
    
        t = Table(data, colWidths=[190, 190, 190, 190])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('FONTNAME', (0, 0), (-1, -1), font_name),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ]))
        
        elements.append(t)
        elements.append(Spacer(1, 25))

    doc.build(elements)
    print(f"\nwygenerowano {filename}")