import os
import spacy
from datasets import load_dataset
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

def register_polish_font():

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

def generate_pdf(test_results, filename="raport2.pdf"):
    """Generuje PDF na podstawie wyników segmentacji."""
    doc = SimpleDocTemplate(filename, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    
    font_name = register_polish_font()
    elements = []
    styles = getSampleStyleSheet()
    
  
    title_style = ParagraphStyle(
        'Title', 
        parent=styles['Heading1'], 
        fontName=font_name, 
        fontSize=16, 
        spaceAfter=20, 
        alignment=1
    )
    elements.append(Paragraph("", title_style))
    elements.append(Spacer(1, 10))

    # Style komórek tabeli
    cell_style = ParagraphStyle(
        'CellStyle',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=9,
        leading=12
    )
    
    idx_style = ParagraphStyle(
        'IdxStyle',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=9,
        leading=12,
        alignment=1 
    )
    
    for res in test_results:

        header_text = f"<b>PRZYKŁAD #{res['id']}</b> (Oryginalna długość: {res['length']} znaków)"
        elements.append(Paragraph(header_text, ParagraphStyle('H3', parent=styles['Normal'], fontName=font_name, fontSize=12, spaceAfter=10)))

        data = [["Nr", "Wyodrębnione zdanie (Segment)"]]
        

        for i, sent in enumerate(res['sentences'], 1):
            data.append([
                Paragraph(f"<b>[{i}]</b>", idx_style), 
                Paragraph(sent, cell_style)
            ])
    

        t = Table(data, colWidths=[40, 490])

        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('FONTNAME', (0, 0), (-1, -1), font_name),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ]))
        
        elements.append(t)
        elements.append(Spacer(1, 25))

    doc.build(elements)


def run_test_and_generate_pdf():
    
    try:
        nlp = spacy.load("./model_final/model-best")
    except Exception as e:
        return

   
    dataset = load_dataset(
        "joelniklaus/Multi_Legal_Pile", 
        "pl_legislation",
        split="train", 
        streaming=True,
        trust_remote_code=True
    )

    test_results = []

    

    for idx, item in enumerate(dataset.shuffle(buffer_size=1000)):
        raw_text = item['text']
        if len(raw_text) > 2000: 
            
            doc = nlp(raw_text)

            sentences = []
            for sent in doc.sents:
                clean_sent = sent.text.replace("\n", " ").strip()
                if clean_sent:
                    sentences.append(clean_sent)

            test_results.append({
                'id': len(test_results) + 1,
                'length': len(raw_text),
                'sentences': sentences
            })
            
        if len(test_results) >= 5:
            break


    generate_pdf(test_results)

if __name__ == "__main__":
    run_test_and_generate_pdf()