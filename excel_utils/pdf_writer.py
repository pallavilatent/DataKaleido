from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether,ListFlowable, ListItem, Frame
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import LongTable
import seaborn as sns
import matplotlib.pyplot as plt
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
import os
import pandas as pd
from reportlab.platypus import Table, TableStyle, Image, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch


PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN_LEFT = inch
MARGIN_RIGHT = inch
MARGIN_TOP = inch
MARGIN_BOTTOM = inch

FRAME_WIDTH = PAGE_WIDTH - MARGIN_LEFT - MARGIN_RIGHT
FRAME_HEIGHT = PAGE_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM

frame_list = [
    Frame(
        x1=MARGIN_LEFT,
        y1=MARGIN_BOTTOM,
        width=FRAME_WIDTH,
        height=FRAME_HEIGHT,
        id='normal_frame'
    ),
]

styles = getSampleStyleSheet()
styleN = styles["Normal"]
elements = []
bullet_style = styles['BodyText']
bullet_points = [
    "Measures of Central Tendency refers to the typical value for a variable (single representative value). Geometric mean and Harmonic mean are used to summarize variables that represent the percentages and ratios respectively. "
    "Measures of Dispersion show how spread out or varied the values in a dataset are from the central tendency value."
]
boxplot_points = [
    "Quartiles are values that divide a variable into 4 equal parts each containing 25% of the data incrementally. "
    "Skewness refers to the lack of symmetricity in a variable."]
boxplot_text= '<br/>'.join([f'• {point}' for point in boxplot_points])
bullet_text = '<br/>'.join([f'• {point}' for point in bullet_points])
bullet_list = ListFlowable(
    [ListItem(Paragraph(item, bullet_style)) for item in bullet_points],
    bulletType='bullet',
    start='circle'
)
boxlist = ListFlowable(
    [ListItem(Paragraph(item, bullet_style)) for item in boxplot_points],
    bulletType='bullet',
    start='circle'
)
center_bold_header_style = ParagraphStyle(
    name='CenterBoldHeader',
    parent=styles['Normal'],
    alignment=TA_CENTER,
    fontName='Helvetica-Bold',
    fontSize=10,
    textColor=colors.black,
    backColor=colors.lightblue
)

cell_style = ParagraphStyle(name='cell_style', fontSize=8, leading=10)

styles.add(ParagraphStyle(
    name='CenteredHeading1',
    parent=styles['Heading1'],
    alignment=TA_CENTER,
    spaceAfter=12  # Adds space after the heading
    #alignment = 1
))
styles.add(ParagraphStyle(
    name='CenteredHeading2',
    parent=styles['Heading2'],
    alignment=TA_CENTER,
    spaceAfter=10  # A slightly smaller space after for subheadings
))

def add_page_num(canvas, doc):

    #Function to add a page number to the bottom center of each page.
    # Get page dimensions
    page_width, page_height = letter

    # Set font and size for the page number
    canvas.setFont('Helvetica', 9)
    
    # Get the current page number
    page_number = canvas.getPageNumber()
    
    # Format the text
    page_num_text = f"Page {page_number}"
    
    # Calculate text position (bottom center)
    text_x = page_width / 2
    text_y = 0.5 * inch
    
    # Draw the string on the canvas
    canvas.drawCentredString(text_x, text_y, page_num_text)


def write_text(elements,text, style_name='Normal', spacer=12):
    if isinstance(text, dict):
        text = "<br/>".join([f"{k}: {v}" for k, v in text.items()])
    elif isinstance(text, list):
        text = "<br/>".join([f"• {str(item)}" for item in text])
    elif not isinstance(text, str):
        text = str(text)
    elements.append(Paragraph(text, styles[style_name]))
    elements.append(Spacer(1, spacer))

def box_table_text(elements,text=None, width=450):
    # Create styles
    center_bold_style = ParagraphStyle(
        name='CenterBold',
        parent=styles['Normal'],
        alignment=TA_CENTER,
        fontName='Helvetica-Bold',
        fontSize=12
    )

    # Create the paragraph
    para = Paragraph(text, center_bold_style)

    # Create table as a box
    box_table = Table([[para]], colWidths=[width])

    # Style the box
    box_table.setStyle(TableStyle([
        ('BOX', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('BACKGROUND', (0, 0), (-1, -1), colors.whitesmoke),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))

    elements.append(box_table)

def wrap_long_text(text, max_char=80):
    #Inserts <br/> in long strings every `max_char` characters. This allows ReportLab to wrap text inside table cells.
    text = str(text)
    return "<br/>".join(text[i:i+max_char] for i in range(0, len(text), max_char))

def add_wrapped_table_to_elements(elements, df, title=None, extra_text=None, max_width=7.0 * inch, max_rows_per_chunk=500):
    # Add a styled and wrapped table to PDF elements, preventing overflow.
    # - Automatically resizes columns based on content
    # - Wraps long cell text
    # - Splits table into chunks if it's too big

    styles = getSampleStyleSheet()
    para_style = ParagraphStyle('table_cell', fontSize=8, leading=10)

    total_chunks = (len(df) // max_rows_per_chunk) + 1 

    for i in range(0, len(df), max_rows_per_chunk):
        chunk = df.iloc[i:i+max_rows_per_chunk]
        data = [chunk.columns.tolist()] + chunk.values.tolist()

        wrapped_data = []
        for row in data:
            wrapped_row = []
            for cell in row:
                text = str(cell)
                if len(text) > 300:
                    text = wrap_long_text(text, max_char=80)
                wrapped_row.append(Paragraph(text, para_style))
            wrapped_data.append(wrapped_row)

        wrapped_data[0] = [Paragraph(str(col), center_bold_header_style) for col in df.columns]


        col_widths = [max_width / len(df.columns)] * len(df.columns)

        table = Table(wrapped_data, colWidths=col_widths, hAlign='CENTER', repeatRows=1, splitByRow=1)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),  # Bold header
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),             # Center header
            #('VALIGN', (0, 0), (-1, 0), 'MIDDLE'),            # Vertically center header
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
            ('ALIGN', (0, 1), (-1, -1), 'CENTER'),            # Center all other rows
            ('GRID', (0, 0), (-1, -1), 0.25, colors.black),
            ('ALIGN', (1, 1), (-1, -1), 'CENTER'),  # Center align all data *except* the first column
            ('ALIGN', (0, 1), (0, -1), 'LEFT'),     # Left align only the data in the first column
            ('VALIGN', (0, 1), (-1, -1), 'MIDDLE') # Vertical align all body rows
        ]))

        if i == 0 and title:
            elements.append(Paragraph(f"<b>{title}</b>", styles['Heading3']))
            elements.append(Spacer(1, 6))

        if extra_text:
            elements.append(Paragraph(extra_text, styles['BodyText']))
            elements.append(Spacer(1, 6))

        elements.append(table)
        elements.append(Spacer(1, 12))

        # Insert page break if another chunk is coming
        if i + max_rows_per_chunk < len(df):
            elements.append(PageBreak())

def generate_pdf_report(filename, data):
    doc = SimpleDocTemplate(filename, pagesize=A4) #letter
    elements=[]
    write_text(elements,'DataKaleido', style_name='CenteredHeading1', spacer=5)
    write_text(elements, 'Accelerate Your Data Discovery through the lens of clarity', style_name='CenteredHeading2', spacer=5)
    write_text(elements,'This report consists of a comprehensive Exploratory Data Analysis for the numeric, categorical and binary variables in your dataset. This gives an end-to-end view right from basic data structures and goes till automated feature engineering. This not only provides automated insights and recommendations but also performs relevant corrective actions to the dataset using Statistical techniques.',spacer=8)


    write_text(elements,data.get('intro_text', ''))

    # ---- Observations ----
    box_table_text(elements,text="Data Structure")
    add_wrapped_table_to_elements(elements,data['data_quality_df'], title="Initial Data Summary")


    #--- remove id column
    extra_text = data['summary_scale_df'] + "\n\n" + data['summary_scale_df_1']
    add_wrapped_table_to_elements(elements,data['scale_df_copy'],extra_text=extra_text, title="Know your Datatypes",)
    write_text(elements, "Summary:", style_name="Heading3", spacer=6)
    write_text(elements,data['no_cat'], spacer=12)
    write_text(elements,data['no_num'], spacer=12)
    write_text(elements,data['no_text'], spacer=12)
    write_text(elements,data['no_datetime'], spacer=12)
    write_text(elements, data['no_percentage'], spacer=12)
    write_text(elements, data['no_ratio'], spacer=12)
    write_text(elements, data['count_geo'], spacer=12)
    write_text(elements,data.get('Removed_id_columns', ''),spacer=12)

    #write_text(elements,'<< This section is left intentionally blank. Please go to the next page>> ',style_name='Normal',spacer=15)
    
    elements.append(PageBreak())

    box_table_text(elements,text="Data Validation")


    add_wrapped_table_to_elements(elements,data['missing_df'],extra_text=data['missing_summ'], title="Missing value Report")
    write_text(elements, "Action(s) taken:", style_name="Heading3", spacer=6)
    write_text(elements, data['summary_message'], spacer=15)
    missing_summary=data['missing_summary']
    for item in missing_summary:
        bullet_paragraph = Paragraph(f"• {item}", styles["Normal"])
        elements.append(bullet_paragraph)
        elements.append(Spacer(1, 5))  # Optional spacing between bullets

    #elements.append(PageBreak())
    write_text(elements, "Data Summary", style_name="Heading3", spacer=6)
    write_text(elements, "Let us understand the basic range of the data in the below table:", spacer=6)
    add_wrapped_table_to_elements(elements,data['summary_df_1'])
    write_text(elements, "Summary Statistics", style_name="Heading3", spacer=6)
    write_text(elements,"Measures of Central Tendency refers to the typical value for a variable (single representative value). Geometric mean and Harmonic mean are used to summarize variables that represent the percentages and ratios respectively.", spacer=6)
    write_text(elements,"Measures of Dispersion show how spread out or varied the values in a dataset are from the central tendency value.", spacer=6)
    write_text(elements,"The below table shows the Measures of Central Tendency & Dispersion for all the variables:", spacer=6)
    add_wrapped_table_to_elements(elements,data['summary_df_2'])
    write_text(elements, "Data Distribution", style_name="Heading3", spacer=6)
    write_text(elements,"Quartiles are values that divide a variable into 4 equal parts each containing 25% of the data incrementally. ", spacer=6)
    write_text(elements,"Skewness refers to the lack of symmetricity in a variable.", spacer=6)
    write_text(elements,"The below table shows the Data distribution based on Quartiles and Skewness:", spacer=6)
    add_wrapped_table_to_elements(elements,data['summary_df_3'])
    

    write_text(elements, "Insights from the Numerical Summary:", style_name="Heading3", spacer=6) 
    write_text(elements,data['summary_info'],style_name='Normal',spacer=8)    
    write_text(elements, "Summary for Checking Skewness:", style_name="Heading3", spacer=6)
    write_text(elements,data['skew_info'], spacer=15)

    #elements.append(PageBreak())

    add_wrapped_table_to_elements(elements,data['outlier_summary'], extra_text="It is observed that certain variables had outliers, which could potentially distort analysis and the overall performance. To address this, outliers were capped using the Interquartile Range (IQR) method, ensuring the values remain within a reasonable range while preserving the overall distribution.", title="Outlier Detection")

    write_text(elements, data['outlier_interpretations'], style_name="Normal", spacer=6)

    elements.append(PageBreak())
    write_text(elements,'Categorical Data Analysis',style_name='Heading1',spacer=8)

    write_text(elements,"Here, we can understand the distribution of various classes present in a categorical variable and their frequency distribution.",style_name='Normal',spacer=8)

    
    
    write_text(elements, "Insights from the Frequency Distribution", style_name="Heading3", spacer=6) 
    write_text(elements,data['freq_summary'],style_name='Normal',spacer=8)
    #elements.append(PageBreak())
    elements.append(Spacer(1, 12))
    #elements.append(Spacer(1, 12))  # Optional spacing

    #elements.append(PageBreak())

    write_text(elements,'GroupBy tables',style_name='Heading3')
    write_text(elements,data['group_pivot_text'])
    write_text(elements,'Summary:',style_name='Heading3')
    write_text(elements, 'Based on the user input,', spacer=12)
    write_text(elements, data['grouped_df_i'], spacer =15)
    elements.append(PageBreak())

    box_table_text(elements,text="Data Relationship")

    write_text(elements, "Correlation Analysis", style_name="Heading3", spacer=6)
    write_text(elements,data['Correlation_text_1'], spacer=15)
    write_text(elements,data['Corr_text_2'], spacer=15)
    write_text(elements,data['Corr_text_3'], spacer=15)
    write_text(elements,data['Corr_text_4'], spacer=15)
    write_text(elements,data['Corr_text_5'], spacer=15)
    write_text(elements,data['Corr_text_6'], spacer=15)
    write_text(elements,data['Corr_text_7'], spacer=15)
    threshold = data.get('threshold')
    add_wrapped_table_to_elements(elements,data['correlation_table'], title=f"Correlation value between columns in the dataset.") #" Table with a threshold of {threshold}")
    write_text(elements,data['corr_interprtations'], spacer=15)


    elements.append(PageBreak())
    if data['flag'] == 1:
        box_table_text(elements,text="Feature Engineering")
        write_text(elements, "Description", style_name="Heading3", spacer=6)
        write_text(elements,"Feature Engineering is the process of transforming raw variables into meaningful features that better represent the underlying patterns to the algorithms, resulting in improved accuracy and performance.", spacer=15)
        
        write_text(elements, "Feature Transformation", style_name="Heading3", spacer=6)
        write_text(elements,"Feature Transformation involves the process of transforming the existing variables in the dataset to make them more suitable or relevant for the algorithms to understand.", spacer=15)
        write_text(elements, "<b>Recommendation:</b>", style_name="Heading3", spacer=6)
        write_text(elements,data['skew'], spacer=15)
        write_text(elements, "Action taken:", style_name="Heading3", spacer=6)
        write_text(elements,data['box'], spacer=15)

        write_text(elements, "Feature Extraction", style_name="Heading3", spacer=6)
        write_text(elements,"Feature Extraction is the process of extracting features from the existing feature in the dataset.", spacer=15)
        write_text(elements, "Action taken:", style_name="Heading3", spacer=6)
        write_text(elements,data['time_features'], spacer=15)

        write_text(elements, "Feature Scaling/ Normalization", style_name="Heading3", spacer=6)
        write_text(elements,"Feature Scaling is the process of transforming numerical features to be on a similar scale, so that no single feature dominates others due to differences in magnitude or units.", spacer=15)
        write_text(elements, "Action taken:", style_name="Heading3", spacer=6)
        write_text(elements,data['scale_i'], spacer=15)

        write_text(elements, "Feature Creation", style_name="Heading3", spacer=6)
        write_text(elements,"Feature Creation is a type of feature engineering where you generate new features from existing variables to enhance performance of the solution.", spacer=15)
        write_text(elements, "Action taken:", style_name="Heading3", spacer=6)
        write_text(elements,data['arith_i'], spacer=15)

        write_text(elements, "Feature Binning", style_name="Heading3", spacer=6)
        write_text(elements,"Feature Binning is the process of converting continuous numerical features into categorical bins or intervals.", spacer=15)
        write_text(elements, "Action taken:", style_name="Heading3", spacer=6)
        write_text(elements,data['binning'], spacer=15)
        
        


        write_text(elements, "Feature Labelling", style_name="Heading3", spacer=6)
        write_text(elements,"Feature Labelling the process of identifying and assigning meaningful labels (includes One-hot encoding and Label encoding) to the variables in a dataset. ", spacer=15)
        write_text(elements, "Action taken:", style_name="Heading3", spacer=6)
        write_text(elements,data['onehot'], spacer=15)
        write_text(elements,data['ordinal'], spacer=15)

        

        write_text(elements,data.get('end_feature', ''))
    
    write_text(elements,'End of Report', style_name='CenteredHeading1', spacer=12)
    
    elements.append(PageBreak()) 
   
    # ---- Build PDF ----
    doc.build(elements, onLaterPages=add_page_num, onFirstPage=add_page_num)
    print(f" PDF report saved to {filename}")
