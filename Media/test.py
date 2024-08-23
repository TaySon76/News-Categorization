import requests
from PyPDF2 import PdfReader
from io import BytesIO

def extract_text_from_pdf_url(pdf_url):
    data_d, data_l = {}, []
    temp = [76, 276, 353, 431, 582, 649, 684, 749, 806, 833, 870]
    lines = ""
    response = requests.get(pdf_url)
    with BytesIO(response.content) as file:
        pdf_reader = PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            if "RSS URLs" not in page.extract_text() and "country category rss-url" not in page.extract_text():
                lines += page.extract_text()
        data_l = lines.split('\n')
        for i in temp:
            data_l[i-1] += data_l[i]
       
        data_l = [line for idx, line in enumerate(data_l) if idx not in temp]

        for j in data_l:
            space_indexes = [i for i, char in enumerate(j) if char == " "]

            if len(space_indexes) > 2:
                temp = j[:space_indexes[1]+1] + j[space_indexes[1]+1:].replace(" ", "")
                j = temp

            space_indexes = space_indexes[:2]

            if len(space_indexes) == 2:
                data_d[j[space_indexes[1]+1:len(j)]] = (j[0:space_indexes[0]], j[space_indexes[0]+1:space_indexes[1]])

    return data_d

pdf_data = extract_text_from_pdf_url('https://about.fb.com/wp-content/uploads/2016/05/rss-urls-1.pdf')
for key in pdf_data:
    print(f"  - Country: {pdf_data[key][1]}, Category: {pdf_data[key][0]}, Link: {key}")
