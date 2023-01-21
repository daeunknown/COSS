import io
import glob
import re
import PyPDF2
import os
from pdf2image import convert_from_path
from google.cloud import vision

os.chdir('C:/Users/hp/Desktop/COSS_NLP/OCR') # working directory 설정

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './cossbigdata-298e5e79d9ad.json'
client = vision.ImageAnnotatorClient()


if __name__ == '__main__':
    file_path_list = glob.glob('./data/교육자료들_수정/*.pdf', recursive=True) # 최하위 디렉토리의 모든 pdf파일 경로들을 리스트 형태로 저장
    print(file_path_list)
    
    # 모든 pdf 파일 순회 
    for file_path in file_path_list:
        png_path = file_path.replace(".pdf", "")
        os.makedirs(png_path, exist_ok=True) # pdf별 이미지 파일이 들어갈 디렉토리 생성
        reader = PyPDF2.PdfFileReader(file_path)
        imaged_pages = convert_from_path(file_path) # 각 페이지를 image로 변환

        # PDF To Image
        for i, imaged_page in enumerate(imaged_pages):
            save_file_path = png_path + "/" + str(i + 1) + '.png'
            imaged_page.save(save_file_path, 'PNG') # PDF와 동일한 이름의 폴더 생성 후 그 안에 png 형태로 저장
        

        # Image To TXT
        page_paths = glob.glob(file_path.replace(".pdf", "/*.png")) # png 파일들 리스트를 생성
        for page_path in page_paths:
            print(page_path)
            client = vision.ImageAnnotatorClient() # Google cloud platform (GCP) 사용 계정

            # 파일을 open
            with io.open(page_path, 'rb') as image_file:
                content = image_file.read()

            image = vision.Image(content=content) # GCP에서 편집 가능한 상태로 변환
            context = vision.ImageContext(language_hints = ['en', 'ko']) # ImageContext method 사용. 영어, 한국어
            response = client.document_text_detection(image=image, image_context = context) # google OCR에 image를 전송 + 응답 받음
            texts = response.text_annotations
            
            try:
                print(texts[0].description)
                text = texts[0].description # text 부분만 추출
                save_file_path = file_path.replace(".pdf", ".txt") # PDF와 동일한 이름을 가진 TXT 파일 경로 설정
                save_file = open(save_file_path, 'a', encoding="utf-8") # TXT 파일 생성(append mode)
                save_file.write(text) # 파일 수정
                save_file.close() # Close
                
            except: # Error가 생겼을때는 "ERROR"라고 출력하고 다음 파일로 이동
                print("ERROR")
                continue

