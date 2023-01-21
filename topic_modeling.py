import pandas as pd
import re
import konlpy
import os
import glob
from konlpy.tag import Okt
from ckonlpy.tag import Twitter
from ckonlpy.tag import Postprocessor
from ckonlpy.utils import load_wordset
from wordcloud import WordCloud
from collections import Counter
import matplotlib.pyplot as plt

###Topic analysis를 위한 모듈
import pprint
import numpy as np
from gensim.models import LdaSeqModel
from gensim import corpora, models
import gensim
from gensim.models import CoherenceModel
from gensim.utils import simple_preprocess
from gensim.models.ldamodel import LdaModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis


# working directory 설정
os.chdir('C:/Users/hp/Desktop/COSS_NLP')


# 불용어 리스트 txt 파일 불러오기
with open('./korean_stopwords.txt', 'r', encoding="UTF8") as file:
    stop_words = file.read().replace('\n', ' ')

stop_words = set(stop_words.split(' '))
stop_words = [a for a in stop_words if a != '']

# 불용어 리스트에 불용어 추가
stop_words.extend(['미디어', '교육', '학교', '정회원', '중학교',
                       '초등학교', '고등학교', 'ㅋㅋ', '는', '다', '적', '있', '고', '은', '되', '본', '였', '성',
                       'the', 'of', '기', '게', '한다', '대한', '재', '라고', '지',
                       '점', '않', '도', '화', '나타났', '절', '었', '므로', '면',
                       '해', '된', '이러', '위한', '시키', '장', '된다', '중', '다는',
                       '도록', '는데', '간', '될', '대', '더', '으며', '같', '면서',
                       '며', '여러', '는지', '했', '때문', '이나', '보다', '았', '함',
                       '통해', '음', '률', '위해', '던', '될지', '데', '어서', '개',
                       '라는', '해야', '어떠', '또는', '던', '다고', '으나', '거나',
                       '는가', '여야', '였으며', '인해', '어야', '겠', '려는',
                       '됨', '는다', '으므로', '계', '미', '으로부터', '졌', '마다',
                       '임', '용', '그것', '어떻', '코', '란', '한다고', '된다는', '져',
                       '환', '려고', '려', '그렇', '였으나', '다면', '으면', '아서', '요',
                       '본다면', '할지', '느냐', '어도', '했으며', '타', '였음을',
                       '으려는', '롭', '토록', '라기', '납', '즈', '으면서', '관해서',
                       '는다고', '싱', '했었으나', '쿤', '왔으나', '째', '으', '떻게',
                       '브', '인데', '는다는', '왔었', '닝', '옥', '왔으며', '다는데',
                       '상', '비', '업', '테', '먼', '트', '세', '담', '보인다',
                       '워', '정', '공', '별', '당', '을수록', '형', '신', '잘', '가지',
                       '위하', '부를', '율', '고자', '가장', '주', '살펴보', '미치',
                       '차'])



# 파일 로딩 + 특수문자 제거
path = './data/2018~22_재단 발행 미디어교육 전문지_pdf_and_txt/2018~22_재단 발행 미디어교육 전문지_pdf_and_txt/'
years = ['2018', '2019', '2020', '2021', '2022']

for year in years:
    file_pathes = glob.glob(path + year +'/*.txt')
    for file in file_pathes:
        with open(file, 'r', encoding="UTF8") as f:
            texts = f.read().replace('\n', '')            
            texts = re.sub('[^0-9a-zA-Zㄱ-ㅎ가-힣]', ' ', texts)

# 명사 추출
    okt = Okt()
    nouns = okt.nouns(texts)

# 불용어 제거
    cleaned_nouns = [word for word in nouns if not word in stop_words] 

# 빈도수 높은 명사 출력
#     count = Counter(cleaned_nouns)
#     count_top100 = count.most_common(100)
#     for v in count_top100:
#         print(v)

# # 워드클라우드 출력 및 저장    
#     wordcloud = WordCloud(font_path='./font/malgun.ttf', width=800, height=800, background_color='white')
#     wordcloud = wordcloud.generate_from_frequencies(dict(count_top100))
#     array = wordcloud.to_array()
#     fig = plt.figure(figsize=(10, 10))
#     plt.imshow(array, interpolation='bilinear')
#     plt.axis('off')
#     plt.title(year, fontsize=30)

#     save_path = './wordcloud/' + year + ".png"
#     plt.savefig(save_path)
#     plt.show()

# topic model 
    texts = cleaned_nouns
    id2word = corpora.Dictionary([texts])
    corpus = [id2word.doc2bow([text]) for text in texts]

    np.random.seed(0)  # this is for representaion of result
# LDA
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=3, random_state=100, update_every=1,
                                                chunksize=100, passes=10, alpha='auto', per_word_topics=True)


    print("The Result of " + year)
    for topic_topwords in lda_model.print_topics():
        print(topic_topwords)