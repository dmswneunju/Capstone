import pymysql
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#DB연결
con = pymysql.connect(host='database-1.c7rig0vgsupo.ap-northeast-2.rds.amazonaws.com', 
                      user='root', 
                      password='capstone',
                      db='capstone_project', 
                      charset='utf8', 
                      cursorclass=pymysql.cursors.DictCursor) # 한글처리 (charset = 'utf8')

cur = con.cursor()

cur.execute("SELECT * FROM silver")

result = cur.fetchall()


#데이터프레임으로 변환
result = pd.DataFrame(result)

# 코사인 유사도    
# category 컬럼의 값을 쉼표로 분할하여 리스트로 변환
result['category'] = result['category'].str.split(', ')
# province 컬럼을 리스트로 변환
result['province'] = result['province'].apply(lambda x: [x])

# 병원 진료과목과 병원 위치를 토큰화
def create_soup(x):
    return ' '.join(x['category']) + ' ' + ' '.join(x['province'])

result['soup'] = result.apply(create_soup, axis=1)


count = CountVectorizer()
count_matrix = count.fit_transform(result['soup']) #1372개의 rwo에 대해서 52개의 단어(= 카테고리, 행정구역)를 가지는 matirx 생성, 그 중 0이아닌 유효한 데이터가 3487개가 있다.

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

#병원 이름 중복 제거
indices = pd.Series(result.index, index=result['name']).drop_duplicates()


# 병원이름을 입력받으면 코사인 유사도를 통해서 가장 유사도가 높은 상위 5개의 병원 리스트 반환
def get_recommendations(name, cosine_sim=cosine_sim2):
    # 병원이름을 통해서 전체 데이터 기준 그 영화의 index 값을 얻기
    idx = indices[name]
    
    # 코사인 유사도 매트릭스 (cosine_sim) 에서 idx 에 해당하는 데이터를 (idx, 유사도) 형태로 얻기
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # 코사인 유사도 기준으로 내림차순 정렬
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # 자기 자신을 제외한 5개의 추천 병원을 슬라이싱
    sim_scores = sim_scores[1:6]
    
    # 추천 병원 목록 10개의 인덱스 정보 추출
    hospital_indices = [i[0] for i in sim_scores]
    
    # 인덱스 정보를 통해 병원 이름 추출
    return result['name'].iloc[hospital_indices]



def update_recommend(df_hospital_name):
    # 기존 데이터 리셋
    # 기존 추천 결과 리스트 전부 삭제 (선택 사항)
    delete_query = "DELETE FROM recommended_hospitals"
    cur.execute(delete_query)
    con.commit()

    # id 초기화(선택 사항)
    alter_table_query ="""ALTER TABLE recommended_hospitals AUTO_INCREMENT=1"""
    cur.execute(alter_table_query)
    
    
    names = df_hospital_name.tolist()
    
    for name in names:
        original_hospital = name
        recommend_hopitals5 = get_recommendations(original_hospital, cosine_sim2)
        
        # 새로운 추천 결과 삽입
        insert_query = "INSERT INTO recommended_hospitals (original_hospital, recommended_hospital) VALUES (%s, %s)"
        for recommended_hospital in recommend_hopitals5:
            cur.execute(insert_query, (original_hospital, recommended_hospital))
        con.commit()

result = update_recommend(result['name'])
