# Capstone Design
# 프로젝트 주제 : 요양병원 추천 시스템 구현
구현한 기능
- 요양병원 추천
  1. 공공데이터포털에서 요양병원 api를 AWS에 저장한다.
  2. AWS와 연동한 DB를 pymysql을 이용하여 연동한다.
  3. AWS에 있는 데이터를 가져와 추천시스템을 구현한다.
  4. 병원의 진료과목명과 병원이 속한 행정구역을 숫자로 바꾸어 토큰화한다. sckit-learn의 CountVectorizer사용.
  5. 유사한 병원을 추천하기위해 sckit-learn의 cosine_similarity사용.
  6. 병원이름을 입력받으면 코사인 유사도를 통해서 가장 유사도가 높은 상위 5개의 병원 리스트 반환한다.
  7. 추천 병원 리스트를 DB에 다시 저장한다. 
