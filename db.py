import psycopg2
from psycopg2.extras import RealDictCursor
import json

# 1. DB 접속 정보 (따로 관리하면 편합니다)
DB_CONFIG = {
    "host": "localhost",
    "database": "postgres",
    "user": "postgres",
    "password": "1111", # 설치 시 설정한 비번
    "port": "5432"
}

def get_connection():
    """DB 연결 객체를 반환합니다."""
    return psycopg2.connect(**DB_CONFIG)

def init_db():
    """문서 데이터를 저장할 테이블을 생성합니다."""
    query = """
    CREATE TABLE IF NOT EXISTS processed_documents (
        id SERIAL PRIMARY KEY,
        doc_id TEXT,             -- SOP 문서 번호 (예: EQ-SOP-001)
        title TEXT,              -- 문서 제목
        content TEXT,            -- 추출된 마크다운 텍스트
        metadata JSONB,          -- 목차 경로, 페이지 번호 등
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            conn.commit()
    print("✅ DB 테이블이 준비되었습니다.")

def save_chunks_to_db(doc_id, filename, chunks):
    """추출된 청크들을 PostgreSQL에 저장합니다."""
    query = """
    INSERT INTO processed_documents (doc_id, title, content, metadata)
    VALUES (%s, %s, %s, %s)
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                for chunk in chunks:
                    # chunk 객체에서 텍스트와 메타데이터 추출
                    cur.execute(query, (
                        doc_id, 
                        filename, 
                        chunk.text, 
                        json.dumps(chunk.metadata)
                    ))
                conn.commit()
        print(f"✅ PostgreSQL 저장 성공: {len(chunks)}개 청크")
    except Exception as e:
        print(f"❌ PostgreSQL 저장 실패: {e}")

if __name__ == "__main__":
    # 처음 한 번만 실행해서 테이블을 만듭니다.
    try:
        init_db()
        print("✅ 연결 및 초기화 성공!")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")