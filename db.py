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
    
    -- 시퀀스 연결 보장 (SERIAL이 제대로 작동하도록)
    CREATE SEQUENCE IF NOT EXISTS processed_documents_id_seq;
    ALTER TABLE processed_documents ALTER COLUMN id SET DEFAULT nextval('processed_documents_id_seq');
    ALTER SEQUENCE processed_documents_id_seq OWNED BY processed_documents.id;
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            conn.commit()
    print("✅ DB 테이블이 준비되었습니다.")

def save_chunks_to_db(doc_id, filename, chunks):
    """추출된 청크들을 저장하기 전, 기존 동일 doc_id 데이터를 먼저 지웁니다."""
    delete_query = "DELETE FROM processed_documents WHERE doc_id = %s" # 중복 방지용 삭제
    insert_query = """
    INSERT INTO processed_documents (doc_id, title, content, metadata)
    VALUES (%s, %s, %s, %s)
    RETURNING id
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # 1. 기존 데이터 삭제 (중복 방지)
                cur.execute(delete_query, (doc_id,))
                
                # 2. 새로운 청크 저장
                inserted_ids = []
                for chunk in chunks:
                    cur.execute(insert_query, (
                        doc_id, 
                        filename, 
                        chunk.text, 
                        json.dumps(chunk.metadata)
                    ))
                    inserted_id = cur.fetchone()[0]
                    inserted_ids.append(inserted_id)
                conn.commit()
        print(f"✅ PostgreSQL 갱신 성공: {doc_id} ({len(chunks)}개 청크, IDs: {inserted_ids[:3]}...)")
    except Exception as e:
        print(f"❌ PostgreSQL 저장 실패: {e}")

if __name__ == "__main__":
    # 처음 한 번만 실행해서 테이블을 만듭니다.
    try:
        init_db()
        print("✅ 연결 및 초기화 성공!")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")