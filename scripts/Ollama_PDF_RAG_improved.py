# Ollama_PDF_RAG_improved.py
"""
개선된 PDF RAG 시스템
- 파일 해시 기반 캐시 시스템
- 검색 파라미터 조정 가능
- 향상된 에러 처리 및 로깅
- 클래스 기반 구조
"""

import gradio as gr
import ollama
import os
import hashlib
import json
import time
from pathlib import Path
from typing import Optional, List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document


class Config:
    """설정 관리 클래스"""
    VECTOR_CACHE_DIR = "chroma_pdf_cache"
    CACHE_METADATA_FILE = "cache_metadata.json"
    DEFAULT_CHUNK_SIZE = 1000
    DEFAULT_CHUNK_OVERLAP = 200
    DEFAULT_EMBEDDING_MODEL = "mxbai-embed-large"
    DEFAULT_LLM_MODEL = "llama3"
    DEFAULT_SEARCH_K = 4
    DEFAULT_SEARCH_SCORE_THRESHOLD = 0.0


class CacheManager:
    """캐시 관리 클래스"""
    
    def __init__(self, cache_dir: str = Config.VECTOR_CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.metadata_file = self.cache_dir / Config.CACHE_METADATA_FILE
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """캐시 메타데이터 로드"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ 메타데이터 로드 실패: {e}")
                return {}
        return {}
    
    def _save_metadata(self):
        """캐시 메타데이터 저장"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ 메타데이터 저장 실패: {e}")
    
    def get_file_hash(self, file_path: str) -> str:
        """파일 해시 계산"""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            raise ValueError(f"파일 해시 계산 실패: {e}")
    
    def get_cache_path(self, file_hash: str) -> Path:
        """캐시 경로 반환"""
        return self.cache_dir / file_hash
    
    def is_cached(self, file_path: str) -> bool:
        """파일이 캐시되어 있는지 확인"""
        try:
            file_hash = self.get_file_hash(file_path)
            cache_path = self.get_cache_path(file_hash)
            
            # 메타데이터에 존재하고 디렉토리가 있는지 확인
            if file_hash in self.metadata and cache_path.exists():
                # 파일 수정 시간 확인
                cached_mtime = self.metadata[file_hash].get('mtime', 0)
                current_mtime = os.path.getmtime(file_path)
                if cached_mtime == current_mtime:
                    return True
            return False
        except Exception as e:
            print(f"⚠️ 캐시 확인 중 오류: {e}")
            return False
    
    def get_cached_info(self, file_path: str) -> Optional[Dict]:
        """캐시된 파일 정보 반환"""
        try:
            file_hash = self.get_file_hash(file_path)
            return self.metadata.get(file_hash)
        except Exception:
            return None
    
    def save_cache_info(self, file_path: str, cache_path: Path, chunk_count: int):
        """캐시 정보 저장"""
        try:
            file_hash = self.get_file_hash(file_path)
            file_name = os.path.basename(file_path)
            file_size = os.path.getsize(file_path)
            mtime = os.path.getmtime(file_path)
            
            self.metadata[file_hash] = {
                'file_name': file_name,
                'file_path': file_path,
                'file_size': file_size,
                'mtime': mtime,
                'cache_path': str(cache_path),
                'chunk_count': chunk_count,
                'created_at': time.time()
            }
            self._save_metadata()
        except Exception as e:
            print(f"⚠️ 캐시 정보 저장 실패: {e}")
    
    def list_cached_files(self) -> List[Dict]:
        """캐시된 파일 목록 반환"""
        return list(self.metadata.values())
    
    def clear_cache(self, file_hash: Optional[str] = None):
        """캐시 삭제"""
        if file_hash:
            if file_hash in self.metadata:
                cache_path = Path(self.metadata[file_hash]['cache_path'])
                if cache_path.exists():
                    import shutil
                    shutil.rmtree(cache_path)
                del self.metadata[file_hash]
                self._save_metadata()
        else:
            # 전체 캐시 삭제
            import shutil
            for file_hash, info in list(self.metadata.items()):
                cache_path = Path(info['cache_path'])
                if cache_path.exists():
                    shutil.rmtree(cache_path)
            self.metadata = {}
            self._save_metadata()


class PDFRAGSystem:
    """PDF RAG 시스템 메인 클래스"""
    
    def __init__(
        self,
        embedding_model: str = Config.DEFAULT_EMBEDDING_MODEL,
        llm_model: str = Config.DEFAULT_LLM_MODEL,
        chunk_size: int = Config.DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = Config.DEFAULT_CHUNK_OVERLAP
    ):
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.cache_manager = CacheManager()
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def load_pdf(self, file_path: str, use_cache: bool = True) -> tuple:
        """
        PDF 로드 및 벡터화
        Returns: (retriever, is_cached, info_dict)
        """
        start_time = time.time()
        
        try:
            # 파일 존재 확인
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
            
            # 캐시 확인
            if use_cache and self.cache_manager.is_cached(file_path):
                print(f"✅ 캐시에서 로드: {os.path.basename(file_path)}")
                cached_info = self.cache_manager.get_cached_info(file_path)
                file_hash = self.cache_manager.get_file_hash(file_path)
                cache_path = self.cache_manager.get_cache_path(file_hash)
                
                # 기존 벡터 저장소 로드
                vectorstore = Chroma(
                    persist_directory=str(cache_path),
                    embedding_function=self.embeddings
                )
                retriever = vectorstore.as_retriever()
                
                load_time = time.time() - start_time
                info = {
                    'cached': True,
                    'load_time': load_time,
                    'chunk_count': cached_info.get('chunk_count', 0),
                    'file_name': cached_info.get('file_name', '')
                }
                return retriever, True, info
            
            # 새로 처리
            print(f"📄 PDF 처리 중: {os.path.basename(file_path)}")
            loader = PyMuPDFLoader(file_path)
            docs = loader.load()
            
            if not docs:
                raise ValueError("PDF에서 텍스트를 추출할 수 없습니다.")
            
            print(f"✅ {len(docs)} 페이지 로드 완료")
            
            # 텍스트 분할
            splits = self.text_splitter.split_documents(docs)
            print(f"✅ {len(splits)}개 청크로 분할 완료")
            
            # 벡터화 및 저장
            file_hash = self.cache_manager.get_file_hash(file_path)
            cache_path = self.cache_manager.get_cache_path(file_hash)
            
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                persist_directory=str(cache_path)
            )
            vectorstore.persist()
            
            # 캐시 정보 저장
            self.cache_manager.save_cache_info(
                file_path, cache_path, len(splits)
            )
            
            retriever = vectorstore.as_retriever()
            load_time = time.time() - start_time
            
            info = {
                'cached': False,
                'load_time': load_time,
                'chunk_count': len(splits),
                'file_name': os.path.basename(file_path)
            }
            return retriever, False, info
            
        except Exception as e:
            raise Exception(f"PDF 로드 실패: {str(e)}")
    
    def search_documents(
        self,
        retriever,
        question: str,
        k: int = Config.DEFAULT_SEARCH_K,
        score_threshold: float = Config.DEFAULT_SEARCH_SCORE_THRESHOLD
    ) -> List[Document]:
        """문서 검색"""
        try:
            # 검색 타입 설정 (similarity 또는 mmr)
            search_kwargs = {
                'k': k,
                'score_threshold': score_threshold
            }
            retriever.search_kwargs = search_kwargs
            
            retrieved_docs = retriever.invoke(question)
            return retrieved_docs
        except Exception as e:
            raise Exception(f"문서 검색 실패: {str(e)}")
    
    def format_docs(self, docs: List[Document]) -> str:
        """문서 포맷팅"""
        if not docs:
            return ""
        return "\n\n".join([
            f"[문서 {i+1}]\n{doc.page_content}" 
            for i, doc in enumerate(docs)
        ])
    
    def generate_answer(
        self,
        question: str,
        context: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """답변 생성"""
        try:
            if not context:
                return "❌ 관련 문서를 찾을 수 없습니다. 질문을 더 구체적으로 작성해 보세요."
            
            default_system_prompt = (
                "You are a helpful assistant. Read the PDF content and answer the question. "
                "Translate the answer in Korean with emoji. Be accurate and cite sources when possible."
            )
            
            system_content = system_prompt or default_system_prompt
            user_prompt = f"Question: {question}\n\nContext:\n{context}"
            
            response = ollama.chat(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            return response['message']['content']
        except Exception as e:
            raise Exception(f"답변 생성 실패: {str(e)}")
    
    def process_query(
        self,
        file_path: str,
        question: str,
        k: int = Config.DEFAULT_SEARCH_K,
        score_threshold: float = Config.DEFAULT_SEARCH_SCORE_THRESHOLD,
        use_cache: bool = True
    ) -> Dict:
        """전체 RAG 파이프라인 실행"""
        start_time = time.time()
        result = {
            'answer': '',
            'error': None,
            'info': {}
        }
        
        try:
            # PDF 로드
            retriever, is_cached, load_info = self.load_pdf(file_path, use_cache)
            result['info']['load'] = load_info
            
            # 문서 검색
            search_start = time.time()
            retrieved_docs = self.search_documents(
                retriever, question, k, score_threshold
            )
            search_time = time.time() - search_start
            
            if not retrieved_docs:
                result['answer'] = "관련 문서를 찾을 수 없습니다. 질문을 더 구체적으로 작성해 보세요."
                result['info']['search'] = {
                    'time': search_time,
                    'doc_count': 0
                }
                return result
            
            # 컨텍스트 구성
            context = self.format_docs(retrieved_docs)
            
            # 답변 생성
            gen_start = time.time()
            answer = self.generate_answer(question, context)
            gen_time = time.time() - gen_start
            
            total_time = time.time() - start_time
            
            result['answer'] = answer
            result['info']['search'] = {
                'time': search_time,
                'doc_count': len(retrieved_docs)
            }
            result['info']['generation'] = {
                'time': gen_time
            }
            result['info']['total_time'] = total_time
            
            return result
            
        except Exception as e:
            result['error'] = str(e)
            result['answer'] = f"❌ 오류 발생: {str(e)}"
            return result


# Gradio 인터페이스
def create_interface():
    """Gradio 인터페이스 생성"""
    
    # 기본 RAG 시스템 인스턴스
    rag_system = PDFRAGSystem()
    
    def process_rag(
        file,
        question: str,
        llm_model: str,
        embedding_model: str,
        search_k: int,
        use_cache: bool
    ) -> str:
        """RAG 처리 함수"""
        if not file:
            return "❌ PDF 파일을 업로드해주세요."
        
        if not question or not question.strip():
            return "❌ 질문을 입력해주세요."
        
        try:
            # 모델이 변경된 경우 새 인스턴스 생성
            if (rag_system.llm_model != llm_model or 
                rag_system.embedding_model != embedding_model):
                rag_system.llm_model = llm_model
                rag_system.embedding_model = embedding_model
                rag_system.embeddings = OllamaEmbeddings(model=embedding_model)
            
            # RAG 처리
            result = rag_system.process_query(
                file_path=file.name,
                question=question,
                k=search_k,
                use_cache=use_cache
            )
            
            if result['error']:
                return result['answer']
            
            # 결과 포맷팅
            answer = result['answer']
            info = result['info']
            
            # 정보 추가
            info_text = "\n\n---\n📊 처리 정보:\n"
            if 'load' in info:
                load_info = info['load']
                cache_status = "✅ 캐시 사용" if load_info.get('cached') else "🔄 새로 처리"
                info_text += f"- {cache_status}\n"
                info_text += f"- 로드 시간: {load_info.get('load_time', 0):.2f}초\n"
                info_text += f"- 청크 수: {load_info.get('chunk_count', 0)}개\n"
            
            if 'search' in info:
                search_info = info['search']
                info_text += f"- 검색 시간: {search_info.get('time', 0):.2f}초\n"
                info_text += f"- 검색된 문서: {search_info.get('doc_count', 0)}개\n"
            
            if 'generation' in info:
                gen_info = info['generation']
                info_text += f"- 생성 시간: {gen_info.get('time', 0):.2f}초\n"
            
            if 'total_time' in info:
                info_text += f"- 총 처리 시간: {info.get('total_time', 0):.2f}초\n"
            
            return answer + info_text
            
        except Exception as e:
            return f"❌ 오류 발생: {str(e)}"
    
    def list_cached_files():
        """캐시된 파일 목록 반환"""
        try:
            cached_files = rag_system.cache_manager.list_cached_files()
            if not cached_files:
                return "캐시된 파일이 없습니다."
            
            result = "📁 캐시된 파일 목록:\n\n"
            for i, file_info in enumerate(cached_files, 1):
                result += f"{i}. {file_info.get('file_name', 'Unknown')}\n"
                result += f"   - 청크 수: {file_info.get('chunk_count', 0)}개\n"
                result += f"   - 파일 크기: {file_info.get('file_size', 0) / 1024:.2f} KB\n"
                created_at = file_info.get('created_at', 0)
                if created_at:
                    from datetime import datetime
                    dt = datetime.fromtimestamp(created_at)
                    result += f"   - 생성 시간: {dt.strftime('%Y-%m-%d %H:%M:%S')}\n"
                result += "\n"
            
            return result
        except Exception as e:
            return f"❌ 오류: {str(e)}"
    
    def clear_cache():
        """캐시 삭제"""
        try:
            rag_system.cache_manager.clear_cache()
            return "✅ 캐시가 모두 삭제되었습니다."
        except Exception as e:
            return f"❌ 오류: {str(e)}"
    
    # 인터페이스 구성
    with gr.Blocks(title="개선된 PDF RAG 시스템") as iface:
        gr.Markdown("# 📚 개선된 PDF RAG 시스템")
        gr.Markdown("PDF 파일을 업로드하고 질문하면, AI가 문서 내용을 기반으로 답변합니다.")
        
        with gr.Row():
            with gr.Column(scale=2):
                file_input = gr.File(
                    label="📄 PDF 파일 업로드",
                    type="filepath",
                    file_types=[".pdf"]
                )
                
                question_input = gr.Textbox(
                    label="❓ 질문을 입력하세요",
                    placeholder="예: 이 문서의 주요 내용은 무엇인가요?",
                    lines=3
                )
                
                with gr.Accordion("⚙️ 고급 설정", open=False):
                    llm_model = gr.Dropdown(
                        label="LLM 모델",
                        choices=["llama3", "llama3.1", "llama3.2", "mistral", "qwen2"],
                        value=Config.DEFAULT_LLM_MODEL,
                        info="사용할 언어 모델을 선택하세요"
                    )
                    
                    embedding_model = gr.Dropdown(
                        label="임베딩 모델",
                        choices=["mxbai-embed-large", "nomic-embed-text"],
                        value=Config.DEFAULT_EMBEDDING_MODEL,
                        info="사용할 임베딩 모델을 선택하세요"
                    )
                    
                    search_k = gr.Slider(
                        label="검색 결과 개수 (k)",
                        minimum=1,
                        maximum=10,
                        value=Config.DEFAULT_SEARCH_K,
                        step=1,
                        info="검색할 문서 청크의 개수"
                    )
                    
                    use_cache = gr.Checkbox(
                        label="캐시 사용",
                        value=True,
                        info="이전에 처리한 PDF는 캐시에서 재사용합니다"
                    )
                
                submit_btn = gr.Button("🚀 질문하기", variant="primary", size="lg")
            
            with gr.Column(scale=3):
                output = gr.Textbox(
                    label="💬 답변",
                    lines=15,
                    max_lines=20
                )
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📋 캐시 관리")
                cache_list_btn = gr.Button("📁 캐시된 파일 목록 보기")
                cache_clear_btn = gr.Button("🗑️ 캐시 모두 삭제", variant="stop")
                cache_output = gr.Textbox(
                    label="캐시 정보",
                    lines=10
                )
        
        # 이벤트 연결
        submit_btn.click(
            fn=process_rag,
            inputs=[file_input, question_input, llm_model, embedding_model, search_k, use_cache],
            outputs=output
        )
        
        cache_list_btn.click(
            fn=list_cached_files,
            outputs=cache_output
        )
        
        cache_clear_btn.click(
            fn=clear_cache,
            outputs=cache_output
        )
        
        # 예제
        gr.Markdown("### 💡 사용 팁")
        gr.Markdown("""
        - 같은 PDF를 여러 번 질문할 때는 캐시를 사용하면 빠르게 처리됩니다
        - 검색 결과 개수(k)를 조정하여 더 많은 또는 더 적은 컨텍스트를 사용할 수 있습니다
        - 질문을 구체적으로 작성할수록 더 정확한 답변을 받을 수 있습니다
        """)
    
    return iface


if __name__ == "__main__":
    print("🚀 개선된 PDF RAG 시스템 시작 중...")
    print(f"📁 캐시 디렉토리: {Config.VECTOR_CACHE_DIR}")
    
    iface = create_interface()
    iface.launch()

