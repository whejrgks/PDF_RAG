# PDF RAG 프로젝트

PDF 문서를 기반으로 한 RAG (Retrieval-Augmented Generation) 시스템입니다.

## 📁 프로젝트 구조

```
PDF_RAG/
├── scripts/              # Python 스크립트 파일들
├── documents/            # PDF 문서 파일들
├── docs/                 # 프로젝트 문서
├── config/               # 설정 파일
└── chroma_pdf_cache/     # 벡터 캐시 디렉토리
```

## 🚀 빠른 시작

1. **의존성 설치**
   ```bash
   pip install -r config/requirement.txt
   ```

2. **Ollama 모델 설치**
   ```bash
   ollama pull llama3
   ollama pull mxbai-embed-large
   ```

3. **프로그램 실행**
   ```bash
   python scripts/Ollama_PDF_RAG_improved.py
   ```

## 📖 상세 문서

자세한 사용법과 기능 설명은 [docs/README.md](docs/README.md)를 참고하세요.
