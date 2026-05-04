# HR RAG Chatbot

Bu proje, Week 6 bootcamp ödevi kapsamında geliştirilen kısa süreli hafızaya
sahip bir RAG chatbot uygulamasıdır. Sistem HR dokümanlarını okuyup ChromaDB
içine aktarır, kullanıcı sorularında ilgili doküman parçalarını getirir ve
LangChain agent yapısı ile kısa, kaynaklı cevaplar üretir.

## Özellikler

- DOCX, PDF ve TXT formatındaki HR dokümanlarını yükleme
- `RecursiveCharacterTextSplitter` ile 500 karakterlik chunk üretme
- Her chunk için 13 metadata alanı ekleme
- ChromaDB ile vektör tabanlı arama
- LangChain `create_agent` ile tool kullanan RAG agent
- PostgreSQL destekli `PostgresSaver` ile kısa süreli konuşma hafızası
- CLI üzerinden ingest, chat ve test akışı
- OpenRouter free router ile ücretsiz chat modeli kullanımı

## Ücretsiz Kullanım Notu

OpenRouter tarafında chat modeli özellikle free router olarak ayarlandı:

```text
openrouter/free
```

Embedding tarafında da OpenRouter üzerindeki ücretsiz model kullanılıyor:

```text
nvidia/llama-nemotron-embed-vl-1b-v2:free
```

Kodda chat ve embedding modellerinin free kullanımda kalması için kontrol
eklenmiştir.

OpenRouter free endpoint'lerinde daha düşük rate limit ve veri işleme/loglama
politikaları olabilir. Bu nedenle gerçek, gizli veya kişisel HR verileriyle
kullanılmamalıdır.

## Proje Yapısı

```text
hr_rag_chatbot/
├── document_loader.py
├── vector_store.py
├── rag_agent.py
├── main.py
├── requirements.txt
├── .env.example
├── docker-compose.yml
├── README.md
└── hr_documents_pack/
    └── initial_docs/
```

## Teknolojiler

- Python 3.12
- LangChain 1.2+
- ChromaDB
- PostgreSQL
- Docker
- OpenRouter

## Kurulum

Python 3.10 veya üzeri gereklidir. Bu projede Python 3.12 kullanıldı.

```bash
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

PostgreSQL servisini Docker ile başlat:

```bash
docker compose up -d postgres
```

`.env.example` dosyasını örnek alarak `.env` dosyasını oluştur:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
DB_URI=postgresql://postgres:postgres@localhost:5432/hr_rag
```

API key paylaşılmamalı ve repoya eklenmemelidir.

## Kullanım

Önce dokümanları ChromaDB'ye yükle:

```bash
python main.py ingest --reset
```

Chat modunu başlat:

```bash
python main.py chat
```

Ödev test sorularını çalıştır:

```bash
python main.py test
```

## Test Soruları

`python main.py test` komutu şu soruları çalıştırır:

```text
What is the company's leave policy?
How many vacation days do employees get?
What are the steps in the offboarding process?
What are the IT security requirements for new employees?
What is the performance review process?
How do I submit travel expenses for reimbursement?
```

Kısa süreli hafıza kontrolü için:

```text
What is the leave policy?
What about sick leave?
```

İkinci soruda agent'ın önceki konuşma bağlamını kullanması beklenir.

## Metadata Alanları

Her chunk için aşağıdaki metadata alanları eklenir:

```text
file_name
file_extension
file_size_bytes
character_count
chunk_index
chunk_size
chunk_overlap
document_type
creation_date
last_modified
ingestion_timestamp
page_number
section_title
```

## Durum

Kod yapısı tamamlandı. Doküman yükleme, ChromaDB ingest akışı ve PostgreSQL
checkpointer kurulumu doğrulandı. OpenRouter API key eklendikten sonra gerçek
chat modeliyle `python main.py test` çalıştırılarak son kontrol yapılmalıdır.
