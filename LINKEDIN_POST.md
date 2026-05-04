# LinkedIn Paylaşım Taslağı

Kısa süreli hafızaya sahip bir RAG chatbot geliştirdim.

Bu projede HR dokümanlarını okuyup ChromaDB içine aktaran, kullanıcı sorularına
doküman bağlamıyla cevap veren ve PostgreSQL tabanlı checkpointer ile konuşma
geçmişini koruyan uçtan uca bir CLI uygulaması kurdum.

Kullandığım teknolojiler:

- Python 3.12
- LangChain `create_agent`
- ChromaDB
- PostgreSQL + `PostgresSaver`
- Docker
- OpenRouter free chat router
- OpenRouter free embedding model

Bu çalışmada özellikle doküman yükleme, metadata yönetimi, vector search, agent
tool kullanımı ve konuşma hafızası tarafını uygulamalı olarak deneyimledim.

RAG mimarisini sadece teoride değil, çalışan bir akış üzerinden kurmak benim
için çok öğretici bir pratik oldu.

#Python #LangChain #RAG #ChromaDB #PostgreSQL #Docker #LLM #AI
