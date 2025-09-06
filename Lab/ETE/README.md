# Advanced Domain-Specific QA Chatbot

A comprehensive Retrieval-Augmented Generation (RAG) pipeline with multi-turn conversational capabilities, evaluation metrics, and interactive visualizations built with Streamlit.

## Assignment Requirements Coverage

This implementation addresses all evaluation criteria for the Domain-Specific Question Answering & Chatbot assignment:

[✓] **Domain Knowledge Base Preparation**
- Advanced document preprocessing with comprehensive analysis
- Intelligent text chunking with statistical evaluation
- Support for multiple document formats

[✓] **Question Answering System**
- Retrieval model using sentence transformers
- Generation using state-of-the-art LLMs via Groq API
- Proper RAG pipeline integration

[✓] **Chatbot Integration**
- Multi-turn conversation capabilities
- Conversation memory and context awareness
- Coherent response generation

[✓] **Evaluation Metrics**
- Retrieval performance evaluation (similarity scores)
- Generation quality metrics (ROUGE, BLEU)
- Conversation coherence analysis

[✓] **Streamlit Integration & Visualizations**
- Interactive web interface
- Document embeddings visualization (t-SNE, UMAP, PCA)
- Performance metrics dashboards
- Conversation analytics

[✓] **RAG Pipeline Knowledge**
- Complete implementation with embeddings and vector stores
- Advanced conversational AI features
- Comprehensive evaluation system

## 🚀 Features

### Core Functionality
- **Multi-language Support**: Automatic language detection and translation
- **Memory-Enabled Conversations**: Context-aware multi-turn dialogues
- **Real-time Evaluation**: Live performance metrics and quality assessment
- **Interactive Visualizations**: Advanced data visualization and analytics

### Technical Capabilities
- **Advanced Text Processing**: Recursive character text splitting with overlap
- **State-of-the-art Embeddings**: Sentence Transformers for semantic search
- **Efficient Vector Storage**: FAISS for fast similarity search
- **Comprehensive Evaluation**: ROUGE, BLEU, and coherence metrics
- **Scalable Architecture**: Modular design for easy extension

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Groq API key (get from [console.groq.com](https://console.groq.com))

### Setup Instructions

1. **Clone or Download** the project files to your local machine

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK Data** (if not automatically downloaded):
   ```python
   import nltk
   nltk.download('punkt')
   ```

4. **Configure API Key**:
   - The application uses the existing .env file with GROQ_API_KEY
   - If needed, get a new key from [console.groq.com](https://console.groq.com)
   - Update the .env file with your API key

## 🏃‍♂️ Usage

### Running the Application

1. **Start the Streamlit App**:
   ```bash
   streamlit run enhanced_app.py
   ```

2. **Open in Browser**: The app will automatically open at `http://localhost:8501`

### Using the Application

#### Step 1: Configuration
1. Upload your domain-specific document (TXT format supported)
2. API key is automatically loaded from .env file
3. Select your preferred LLM model from available production models

#### Step 2: Document Analysis
- View comprehensive document statistics
- Analyze text preprocessing results
- Examine chunk distribution and vocabulary metrics

#### Step 3: Start Chatting
- Ask questions in any language
- Engage in multi-turn conversations
- View real-time response metrics

#### Step 4: Evaluate Performance
- Monitor evaluation metrics dashboard
- Analyze conversation coherence
- Review detailed performance statistics

#### Step 5: Explore Visualizations
- Generate document embeddings visualizations
- View conversation analytics
- Explore performance trends

## 📊 Evaluation Methodology

### Retrieval Evaluation
- **Semantic Similarity**: Cosine similarity between query and retrieved documents
- **Relevance Scoring**: Top-K retrieval accuracy assessment
- **Context Quality**: Evaluation of retrieved document relevance

### Generation Evaluation
- **ROUGE Scores**: ROUGE-1, ROUGE-2, ROUGE-L for answer quality
- **BLEU Scores**: N-gram overlap assessment
- **Structural Analysis**: Answer completeness and organization

### Conversation Evaluation
- **Coherence Scoring**: Context maintenance across dialogue turns
- **Memory Utilization**: Assessment of conversation history usage
- **Response Relevance**: Contextual appropriateness evaluation

## 🏗️ Architecture

### Core Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Document      │───▶│   Preprocessing  │───▶│   Vector Store  │
│   Upload        │    │   & Analysis     │    │   (FAISS)       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌──────────────────┐            │
│   User Query    │───▶│   Retrieval      │◄───────────┘
└─────────────────┘    │   System         │
                       └──────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Response      │◄───│   LLM Generation │◄───│   Conversation  │
│   Generation    │    │   (Groq API)     │    │   Memory        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   Evaluation &   │
                       │   Metrics        │
                       └──────────────────┘
```

### Technical Stack
- **Frontend**: Streamlit with custom CSS styling
- **Backend**: LangChain for orchestration
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Database**: FAISS for similarity search
- **LLM API**: Groq (Llama, Mixtral, Gemma models)
- **Evaluation**: ROUGE, BLEU, custom coherence metrics
- **Visualization**: Plotly, Matplotlib, Seaborn

## 🔧 Configuration Options

### Model Selection
- **Llama 3.1 8B Instant**: Fast inference, good quality
- **Llama 3.3 70B Versatile**: High quality, slower inference
- **OpenAI GPT OSS 20B**: Balanced performance
- **OpenAI GPT OSS 120B**: Highest quality

### Preprocessing Parameters
- **Chunk Size**: 1000 characters (configurable)
- **Chunk Overlap**: 200 characters (configurable)
- **Text Splitter**: Recursive character splitting
- **Separators**: Paragraph, sentence, and word boundaries

### Visualization Methods
- **t-SNE**: Traditional dimensionality reduction
- **UMAP**: Modern, topology-preserving reduction
- **PCA**: Linear dimensionality reduction

## ⚠️ Limitations

### Current Limitations
1. **Single Document Processing**: Currently supports one document at a time
2. **API Dependency**: Requires external API for LLM inference
3. **Memory Constraints**: Limited conversation history in session state
4. **Evaluation Scope**: No human evaluation integration

### Known Issues
- Large documents may require processing time
- Complex document structures need preprocessing
- Translation quality depends on GoogleTranslator API

## 🚀 Future Improvements

### Enhanced Features
1. **Multi-document Support**: Process multiple documents simultaneously
2. **Advanced Retrieval**: Hybrid search (semantic + keyword)
3. **Fine-tuning Capabilities**: Domain-specific model adaptation
4. **Human Evaluation**: Integration of human assessment tools

### Performance Optimizations
1. **Caching System**: Reduce repeated computations
2. **Batch Processing**: Efficient handling of multiple queries
3. **Model Optimization**: Quantization and acceleration techniques
4. **Scalable Deployment**: Production-ready architecture

## 📝 Usage Examples

### Example Queries for Multi-Agent Systems Domain

Based on the provided MAS document, try these sample questions:

```
1. "What are the key components of a BDI architecture?"
2. "Explain the difference between reactive and proactive agents"
3. "How does the subsumption architecture work?"
4. "What are the advantages and disadvantages of layered architectures?"
5. "Compare different types of intelligent agents"
```

### Multi-turn Conversation Example

```
User: "What is a multi-agent system?"
Bot: [Provides definition and key concepts]

User: "Can you give me an example?"
Bot: [References previous context and provides specific examples]

User: "How do they communicate?"
Bot: [Continues conversation thread about MAS communication]
```

## 🤝 Contributing

This is an educational project demonstrating RAG pipeline implementation. Suggestions for improvements are welcome!

### Development Setup
1. Fork the repository
2. Install development dependencies
3. Make your changes
4. Test thoroughly
5. Submit pull request

## 📄 License

This project is for educational purposes as part of an LLM laboratory assignment.

## 🙏 Acknowledgments

- LangChain for orchestration framework
- Streamlit for web interface
- Hugging Face for embeddings models
- Groq for LLM inference API
- CHRIST University for the academic framework

## 📞 Support

For questions or issues related to this assignment implementation, please refer to the comprehensive documentation in the "About" tab of the application or consult the course materials.

---

**Note**: This implementation demonstrates understanding of RAG pipelines, conversational AI, evaluation methodologies, and system design principles as required by the assignment criteria.
