from rag.data.data_loader import load_arcd_dataset
from rag.data.text_cleaning import clean_dataframe
from rag.embeddings.embeddings import EmbeddingGenerator
from rag.vector_store.qdrant_store import VectorDB
from rag.generation.models_loader import ModelLoader
from rag.generation.gpt2_generator import GPT2Generator
from rag.generation.gemini_generator import GeminiGenerator
from rag.retrieval.retrieval import Retriever

class RAGPipeline:
    def __init__(self):
        self.df_train = None
        self.df_val = None
        self.embedding_generator = None
        self.qdrant_store = None
        self.gpt2_generator = None
        self.gemini_generator = None
        self.retriever = None
        self.initialized = False
        
    def initialize_pipeline(self):
        """Initialize the complete RAG pipeline """
        
        print("Loading dataset...")
        self.df_train, self.df_val = load_arcd_dataset()
        
        print("Cleaning data...")
        self.df_train = clean_dataframe(self.df_train)
        self.df_val = clean_dataframe(self.df_val)
        
        print("Initializing embedding generator...")
        self.embedding_generator = EmbeddingGenerator()
        
        print("Generating embeddings for data...")
        train_embeddings = self.embedding_generator.generate_embeddings(
            self.df_train["context"].tolist()
        )
        val_embeddings = self.embedding_generator.generate_embeddings(
            self.df_val["context"].tolist()
        )
        
        print("Initializing vector database...")
        self.qdrant_store = VectorDB()
        self.qdrant_store.create_collection(train_embeddings.shape[1])
        
        print("Inserting documents into vector database...")
        insertion_result = self.qdrant_store.insert_all_samples(
            self.df_train, train_embeddings,
            self.df_val, val_embeddings
        )
        
        print(f"Inserted {insertion_result['total']} documents total "
              f"({insertion_result['train']['count']} train, "
              f"{insertion_result['val']['count']} validation)")
        
        print("Loading models...")
        gpt2_tokenizer, gpt2_model = ModelLoader.load_gpt2()
        gemini_model = ModelLoader.load_gemini()
        
        print("Initializing generators...")
        self.gpt2_generator = GPT2Generator(gpt2_tokenizer, gpt2_model)
        self.gemini_generator = GeminiGenerator(gemini_model)
        
        print("Initializing retriever...")
        self.retriever = Retriever()
        
        self.initialized = True
        print("Pipeline initialization complete!")
        
        return self
    
    def generate_answer(self, question, model="gpt2", use_rag=True):
        """Generate answer using specified model"""
        if not self.initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize_pipeline() first.")
        
        if model.lower() == "gpt2":
            if use_rag:
                return self.gpt2_generator.generate_with_rag(question)
            else:
                return self.gpt2_generator.generate_without_rag(question)
        elif model.lower() == "gemini":
            if use_rag:
                return self.gemini_generator.generate_with_rag(question)
            else:
                return self.gemini_generator.generate_without_rag(question)
        else:
            raise ValueError(f"Unsupported model: {model}")