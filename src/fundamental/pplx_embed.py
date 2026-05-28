import logging
import torch
from transformers import AutoModel, AutoTokenizer

class PerplexityEmbeddingService:
    def __init__(self, model_id="perplexity-ai/pplx-embed-context-v1-4b"):
        """
        Khởi tạo và nạp mô hình vào GPU T4 một lần duy nhất.
        """
        logging.info(f"Đang nạp mô hình {model_id} vào GPU... Vui lòng đợi...")
        
        # 1. Load Tokenizer & Model tối ưu cho GPU T4
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16,   # Giảm dung lượng VRAM xuống 1/2
            device_map="auto",           # Tự động đẩy lên GPU có sẵn (T4)
            low_cpu_mem_usage=True       # Tránh tràn RAM hệ thống khi nạp
        )
        logging.info("Nạp mô hình thành công! Sẵn sàng trích xuất embedding.")

    def get_embedding(self, text: str):
        """
        Hàm trích xuất Vector Embedding từ văn bản đầu vào.
        """
        if not text.strip():
            return []

        # Chuyển văn bản thành token và đẩy lên cùng thiết bị với model
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(self.model.device)

        with torch.no_grad(): # Tắt tính toán gradient để tiết kiệm VRAM và tăng tốc
            outputs = self.model(**inputs)

        # Trích xuất vector từ hidden state của token cuối cùng
        embeddings = outputs.last_hidden_state[:, -1, :]
        
        # Chuyển kết quả về CPU, dạng danh sách số (List float) để giải phóng bộ nhớ GPU nhanh hơn
        vector = embeddings.squeeze().cpu().tolist()

        # Dọn dẹp các biến trung gian để giải phóng bộ nhớ cache của GPU
        del inputs, outputs, embeddings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return vector

    def get_embeddings_batch(self, texts: list):
        """
        (Mở rộng) Hàm xử lý danh sách nhiều văn bản cùng lúc để tối ưu tốc độ GPU
        """
        if not texts:
            return []
            
        inputs = self.tokenizer(
            texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        embeddings = outputs.last_hidden_state[:, -1, :]
        vectors = embeddings.cpu().tolist()

        del inputs, outputs, embeddings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return vectors
