from sqlalchemy import Column, Integer, String, Float, Text, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime
from src.store.database import Base

class Campaign(Base):
    __tablename__ = "campaigns"

    id = Column(Integer, primary_key=True, index=True)
    job_title = Column(String(255), nullable=False)
    job_description = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    candidates = relationship("Candidate", back_populates="campaign", cascade="all, delete-orphan")

class Candidate(Base):
    __tablename__ = "candidates"

    id = Column(Integer, primary_key=True, index=True)
    campaign_id = Column(Integer, ForeignKey("campaigns.id"))
    file_name = Column(String(255), nullable=False)  # tên file gốc
    candidate_name = Column(String(255), default="Unknown")  # tên ứng viên / ID
    skills = Column(Text)  # Comma separated
    skill_overlap = Column(Float, default=0.0)
    semantic_score = Column(Float, default=0.0)
    bm25_score = Column(Float, default=0.0)
    experience_score = Column(Float, default=0.0)
    total_score = Column(Float, default=0.0)
    analysis_json = Column(Text)  # JSON toàn bộ kết quả để vẽ lại biểu đồ

    campaign = relationship("Campaign", back_populates="candidates")
