import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.orm import relationship
from agent.database import Base

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    documents = relationship("Document", back_populates="owner", cascade="all, delete-orphan")
    chat_messages = relationship("ChatMessage", back_populates="user", cascade="all, delete-orphan")

class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True, nullable=False)
    content_hash = Column(String, index=True) # To potentially avoid duplicates
    storage_path = Column(String) # Or store content directly if small (not recommended)
    uploaded_at = Column(DateTime, default=datetime.datetime.utcnow)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    owner = relationship("User", back_populates="documents")
    # Add vector store references if needed, e.g., collection name or vector IDs

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True) # To group conversations
    query = Column(Text, nullable=False)
    response = Column(Text)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    user = relationship("User", back_populates="chat_messages")


# --- Guest Models ---

class GuestSession(Base):
    __tablename__ = "guest_sessions"
    id = Column(String, primary_key=True, index=True) # Use UUID generated in code
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    last_accessed = Column(DateTime, default=datetime.datetime.utcnow)
    # expires_at = Column(DateTime) # Can add expiration logic

    guest_documents = relationship("GuestDocument", back_populates="session", cascade="all, delete-orphan")
    guest_chat_messages = relationship("GuestChatMessage", back_populates="session", cascade="all, delete-orphan")


class GuestDocument(Base):
    __tablename__ = "guest_documents"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True, nullable=False)
    content_hash = Column(String, index=True)
    storage_path = Column(String)
    uploaded_at = Column(DateTime, default=datetime.datetime.utcnow)
    guest_session_id = Column(String, ForeignKey("guest_sessions.id"), nullable=False)

    session = relationship("GuestSession", back_populates="guest_documents")


class GuestChatMessage(Base):
    __tablename__ = "guest_chat_messages"
    id = Column(Integer, primary_key=True, index=True)
    query = Column(Text, nullable=False)
    response = Column(Text)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    guest_session_id = Column(String, ForeignKey("guest_sessions.id"), nullable=False)

    session = relationship("GuestSession", back_populates="guest_chat_messages")