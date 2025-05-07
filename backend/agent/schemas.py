from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
import datetime
import uuid

# --- Base Models ---
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

# --- User Schemas ---
class UserBase(BaseModel):
    username: str
    email: EmailStr

class UserCreate(UserBase):
    password: str

class UserInDB(UserBase):
    id: int
    is_active: bool
    created_at: datetime.datetime

    class Config:
        orm_mode = True # Allows creating schema from ORM model

class UserPublic(UserBase):
    id: int
    created_at: datetime.datetime
    class Config:
        orm_mode = True

# --- Document Schemas ---
class DocumentBase(BaseModel):
    filename: str

class DocumentCreate(DocumentBase):
    content_hash: Optional[str] = None # Calculated server-side
    storage_path: Optional[str] = None # Set server-side

class DocumentPublic(DocumentBase):
    id: int
    uploaded_at: datetime.datetime
    user_id: int
    class Config:
        orm_mode = True

# --- Chat Schemas ---
class ChatMessageBase(BaseModel):
    query: str

class ChatMessageCreate(ChatMessageBase):
    response: Optional[str] = None
    session_id: str

class ChatMessagePublic(ChatMessageBase):
    id: int
    response: Optional[str]
    timestamp: datetime.datetime
    session_id: str
    user_id: int
    class Config:
        orm_mode = True

# --- Guest Schemas ---
class GuestSessionCreate(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))

class GuestSessionPublic(BaseModel):
    id: str
    created_at: datetime.datetime
    last_accessed: datetime.datetime
    class Config:
        orm_mode = True

class GuestDocumentCreate(DocumentBase):
    content_hash: Optional[str] = None
    storage_path: Optional[str] = None

class GuestDocumentPublic(DocumentBase):
    id: int
    uploaded_at: datetime.datetime
    guest_session_id: str
    class Config:
        orm_mode = True

class GuestChatMessageCreate(ChatMessageBase):
    response: Optional[str] = None

class GuestChatMessagePublic(ChatMessageBase):
    id: int
    response: Optional[str]
    timestamp: datetime.datetime
    guest_session_id: str
    class Config:
        orm_mode = True

# --- Agent Interaction Schemas ---
class AgentQuery(BaseModel):
    query: str
    session_id: Optional[str] = None # For user sessions, create/pass one
    guest_session_id: Optional[str] = None # For guest sessions

class AgentResponse(BaseModel):
    response: str
    session_id: Optional[str] = None
    guest_session_id: Optional[str] = None
    suggested_questions: Optional[List[str]] = None # Add this line
    error: Optional[str] = None
