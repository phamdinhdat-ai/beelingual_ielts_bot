from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import update # Import update statement
import datetime
from agent import models, schemas
from agent.auth import get_password_hash
from sqlalchemy import delete # Import delete statement
from sqlalchemy.orm import Session
import os
from datetime import datetime, timedelta
from typing import Optional
# --- User CRUD ---
async def get_user(db: AsyncSession, user_id: int):
    result = await db.execute(select(models.User).filter(models.User.id == user_id))
    return result.scalar_one_or_none()

async def get_user_by_username(db: AsyncSession, username: str):
    result = await db.execute(select(models.User).filter(models.User.username == username))
    return result.scalar_one_or_none()

async def create_user(db: AsyncSession, user: schemas.UserCreate):
    hashed_password = get_password_hash(user.password)
    db_user = models.User(
        username=user.username,
        email=user.email,
        hashed_password=hashed_password
    )
    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)
    return db_user

# --- Document CRUD ---
async def create_user_document(db: AsyncSession, doc: schemas.DocumentCreate, user_id: int):
    db_doc = models.Document(**doc.dict(), user_id=user_id)
    db.add(db_doc)
    await db.commit()
    await db.refresh(db_doc)
    return db_doc

async def get_user_documents(db: AsyncSession, user_id: int, skip: int = 0, limit: int = 100):
    result = await db.execute(
        select(models.Document)
        .filter(models.Document.user_id == user_id)
        .offset(skip).limit(limit)
    )
    return result.scalars().all()

async def get_document_by_id(db: AsyncSession, doc_id: int, user_id: int):
     result = await db.execute(
        select(models.Document)
        .filter(models.Document.id == doc_id, models.Document.user_id == user_id)
    )
     return result.scalar_one_or_none()

async def delete_document(db: AsyncSession, doc_id: int, user_id: int):
    db_doc = await get_document_by_id(db, doc_id, user_id)
    if db_doc:
        await db.delete(db_doc)
        await db.commit()
        return True
    return False


# --- Chat Message CRUD ---
async def create_chat_message(db: AsyncSession, message: schemas.ChatMessageCreate, user_id: int):
    db_message = models.ChatMessage(**message.dict(), user_id=user_id)
    db.add(db_message)
    await db.commit()
    await db.refresh(db_message)
    return db_message

async def get_chat_history(db: AsyncSession, session_id: str, user_id: int, limit: int = 20):
    result = await db.execute(
        select(models.ChatMessage)
        .filter(models.ChatMessage.user_id == user_id, models.ChatMessage.session_id == session_id)
        .order_by(models.ChatMessage.timestamp.desc())
        .limit(limit)
    )
    # Return in chronological order for agent context
    return result.scalars().all()[::-1]


# --- Guest CRUD ---
async def create_guest_session(db: AsyncSession, session_id: str):
    db_session = models.GuestSession(id=session_id)
    db.add(db_session)
    await db.commit()
    await db.refresh(db_session)
    return db_session

async def get_guest_session(db: AsyncSession, session_id: str):
    result = await db.execute(select(models.GuestSession).filter(models.GuestSession.id == session_id))
    guest_session = result.scalar_one_or_none()
    # Update last accessed time
    if guest_session:
        guest_session.last_accessed = datetime.datetime.utcnow()
        await db.commit()
        await db.refresh(guest_session)
    return guest_session

async def create_guest_document(db: AsyncSession, doc: schemas.GuestDocumentCreate, session_id: str):
    db_doc = models.GuestDocument(**doc.dict(), guest_session_id=session_id)
    db.add(db_doc)
    await db.commit()
    await db.refresh(db_doc)
    return db_doc

async def get_guest_documents(db: AsyncSession, session_id: str, limit: int = 100):
    result = await db.execute(
        select(models.GuestDocument)
        .filter(models.GuestDocument.guest_session_id == session_id)
        .limit(limit) # Limit guest documents
    )
    return result.scalars().all()

async def delete_guest_documents(db: AsyncSession, session_id: str):
    # This deletes DB entries. Actual file/vector store cleanup is separate.
    stmt = delete(models.GuestDocument).where(models.GuestDocument.guest_session_id == session_id) # Import delete
    await db.execute(stmt)
    await db.commit()

async def create_guest_chat_message(db: AsyncSession, message: schemas.GuestChatMessageCreate, session_id: str):
    db_message = models.GuestChatMessage(**message.dict(), guest_session_id=session_id)
    db.add(db_message)
    await db.commit()
    await db.refresh(db_message)
    return db_message

async def get_guest_chat_history(db: AsyncSession, session_id: str, limit: int = 20):
     result = await db.execute(
        select(models.GuestChatMessage)
        .filter(models.GuestChatMessage.guest_session_id == session_id)
        .order_by(models.GuestChatMessage.timestamp.desc())
        .limit(limit)
    )
     return result.scalars().all()[::-1] # Chronological