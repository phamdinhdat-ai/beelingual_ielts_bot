
# --- Modify app/routers/documents.py ---
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

from agent import crud, schemas, models, auth
from agent.database import get_db
# Import vector store manager functions
from agent.vector_store_manager import (
    process_file_content,
    add_documents,
    get_collection_name,
    
)

router = APIRouter(prefix="/documents", tags=["Documents"])

@router.post("/user", response_model=schemas.DocumentPublic, status_code=status.HTTP_201_CREATED)
async def upload_user_document(
    file: UploadFile = File(...),
    current_user: models.User = Depends(auth.get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    contents = await file.read()
    if not contents: raise HTTPException(status_code=400, detail="Empty file")

    # Process content into LangChain Documents using vector_store_manager function
    docs = process_file_content(contents, file.filename)
    if not docs: raise HTTPException(status_code=400, detail="Could not process file content")

    # content_hash = calculate_content_hash(contents) # Optional

    # Store metadata in DB
    doc_create = schemas.DocumentCreate(filename=file.filename, content_hash=None)
    db_doc = await crud.create_user_document(db, doc=doc_create, user_id=current_user.id)

    # Add Documents to Vector Store
    collection_name = get_collection_name(user_id=current_user.id)
    add_documents(docs, collection_name) # Handles errors internally via logging

    return db_doc

# Add GET and DELETE for user documents as before...

@router.post("/guest/{guest_session_id}", response_model=schemas.GuestDocumentPublic, status_code=status.HTTP_201_CREATED)
async def upload_guest_document(
    guest_session_id: str,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    guest_session = await crud.get_guest_session(db, session_id=guest_session_id)
    if not guest_session: raise HTTPException(status_code=404, detail="Guest session not found.")

    contents = await file.read()
    if not contents: raise HTTPException(status_code=400, detail="Empty file")
    docs = process_file_content(contents, file.filename)
    if not docs: raise HTTPException(status_code=400, detail="Could not process file")
    # content_hash = calculate_content_hash(contents) # Optional

    # Store metadata in DB
    doc_create = schemas.GuestDocumentCreate(filename=file.filename, content_hash=None)
    db_doc = await crud.create_guest_document(db, doc=doc_create, session_id=guest_session_id)

    # Add Documents to Vector Store
    collection_name = get_collection_name(guest_session_id=guest_session_id)
    add_documents(docs, collection_name)

    return db_doc