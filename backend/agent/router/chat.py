
# --- Modify app/routers/chat.py ---
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
import uuid
from loguru import logger # Import logger

from agent import crud, schemas, models, auth
from agent.database import get_db
from agent.agent_system import run_agent_sync # Import the specific runner
from agent.vector_store_manager import cleanup_guest_store

router = APIRouter(prefix="/chat", tags=["Chat"])

@router.post("/user", response_model=schemas.AgentResponse)
async def chat_with_agent_user(
    payload: schemas.AgentQuery,
    current_user: models.User = Depends(auth.get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    session_id = payload.session_id or str(uuid.uuid4())
    # Use session_id as the thread_id for conversation memory
    thread_id = f"user_{current_user.id}_session_{session_id}"
    logger.info(f"User chat request. Thread ID: {thread_id}")

    # Optional: Load history here if needed by agent state
    # history = await crud.get_chat_history(...)
    # formatted_history = ...

    # Run agent synchronously (consider asyncio.to_thread for async FastAPI)
    final_state = run_agent_sync(
        query=payload.query,
        thread_id=thread_id,
        user_id=current_user.id,
        guest_session_id=None
        # chat_history=formatted_history
    )

    # Save to DB
    chat_message = schemas.ChatMessageCreate(
        query=payload.query,
        response=final_state.get("agent_response"),
        session_id=session_id # Store the user's chat session ID
    )
    await crud.create_chat_message(db, message=chat_message, user_id=current_user.id)

    return schemas.AgentResponse(
        response=final_state.get("agent_response", "Error processing request."),
        session_id=session_id, # Return user chat session ID
        suggested_questions=final_state.get("suggested_questions"),
        error=final_state.get("error")
    )

@router.post("/guest", response_model=schemas.AgentResponse)
async def chat_with_agent_guest(
    payload: schemas.AgentQuery,
    db: AsyncSession = Depends(get_db),
):
    guest_session_id = payload.guest_session_id
    if not guest_session_id:
        guest_session_id = str(uuid.uuid4())
        await crud.create_guest_session(db, session_id=guest_session_id)
        logger.info(f"Created new guest session: {guest_session_id}")
    else:
        session = await crud.get_guest_session(db, session_id=guest_session_id)
        if not session:
            # Handle case where client sends invalid ID - create new one
            logger.warning(f"Invalid guest session ID '{guest_session_id}' received. Creating new session.")
            guest_session_id = str(uuid.uuid4())
            await crud.create_guest_session(db, session_id=guest_session_id)

    # Use guest_session_id as thread_id
    thread_id = f"guest_{guest_session_id}"
    logger.info(f"Guest chat request. Thread ID: {thread_id}")

    # Optional: Load history
    # history = await crud.get_guest_chat_history(...)
    # formatted_history = ...

    final_state = run_agent_sync(
        query=payload.query,
        thread_id=thread_id,
        user_id=None,
        guest_session_id=guest_session_id
        # chat_history=formatted_history
    )

    # Save guest chat
    chat_message = schemas.GuestChatMessageCreate(
        query=payload.query,
        response=final_state.get("agent_response")
    )
    await crud.create_guest_chat_message(db, message=chat_message, session_id=guest_session_id)

    return schemas.AgentResponse(
        response=final_state.get("agent_response", "Error processing request."),
        guest_session_id=guest_session_id, # Return the active guest session ID
        suggested_questions=final_state.get("suggested_questions"),
        error=final_state.get("error")
    )

@router.delete("/guest/{guest_session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def end_guest_session(
    guest_session_id: str,
    db: AsyncSession = Depends(get_db)
):
    logger.info(f"Received request to end guest session: {guest_session_id}")
    # Optional: Delete DB records (documents, chat history, session itself)
    # await crud.delete_guest_documents(db, session_id=guest_session_id)
    # ... delete chat history ...
    # ... delete guest session record ...

    # *** Clean up the vector store ***
    cleanup_guest_store(guest_session_id)
    return