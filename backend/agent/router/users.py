# --- app/routers/users.py ---
from fastapi import APIRouter, Depends
from agent import models, schemas, auth

router = APIRouter(prefix="/users", tags=["Users"])

@router.get("/me", response_model=schemas.UserPublic)
async def read_users_me(current_user: models.User = Depends(auth.get_current_active_user)):
    # Simply return the user object obtained from the dependency
    return current_user

