from fastapi import FastAPI
from agent.database import engine, Base 
from agent.router import documents as documents_router
from agent.router import chat as chat_router
from agent.router import auth as auth_router
from agent.router import users as users_router
# Import models to ensure they are registered with Base
from agent import models

# Create tables on startup (use Alembic for production migrations)
async def create_db_and_tables():
    async with engine.begin() as conn:
        # await conn.run_sync(Base.metadata.drop_all)  # Use cautiously for reset
        await conn.run_sync(Base.metadata.create_all)

app = FastAPI(title="Multi-Agent FastAPI")

# --- Event Handlers ---
@app.on_event("startup")
async def on_startup():
    print("Starting up...")
    await create_db_and_tables()
    print("Database tables created (if they didn't exist).")
    # Initialize any other resources if needed

@app.on_event("shutdown")
async def on_shutdown():
    print("Shutting down...")
    # Clean up resources if needed

# --- Include Routers ---
app.include_router(auth_router.router)
app.include_router(users_router.router)
app.include_router(documents_router.router)
app.include_router(chat_router.router)

@app.get("/")
async def root():
    return {"message": "Welcome to the Multi-Agent API"}