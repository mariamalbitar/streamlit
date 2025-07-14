from fastapi import FastAPI, HTTPException, Form, Depends
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from passlib.context import CryptContext
import uvicorn
import json
import os  
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Generator


app = FastAPI(title="Financial Insights API", description="API for user authentication and financial data")


DATABASE_URL = "sqlite:///./users.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)

Base.metadata.create_all(bind=engine)


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class UserCreate(BaseModel):
    username: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str


@app.post("/register")
async def register(username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    """
    Register a new user in the database.
    
    Args:
        username: The username provided by the user.
        password: The password provided by the user.
        db: Database session dependency.
    
    Returns:
        JSON response with success message or error.
    """
    try:
        
        user = db.query(User).filter(User.username == username).first()
        if user:
            raise HTTPException(status_code=400, detail="Username already exists")
        
    
        hashed_password = get_password_hash(password)
        new_user = User(username=username, hashed_password=hashed_password)
        db.add(new_user)
        db.commit()
        return {"msg": "User registered successfully"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Registration error: {str(e)}")


@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    """
    Authenticate a user with username and password.
    
    Args:1
        username: The username provided by the user.
        password: The password provided by the user.
        db: Database session dependency.
    
    Returns:
        JSON response with welcome message or error.
    """
    try:
        
        user = db.query(User).filter(User.username == username).first()
        if not user or not verify_password(password, user.hashed_password):
            raise HTTPException(status_code=400, detail="Invalid username or password")
        return {"msg": f"Welcome {username}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Login error: {str(e)}")


@app.get("/data/cleaned")
async def get_cleaned_data():
    """
    Serve the cleaned dataset from the specified JSON file.
    
    Returns:
        JSON response with the cleaned dataset or error.
    """
    try:
        file_path = r"C:\Users\Fa\Desktop\Streamlit-Authentication-main\cleaned.json"
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Cleaned data file not found")
        with open (file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return JSONResponse(content=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading cleaned data: {str(e)}")


@app.get("/data/financial_phrasebank")
async def get_financial_phrasebank_data():
    """
    Serve the financial_phrasebank dataset from the specified JSON file.
    
    Returns:
        JSON response with the financial_phrasebank dataset or error.
    """
    try:
        file_path = r"C:\Users\Fa\Desktop\Streamlit-Authentication-main\financial_phrasebank (2).json"
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Financial phrasebank data file not found")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return JSONResponse(content=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading financial_phrasebank data: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)