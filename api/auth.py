import os
from datetime import datetime, timedelta
from typing import Optional

import bcrypt
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from pydantic import BaseModel

import config  # noqa: F401  (assicura che .env sia caricato prima di leggere le env var sotto)

router = APIRouter()

SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    # Un fallback a un segreto casuale per processo sembra innocuo in dev ma è
    # esattamente quello che succede in produzione se la env var viene dimenticata
    # in un redeploy: ogni riavvio rigenera la chiave e invalida silenziosamente
    # tutti i token emessi in precedenza (falliscono con lo stesso errore di un
    # token scaduto, anche se erano ben dentro la loro validità). Meglio fallire
    # subito all'avvio che degradare in modo silenzioso.
    raise RuntimeError(
        "SECRET_KEY non impostata nell'ambiente. Imposta SECRET_KEY nel .env "
        "(un valore stabile, non rigenerato ad ogni avvio) prima di avviare il servizio."
    )

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")


class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str


class RefreshRequest(BaseModel):
    refresh_token: str


class TokenData(BaseModel):
    username: str


class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode(), hashed_password.encode())


def get_password_hash(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire, "type": "access"})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def _decode_token(token: str, expected_type: str) -> str:
    """Decode e valida un JWT, verificando che sia del tipo atteso ('access' o 'refresh').
    Ritorna lo username (claim 'sub') o solleva HTTPException 401."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None or not isinstance(username, str):
            raise credentials_exception
        if payload.get("type") != expected_type:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    return username


def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    username = _decode_token(token, expected_type="access")
    return User(username=username)  # type: ignore[arg-type]


_ADMIN_USERNAME = os.getenv("AUTH_ADMIN_USERNAME", "admin")
_ADMIN_PASSWORD = os.getenv("AUTH_ADMIN_PASSWORD")
if not _ADMIN_PASSWORD:
    # Stesso motivo di SECRET_KEY sopra: una password rigenerata ad ogni riavvio
    # non è recuperabile dai client (es. Laravel CognitorClient) che hanno la
    # password attesa fissa nel proprio .env — dopo un riavvio anche il login
    # pieno di ripiego fallirebbe, non solo il refresh del token.
    raise RuntimeError(
        "AUTH_ADMIN_PASSWORD non impostata nell'ambiente. Imposta "
        "AUTH_ADMIN_USERNAME/AUTH_ADMIN_PASSWORD nel .env prima di avviare il servizio."
    )

# Utente unico di servizio (single-user demo auth). Le credenziali vengono da env,
# non da valori hardcoded in sorgente.
FAKE_USERS_DB = {
    _ADMIN_USERNAME: {
        "username": _ADMIN_USERNAME,
        "full_name": "Admin User",
        "email": "admin@example.com",
        "hashed_password": get_password_hash(_ADMIN_PASSWORD),
        "disabled": False,
    }
}


def authenticate_user(username: str, password: str):
    user = FAKE_USERS_DB.get(username)
    if not user:
        return False
    if not verify_password(password, user["hashed_password"]):
        return False
    return user


@router.post("/token", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    refresh_token = create_refresh_token(data={"sub": user["username"]})
    return {"access_token": access_token, "refresh_token": refresh_token, "token_type": "bearer"}


@router.post("/refresh", response_model=Token)
def refresh(payload: RefreshRequest):
    """
    Scambia un refresh token valido (non scaduto, type=refresh) con una nuova coppia
    access/refresh token, senza richiedere di nuovo username/password. Il refresh
    token viene ruotato ad ogni uso (non riutilizzabile due volte).
    """
    username = _decode_token(payload.refresh_token, expected_type="refresh")
    user = FAKE_USERS_DB.get(username)
    if not user or user.get("disabled"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    new_refresh_token = create_refresh_token(data={"sub": user["username"]})
    return {"access_token": access_token, "refresh_token": new_refresh_token, "token_type": "bearer"}


@router.get("/me", response_model=User)
def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user
