import hashlib, os

def hash_password(password: str):
    salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode(),
        salt,
        200000
    )
    return salt + key

def verify_password(stored: bytes, password: str):
    salt = stored[:16]
    key = stored[16:]
    new_key = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode(),
        salt,
        200000
    )
    return new_key == key
