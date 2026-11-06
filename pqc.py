"""Post-quantum secured and authenticated update channel (Algorithm 2).
ML-KEM-768 (FIPS 203) and ML-DSA-65 (FIPS 204) from kyber-py / dilithium-py; HKDF-SHA256 and AES-256-GCM from `cryptography`."""
import os, struct, hashlib
from kyber_py.ml_kem import ML_KEM_768
from dilithium_py.ml_dsa import ML_DSA_65
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes


def hkdf(ss, info):
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=info).derive(ss)


def header(r, k, nonce):  # 8-byte header (round, hospital) + 12-byte nonce
    return struct.pack(">II", r, k) + nonce


class Server:
    def __init__(self):
        self.ek, self.dk = ML_KEM_768.keygen()
        self.pk_s, self.sk_s = ML_DSA_65.keygen()
        self.registry = {}; self.seen = set()

    def register(self, k, pk): self.registry[k] = pk

    def secure_receive(self, m, r_cur):
        k, r, nonce, ct, c, sig = m
        A = header(r, k, nonce)
        pk = self.registry.get(k)
        if pk is None or not ML_DSA_65.verify(pk, hashlib.sha256(A + ct + c).digest(), sig):
            return None  # forgery or tampering
        if r != r_cur or (r, k, nonce) in self.seen:
            return None  # stale or replayed
        self.seen.add((r, k, nonce))
        ss = ML_KEM_768.decaps(self.dk, ct)
        key = hkdf(ss, struct.pack(">II", r, k))
        try:
            return AESGCM(key).decrypt(nonce, c, A)
        except Exception:
            return None  # tampering

    def sign_broadcast(self, r, payload):
        return ML_DSA_65.sign(self.sk_s, hashlib.sha256(struct.pack(">I", r) + payload).digest())


class Client:
    def __init__(self, k, server):
        self.k = k; self.pk, self.sk = ML_DSA_65.keygen(); server.register(k, self.pk); self.ek = server.ek

    def secure_send(self, u, r):
        ss, ct = ML_KEM_768.encaps(self.ek)
        key = hkdf(ss, struct.pack(">II", r, self.k)); nonce = os.urandom(12)
        A = header(r, self.k, nonce)
        c = AESGCM(key).encrypt(nonce, u, A)  # ciphertext || 16-byte tag
        sig = ML_DSA_65.sign(self.sk, hashlib.sha256(A + ct + c).digest())
        return (self.k, r, nonce, ct, c, sig)


def msg_size(m):
    k, r, nonce, ct, c, sig = m
    return 8 + len(nonce) + len(ct) + len(c) + len(sig)
