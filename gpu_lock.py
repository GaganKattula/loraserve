"""
Cross-process GPU mutex using Redis SET NX PX.

Why Redis and not asyncio.Lock / threading.Lock:
    The worker (RQ) and serving engine (FastAPI) are separate processes,
    potentially on the same GPU pod. An in-process primitive can't
    coordinate them. Redis is already in the stack.

Lock lifecycle:
    IDLE      → gpu:lock key absent, GPU free
    TRAINING  → lock held by worker, TTL renewed by heartbeat
    INFERRING → lock held by serving engine, short TTL

Release uses a Lua check-and-delete script — NOT plain DEL.
Plain DEL is unsafe: if the TTL expires and another process acquires
the lock, the original holder's DEL would release someone else's lock.
"""

from redis import Redis

from config import settings

LOCK_KEY = "gpu:lock"
DEFAULT_TTL_MS = 600_000  # 10 minutes — renewed by heartbeat during training

# Lua script: only delete the key if the stored value matches our token.
# Prevents releasing a lock that was already expired and re-acquired by
# another process.
_RELEASE_SCRIPT = """
if redis.call("GET", KEYS[1]) == ARGV[1] then
    return redis.call("DEL", KEYS[1])
else
    return 0
end
"""


def _get_client() -> Redis:
    return Redis.from_url(settings.redis_url)


def acquire(token: str, ttl_ms: int = DEFAULT_TTL_MS) -> bool:
    """Try to acquire the GPU lock. Returns True if acquired, False if held."""
    r = _get_client()
    return r.set(LOCK_KEY, token, nx=True, px=ttl_ms) is not None


def release(token: str) -> bool:
    """Release the lock only if we still own it. Returns True if released."""
    r = _get_client()
    result = r.eval(_RELEASE_SCRIPT, 1, LOCK_KEY, token)
    return result == 1


def renew(token: str, ttl_ms: int = DEFAULT_TTL_MS) -> bool:
    """Renew TTL only if we still own the lock. For heartbeat thread."""
    r = _get_client()
    current = r.get(LOCK_KEY)
    if current is not None and current.decode() == token:
        return r.pexpire(LOCK_KEY, ttl_ms)
    return False


def is_held() -> bool:
    """Check if the GPU lock is currently held by any process."""
    r = _get_client()
    return r.exists(LOCK_KEY) == 1


def holder() -> str | None:
    """Return the token of the current lock holder, or None."""
    r = _get_client()
    val = r.get(LOCK_KEY)
    return val.decode() if val else None
