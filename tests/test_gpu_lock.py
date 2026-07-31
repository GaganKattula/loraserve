"""
GPU lock tests — requires a running Redis instance.

Architecture decision: Why Redis SET NX and not asyncio.Lock?
  The worker (RQ process) and serving engine (FastAPI process) are separate OS
  processes, potentially on the same GPU pod. An in-process lock can't coordinate
  them. Redis is already in the stack for RQ and pub/sub.

Why Lua check-and-delete for release?
  Plain DEL is unsafe. Scenario:
    1. Worker A acquires lock with TTL=10min
    2. Worker A stalls, TTL expires
    3. Worker B acquires the now-free lock
    4. Worker A recovers, calls DEL → releases Worker B's lock
  The Lua script checks ownership before deleting: if the stored token doesn't
  match, the release is a no-op. This is the standard distributed lock pattern
  (Redlock lite).
"""

import time
import gpu_lock


class TestGpuLock:

    def test_acquire_returns_true(self, redis_client):
        assert gpu_lock.acquire("token-a") is True

    def test_acquire_fails_when_held(self, redis_client):
        gpu_lock.acquire("token-a")
        assert gpu_lock.acquire("token-b") is False

    def test_is_held(self, redis_client):
        assert gpu_lock.is_held() is False
        gpu_lock.acquire("token-a")
        assert gpu_lock.is_held() is True

    def test_holder(self, redis_client):
        assert gpu_lock.holder() is None
        gpu_lock.acquire("token-a")
        assert gpu_lock.holder() == "token-a"

    def test_release_with_correct_token(self, redis_client):
        gpu_lock.acquire("token-a")
        assert gpu_lock.release("token-a") is True
        assert gpu_lock.is_held() is False

    def test_release_with_wrong_token_is_noop(self, redis_client):
        """Wrong token must NOT release the lock — prevents the stale-release bug."""
        gpu_lock.acquire("token-a")
        assert gpu_lock.release("token-b") is False
        assert gpu_lock.is_held() is True  # lock still held by token-a

    def test_renew_extends_ttl(self, redis_client):
        gpu_lock.acquire("token-a", ttl_ms=2000)
        assert gpu_lock.renew("token-a", ttl_ms=60000) is True
        # TTL should now be ~60s, not ~2s
        ttl = redis_client.pttl("gpu:lock")
        assert ttl > 50000  # well above the original 2s

    def test_renew_wrong_token_is_noop(self, redis_client):
        gpu_lock.acquire("token-a", ttl_ms=5000)
        assert gpu_lock.renew("token-b", ttl_ms=60000) is False

    def test_lock_auto_expires(self, redis_client):
        """TTL expiry is the crash recovery mechanism — no manual cleanup needed."""
        gpu_lock.acquire("token-a", ttl_ms=200)  # 200ms TTL
        time.sleep(0.3)
        assert gpu_lock.is_held() is False

    def test_acquire_after_release(self, redis_client):
        gpu_lock.acquire("token-a")
        gpu_lock.release("token-a")
        assert gpu_lock.acquire("token-b") is True
        assert gpu_lock.holder() == "token-b"
