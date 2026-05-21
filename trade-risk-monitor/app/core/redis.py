import redis
from app.core.config import settings

redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)

def redis_get(key: str):
    return redis_client.get(key)

def redis_set(key: str, value: str, ttl: int = 30):
    redis_client.setex(key, ttl, value)

def redis_delete(key: str):
    redis_client.delete(key)
