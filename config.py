from pydantic_settings import BaseSettings, SettingsConfigDict
from dataclasses import dataclass



class Settings(BaseSettings):
    
    model_config = SettingsConfigDict(env_file=".env",
                                      extra="ignore" )
    
    database_url: str
    s3_bucket: str
    base_model: str
    redis_url: str
    aws_default_region: str
    aws_access_key_id: str
    aws_secret_access_key: str
    device: str = "mps"
    dtype: str = "float16"
    aws_profile: str = "lora-serve"

settings = Settings()

