from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str
    s3_bucket: str
    base_model: str
    redis_url: str
    aws_default_region: str
    aws_access_key_id: str
    aws_secret_access_key: str
    device: str = "mps"
    dtype: str = "float16"
    aws_profile: str = ""  # set for local dev; empty in Docker (uses explicit keys)
    gpu_shared_secret: str = (
        ""  # set to enforce auth on /infer; empty = skip (dev mode)
    )
    sqs_queue_url: str = ""  # SQS queue URL for job dispatch; empty = skip (local dev)
    sqs_visibility_timeout: int = (
        900  # seconds before SQS redelivers an undeleted message
    )
    orchestrator_lambda_name: str = (
        ""  # Lambda function name for orchestration; empty = skip
    )


settings = Settings()
