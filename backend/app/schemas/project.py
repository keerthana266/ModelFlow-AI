from pydantic import BaseModel, ConfigDict, Field


class ProjectCreateRequest(BaseModel):
    name: str = Field(min_length=2, max_length=160)
    description: str | None = Field(default=None, max_length=1000)


class ProjectUpdateRequest(BaseModel):
    name: str | None = Field(default=None, min_length=2, max_length=160)
    description: str | None = Field(default=None, max_length=1000)


class ProjectResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    project_id: str
    owner_id: str
    name: str
    description: str | None
