from pydantic import BaseModel

class Response(BaseModel):
    content: str

class DebugReport(BaseModel):
    error_analysis: str
    root_cause: str
    suggested_fix: str
    has_issues: bool = True