from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class ChatRequest(BaseModel):
    queries: List[str] = Field(default_factory=list)

class SqlQuery(BaseModel):
    # Logical label only; NOT used to route connections. Keep for logging/debug.
    database: str = "astradb"
    query: str
    params: Dict[str, Any] = Field(default_factory=dict)
    purpose: Optional[str] = None
    dependsOnPrevious: bool = False
    dependencyKeys: List[str] = Field(default_factory=list)

class QueryPlan(BaseModel):
    searchStrategy: str = "hybrid"
    rewrittenQueries: List[str] = Field(default_factory=list)
    reasoning: Optional[str] = None
