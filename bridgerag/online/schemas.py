from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

# --- API Schemas ---

class QueryRequest(BaseModel):
    question: str = Field(..., description="用户的查询问题")
    session_id: Optional[str] = Field(None, description="用于跟踪多轮对话的会话ID (可选)")

class QueryResponse(BaseModel):
    answer: str = Field(..., description="模型生成的最终答案")
    main_documents: List[str] = Field(default_factory=list, description="生成答案所引用的主要文档ID列表")
    conversation_history: str = Field("", description="完整的对话历史，包括中间的思考过程")

class QueryResult(BaseModel):
    """封装单次完整查询结果的数据结构。"""
    question: str
    answer: str
    main_documents: List[str]
    conversation_history: str

# --- LLM I/O Schemas ---

class SynthesisDecision(BaseModel):
    """
    用于验证 SynthesisService 中 LLM 输出的 Pydantic 模型。
    """
    decision: str = Field(..., description="决策类型，必须是 'ANSWER' 或 'SUB_QUESTION'")
    content: str = Field(..., description="决策内容（答案或子问题）")
    next_question: Optional[str] = Field(None, description="如果决策是 SUB_QUESTION，这里是新的子问题")
    summary: Optional[str] = Field(None, description="对当前轮次上下文的总结，用于构建下一轮的历史")
