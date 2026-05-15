import asyncpg
from fastapi import APIRouter, Depends

from neat_rag.api.deps import get_connection
from neat_rag.api.schemas import FeedbackRequest, FeedbackResponse
from neat_rag.db.feedback import FeedbackRepository
from neat_rag.models import Feedback

router = APIRouter(tags=["feedback"])


@router.post("/feedback", response_model=FeedbackResponse, status_code=201)
async def create_feedback(
    body: FeedbackRequest,
    conn: asyncpg.Connection = Depends(get_connection),
):
    """Submit a thumbs-up / thumbs-down rating for an agent message."""
    repo = FeedbackRepository(conn)
    feedback = await repo.create_feedback(
        message_id=body.message_id,
        rating=body.rating,
        user_id=body.user_id,
        comment=body.comment,
    )
    return _to_feedback_response(feedback)


def _to_feedback_response(f: Feedback) -> FeedbackResponse:
    return FeedbackResponse(
        id=f.id,
        message_id=f.message_id,
        user_id=f.user_id,
        rating=f.rating,
        comment=f.comment,
        created_at=f.created_at,
    )
