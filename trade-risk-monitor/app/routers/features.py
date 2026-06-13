from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime
from typing import Optional
from app.database import get_db
from app.models.features import FeatureSnapshot
from app.schemas.features import FeatureSnapshotResponse

router = APIRouter()

@router.get("/portfolios/{portfolio_id}/features/latest", response_model=FeatureSnapshotResponse)
def get_latest_features(
    portfolio_id: int,
    ticker: Optional[str] = None,
    snapshot_type: Optional[str] = None,
    db: Session = Depends(get_db)
):
    query = db.query(FeatureSnapshot).filter(FeatureSnapshot.portfolio_id == portfolio_id)
    if snapshot_type:
        query = query.filter(FeatureSnapshot.snapshot_type == snapshot_type)
    elif ticker:
        query = query.filter(FeatureSnapshot.snapshot_type == "POSITION")
    else:
        query = query.filter(FeatureSnapshot.snapshot_type == "PORTFOLIO")
    if ticker:
        query = query.filter(FeatureSnapshot.ticker == ticker)
    snapshot = query.order_by(FeatureSnapshot.timestamp.desc(), FeatureSnapshot.id.desc()).first()
    if not snapshot:
        raise HTTPException(status_code=404, detail="No feature snapshot found")
    return snapshot

@router.get("/portfolios/{portfolio_id}/features/history", response_model=list[FeatureSnapshotResponse])
def get_features_history(
    portfolio_id: int,
    ticker: Optional[str] = None,
    snapshot_type: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    query = db.query(FeatureSnapshot).filter(FeatureSnapshot.portfolio_id == portfolio_id)
    if snapshot_type:
        query = query.filter(FeatureSnapshot.snapshot_type == snapshot_type)
    elif ticker:
        query = query.filter(FeatureSnapshot.snapshot_type == "POSITION")
    else:
        query = query.filter(FeatureSnapshot.snapshot_type == "PORTFOLIO")
    if ticker:
        query = query.filter(FeatureSnapshot.ticker == ticker)
    if start_time:
        query = query.filter(FeatureSnapshot.timestamp >= start_time)
    if end_time:
        query = query.filter(FeatureSnapshot.timestamp <= end_time)
    snapshots = query.order_by(FeatureSnapshot.timestamp.desc(), FeatureSnapshot.id.desc()).limit(limit).all()
    return snapshots
