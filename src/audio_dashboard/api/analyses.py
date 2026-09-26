from fastapi import APIRouter

from audio_dashboard import registry

from .schemas import AnalysesOut, AnalysisOut

router = APIRouter(prefix="/v1")


@router.get("/analyses", response_model=AnalysesOut)
def list_analyses():
    analyses_list = []
    for a in registry.ANALYSES:
        analysis = AnalysisOut(
            name=a.name,
            group=a.group,
            label=a.label,
            note=a.note,
            requires=a.requires,
            prefers=a.prefers,
            cost=a.cost,
        )
        analyses_list.append(analysis)
    return AnalysesOut(analyses=analyses_list)
