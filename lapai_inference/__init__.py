"""LapAI-Forecast inference and student model package."""

__all__ = ["LapAIStudentConfig", "LapAIStudentCNN"]


def __getattr__(name: str):
    if name in __all__:
        from lapai_inference.model import LapAIStudentConfig, LapAIStudentCNN

        return {"LapAIStudentConfig": LapAIStudentConfig, "LapAIStudentCNN": LapAIStudentCNN}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
