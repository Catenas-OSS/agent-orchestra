"""TUI package for Agent Orchestra dashboard."""

# Optional imports - only load if dependencies are available
try:
    from .model import RunTUIModel, NodeState
    from .main import ProfessionalUrwidTUI
    
    __all__ = [
        "RunTUIModel", 
        "NodeState",
        "ProfessionalUrwidTUI"
    ]
except ImportError:
    # TUI dependencies not available
    __all__ = []
