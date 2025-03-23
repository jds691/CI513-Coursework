from typing import Any

from models import PredictionModel


class DecisionTreeModel(PredictionModel):
    def __init__(self, config: dict[str, Any]):
        super().__init__('Decision Tree', config)