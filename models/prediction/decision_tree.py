from typing import Any

from sklearn.tree import DecisionTreeClassifier

from models import PredictionModel


class DecisionTreeModel(PredictionModel):
    def __init__(self, config: dict[str, Any]):
        super().__init__('Decision Tree', config)
        self.model = DecisionTreeClassifier()

    def train(self, x: Any, y: Any, validation_data: tuple = None) -> None:
        self.model = DecisionTreeClassifier().fit(x, y)

    def predict(self, x: Any) -> Any:
        return self.model.predict(x)