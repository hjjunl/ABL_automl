from .log import get_logger
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
import pandas as pd

logger = get_logger(__name__)

def xai(model, X_test, y_test, large_data: bool = False):
    """
    The function creates xai report in HTML format.
    Args:
        X_test (pd.DataFrame): Testing data for modelling
        y_test (pd.Series): Target variable for testing
        large_data (Bool)
    Returns:
        NoReturns
    """
    if large_data:
        df_target = pd.DataFrame({'target':y_test})
        df = pd.concat(X_test, df_target, axis=1)
        df = df.sample(frac=0.1)
        X_test = df.drop(columns=['target'])
        y_test = df[['target']].values
    
    explainer = ClassifierExplainer(model, X_test, y_test)
    ExplainerDashboard(explainer).run(port=8050) ## port 8050

