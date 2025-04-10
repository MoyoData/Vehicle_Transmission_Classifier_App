from prometheus_client import Gauge
from monitoring_base import TrainingMonitor  # Assuming you have a base class

class RandomForestModelMonitor(TrainingMonitor):
    def __init__(self, port=8002):
        super().__init__(port)

        # Random Forest-specific metrics
        self.tree_depth_avg = Gauge('rf_avg_tree_depth', 'Average tree depth in the Random Forest')
        self.tree_depth_max = Gauge('rf_max_tree_depth', 'Maximum tree depth in the Random Forest')
        self.leaf_count_avg = Gauge('rf_avg_leaf_count', 'Average number of leaf nodes per tree')
        self.trees_count = Gauge('rf_tree_count', 'Number of trees in the forest')
    
    def record_rf_metrics(self, model):
        """
        Record metrics for a fitted RandomForestClassifier model.
        :param model: Trained RandomForestClassifier
        """
        try:
            depths = [estimator.tree_.max_depth for estimator in model.estimators_]
            leaves = [estimator.tree_.n_leaves for estimator in model.estimators_]
            
            self.trees_count.set(len(model.estimators_))
            self.tree_depth_avg.set(sum(depths) / len(depths))
            self.tree_depth_max.set(max(depths))
            self.leaf_count_avg.set(sum(leaves) / len(leaves))
        
        except Exception as e:
            print(f"Error recording Random Forest metrics: {e}")
