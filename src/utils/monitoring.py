
# Add these metrics to your monitoring.py for decision tree models
class TreeModelMonitor(TrainingMonitor):
    def __init__(self, port=8002):
        super().__init__(port)
        
        # Tree metrics
        self.tree_depth = Gauge('tree_max_depth', 'Maximum tree depth')
        self.tree_leaves = Gauge('tree_leaf_count', 'Number of leaf nodes')
        self.trees_count = Gauge('ensemble_tree_count', 'Number of trees in the ensemble')
        
        # Boosting iteration counters
        self.boost_round = Counter('boosting_rounds_total', 'Total boosting rounds completed')
        self.iteration_improvement = Gauge('iteration_improvement', 'Performance improvement in the last iteration')
                
    def record_tree_metrics(self, depth=None, leaves=None, trees=None):
        """Record tree structure metrics"""
        if depth is not None:
            self.tree_depth.set(depth)
        if leaves is not None:
            self.tree_leaves.set(leaves)
        if trees is not None:
            self.trees_count.set(trees)
            
    def record_boost_round(self, improvement=None):
        """Record a completed boosting round"""
        self.boost_round.inc()
        if improvement is not None:
            self.iteration_improvement.set(improvement)
