"""Class to manage parameter configuration."""

from dataclasses import dataclass, field
import typing

@dataclass()
class PCParameters():
    #For computing note weights
    remove_repeats: bool = True
    syncopes: bool = True
    metric_weights: str = 'beatstrength'
    accumulate_weight: bool = False

    #For computing weighted pitch context vectors
    context_type: str = 'beats'

    len_context: 'typing.Any' = None
    len_context_pre: 'typing.Any' = 'auto'
    len_context_pre_auto: bool = True
    len_context_post: 'typing.Any' = 'auto'
    len_context_post_auto: bool = True

    len_context_params: 'typing.Any' = None
    len_context_params_pre: dict = field(default_factory=lambda : {})
    len_context_params_post: dict = field(default_factory=lambda : {})

    use_metric_weights: bool = None
    use_metric_weights_pre: bool = True
    use_metric_weights_post: bool = True

    use_distance_weights: bool = None
    use_distance_weights_pre: bool = True
    use_distance_weights_post: bool = True

    min_distance_weight: float = None
    min_distance_weight_pre: float = 0.0
    min_distance_weight_post: float = 0.0

    include_focus: bool = None
    include_focus_pre: bool = True
    include_focus_post: bool = True

    partial_notes: bool = True
    epsilon: float = 1.0e-4

    def __post_init__(self):
        #make sure these exist
        self.len_context_pre_auto = False
        self.len_context_post_auto = False

        #split into pre an post contexts       
        if self.len_context != None:
            self.len_context_pre = self.len_context
            self.len_context_post = self.len_context
            self.len_context = None
       
        if self.len_context_pre == 'auto':
            self.len_context_pre_auto = True
            self.len_context_pre = None
       
        if self.len_context_post == 'auto':
            self.len_context_post_auto = True
            self.len_context_post = None

        if self.len_context_params != None:
            self.len_context_params_pre = self.len_context_params
            self.len_context_params_post = self.len_context_params
            self.len_context_params = None

        if self.use_metric_weights != None:
            self.use_metric_weights_pre = self.use_metric_weights
            self.use_metric_weights_post = self.use_metric_weights
            self.use_metric_weights = None
       
        if self.use_distance_weights != None:
            self.use_distance_weights_pre = self.use_distance_weights
            self.use_distance_weights_post = self.use_distance_weights
            self.use_distance_weights = None
       
        if self.min_distance_weight != None:
            self.min_distance_weight_pre = self.min_distance_weight
            self.min_distance_weight_post = self.min_distance_weight
            self.min_distance_weight = None
       
        if self.include_focus != None:
            self.include_focus_pre = self.include_focus
            self.include_focus_post = self.include_focus
            self.include_focus = None
