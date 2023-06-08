from .abc import Abstract1NEval


class EVM(Abstract1NEval):
    def __init__(self, confidence_function_name: str) -> None:
        """
        Implemetns Extreme Value Machine (EVM) and uses it for open set recognition
        in case of one sample of each known class. In particular, we do not perform
        Model Reduction, decried in section IV. A

        https://arxiv.org/abs/1506.06112
        """
        self.confidence_function_name = confidence_function_name

    def __call__(
        self,
        probe_feats,
        probe_unc,
        gallery_feats,
        gallery_unc,
        probe_ids,
        gallery_ids,
        fars,
    ):
        pass