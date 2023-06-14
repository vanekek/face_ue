from typing import Any
import numpy as np
from tqdm import tqdm
class VerifEval:
    def __init__(self, distance_function, batch_size=10000) -> None:
        self.distance_function = distance_function
        self.batch_size = batch_size

    def __call__(self,
                template_norm_feats: np.ndarray,
                unique_templates: np.ndarray,
                p1: np.ndarray,
                p2: np.ndarray,
                ) -> Any:
        # try:
        #     print(">>>> Trying cupy.")
        #     import cupy as cp

        #     template_norm_feats = cp.array(template_norm_feats)
        #     score_func = lambda feat1, feat2: cp.sum(feat1 * feat2, axis=-1).get()
        #     test = score_func(
        #         template_norm_feats[:batch_size], template_norm_feats[:batch_size]
        #     )
        # except:
        #     score_func = lambda feat1, feat2: np.sum(feat1 * feat2, -1)

        template2id = np.zeros(max(unique_templates) + 1, dtype=int)
        template2id[unique_templates] = np.arange(len(unique_templates))

        steps = int(np.ceil(len(p1) / self.batch_size))
        scores = []
        for id in tqdm(range(steps), "Verification"):
            feat1 = template_norm_feats[
                template2id[p1[id * self.batch_size : (id + 1) * self.batch_size]]
            ]
            feat2 = template_norm_feats[
                template2id[p2[id * self.batch_size : (id + 1) * self.batch_size]]
            ]
            scores.extend(self.distance_function(feat1, feat2))
        return np.array(scores)
