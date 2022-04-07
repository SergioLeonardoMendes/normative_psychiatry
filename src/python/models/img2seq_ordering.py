from typing import Tuple

import numpy as np
import torch


def raster_scan_idx_3d(rows: int, cols: int, depths: int) -> np.ndarray:
    """Return indices of a raster scan"""
    idx = []
    for r in range(rows):
        for c in range(cols):
            for d in range(depths):
                idx.append((r, c, d))
    return np.array(idx)


class Ordering3D:
    """Ordering for 3D volumes"""

    def __init__(
            self,
            dimensions: Tuple[int, int, int],
            ordering_type: str,
            transposed_1: bool,
            transposed_2: bool,
            transposed_3: bool,
            transposed_4: bool,
            transposed_5: bool,
            reflected_rows: bool,
            reflected_cols: bool,
            reflected_depths: bool,
    ) -> None:
        """
        Note: Only supports raster_scan with no transpose and no reflection.

        Args:
            dimensions: tuple with the dimensions of the 3D image in the format of (channel, height, width, depth)
            ordering_type: name of the ordering type selected ("raster_scan")
            transposed_1: Transpose the image before transform to sequence (axes=[1, 2, 0])
            transposed_2: Transpose the image before transform to sequence (axes=[0, 2, 1])
            transposed_3: Transpose the image before transform to sequence (axes=[1, 0, 2])
            transposed_4: Transpose the image before transform to sequence (axes=[2, 0, 1])
            transposed_5: Transpose the image before transform to sequence (axes=[2, 1, 0])
            reflected_rows: Reflect the rows of the image before transform to sequence
            reflected_cols: Reflect the columns of the image before transform to sequence
            reflected_depths: Reflect the depth of the image before transform to sequence
        """

        self.dimensions = dimensions
        self.ordering_type = ordering_type
        self.transposed_1 = transposed_1
        self.transposed_2 = transposed_2
        self.transposed_3 = transposed_3
        self.transposed_4 = transposed_4
        self.transposed_5 = transposed_5
        self.reflected_rows = reflected_rows
        self.reflected_cols = reflected_cols
        self.reflected_depths = reflected_depths

        self.index_sequence = self.get_ordering(
            dimensions=dimensions,
            ordering_type=ordering_type,
            transposed_1=transposed_1,
            transposed_2=transposed_2,
            transposed_3=transposed_3,
            transposed_4=transposed_4,
            transposed_5=transposed_5,
            reflected_rows=reflected_rows,
            reflected_cols=reflected_cols,
            reflected_depths=reflected_depths,
        )

        self.revert_ordering = np.argsort(self.index_sequence)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[self.index_sequence]

    def get_ordering(
            self,
            dimensions: Tuple[int, int, int],
            ordering_type: str,
            transposed_1: bool,
            transposed_2: bool,
            transposed_3: bool,
            transposed_4: bool,
            transposed_5: bool,
            reflected_rows: bool,
            reflected_cols: bool,
            reflected_depths: bool,
    ) -> np.ndarray:
        """ Use only raster_scan with all transpose and reflections False."""
        rows, cols, depths = dimensions
        template = np.arange(rows * cols * depths).reshape(rows, cols, depths)

        def get_ordering_1d(ordering_3d, template):
            ordering_1d = []
            for r, c, d in ordering_3d:
                ordering_1d.append(template[r, c, d])
            ordering_1d = np.array(ordering_1d)
            return ordering_1d

        # transpose dimensions
        if transposed_1:
            template = np.transpose(template, axes=[1, 2, 0])
        if transposed_2:
            template = np.transpose(template, axes=[0, 2, 1])
        if transposed_3:
            template = np.transpose(template, axes=[1, 0, 2])
        if transposed_4:
            template = np.transpose(template, axes=[2, 0, 1])
        if transposed_5:
            template = np.transpose(template, axes=[2, 1, 0])

        # get current dimensions for rows, cols, depths
        rows, cols, depths = template.shape[0], template.shape[1], template.shape[2]

        ordering_3d = eval(f"{ordering_type}_idx_3d")(rows, cols, depths)

        # flip axis according to params [reflected_rows, reflected_cols, reflected_depths]
        for i, current_axis in enumerate([reflected_rows, reflected_cols, reflected_depths]):
            if current_axis:
                template = np.flip(template, axis=current_axis)

        ordering_1d = get_ordering_1d(ordering_3d, template)

        return ordering_1d
