from typing import Sequence

import torch
import torch.nn as nn


class BoundingBox(nn.Module):

    def __init__(self, dimensions: Sequence):
        '''
        Builds a bounding box with the specified dimensions.
        :param dimensions: [(x_low, x_high), (y_low, y_high), (z_low, z_high)] sequence with dimensions along each axis
        '''

        super(BoundingBox, self).__init__()

        if len(dimensions) != 3:
            raise Exception(f"Dimenions should have dimension 3, but dimension ({len(dimensions)}) was passed")

        dimensions = torch.as_tensor(dimensions, dtype=torch.float32)
        self.register_buffer("dimensions", dimensions, persistent=False)

    def get_center_offset(self, device="cuda:0") -> torch.Tensor:
        '''
        Computes the offset of the bounding box with repect to the the canonical box center (0, 0, 0)
        :return: (x, y, z) tensor with offset of the real center from the canonical one
        '''

        # Computes where the real center is located
        real_center = self.dimensions[:, 0]
        real_center = real_center + (self.dimensions[:, 1] - self.dimensions[:, 0]) / 2

        return real_center.to(device)

    def is_inside(self, points: torch.Tensor):
        '''
        Checks whether a given point is inside the bounding box

        :param points: (..., 3) tensor of points
        :return: (...) tensor of booleans with True if the corresponding point lies inside the bounding box
        '''

        high = self.dimensions[:, 1]
        low = self.dimensions[:, 0]

        below_high = torch.all(points <= high, dim=-1)
        above_low = torch.all(points >= low, dim=-1)
        return torch.logical_and(below_high, above_low)

    def get_size(self) -> torch.Tensor:
        '''
        Returns the size of the box sides

        :return: (x, y, z) tensor with size of the sides of the box
        '''
        return self.dimensions[:, 1] - self.dimensions[:, 0]

    def get_corner_points(self) -> torch.Tensor:
        '''
        Obtain the points that form the corners of the box.
        Point 0 and point 6 represent respectively the one with all lowest and all highest points
        :return: (8, 3) tensor with corner points
        '''

        transposed_dimensions = self.dimensions.T

        corner_points = torch.zeros((8, 3), dtype=torch.float32, device=self.dimensions.device)

        # Computes all the possible combinations selecting the low or the high bound
        # from the dimensions tensor for each x,y,z dimension
        corner_points[0] = transposed_dimensions[0]
        corner_points[6] = transposed_dimensions[1]

        corner_points[1][0] = corner_points[6][0]
        corner_points[1][1] = corner_points[0][1]
        corner_points[1][2] = corner_points[0][2]

        corner_points[2][0] = corner_points[6][0]
        corner_points[2][1] = corner_points[0][1]
        corner_points[2][2] = corner_points[6][2]

        corner_points[3][0] = corner_points[0][0]
        corner_points[3][1] = corner_points[0][1]
        corner_points[3][2] = corner_points[6][2]

        corner_points[4][0] = corner_points[0][0]
        corner_points[4][1] = corner_points[6][1]
        corner_points[4][2] = corner_points[0][2]

        corner_points[5][0] = corner_points[6][0]
        corner_points[5][1] = corner_points[6][1]
        corner_points[5][2] = corner_points[0][2]

        corner_points[7][0] = corner_points[0][0]
        corner_points[7][1] = corner_points[6][1]
        corner_points[7][2] = corner_points[6][2]

        return corner_points

    def get_edge_points(self, points_per_edge: int=5) -> torch.Tensor:
        '''
        Obtains points along the outer edges of the box
        :return: (12 * points_per_edge + 8, 3) tensor with points along each edge
        '''

        # Indexes of the corners that delimit edges. Successive pairs of indexes form an edge
        edge_indexes = torch.as_tensor([0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7], dtype=torch.long, device=self.dimensions.device)
        edge_indexes = edge_indexes.unsqueeze(-1).repeat((1, 3))

        corner_points = self.get_corner_points()
        edge_corners = torch.gather(corner_points, 0, edge_indexes)
        edge_corners = torch.reshape(edge_corners, (12, 2, 3))  # 12 edges, begin and end point

        # Difference between begin and end of the edge (12, 3, 1)
        deltas = edge_corners[:, 1] - edge_corners[:, 0]
        deltas = deltas.unsqueeze(-1)

        # Fractions at which to take points on the edges
        edge_positions = torch.linspace(0.0, 1.0, points_per_edge + 2, device=self.dimensions.device)
        edge_positions = edge_positions[1:-1]  # Interested only in the middle points

        # (12, 3, points_per_edge)
        edge_points = edge_corners[:, 0].unsqueeze(-1) + (deltas * edge_positions)
        # (12, points_per_edge, 3)
        edge_points = torch.transpose(edge_points, 1, 2)
        edge_points = torch.reshape(edge_points, (-1, 3))

        # Adds the corners
        edge_points = torch.cat([corner_points, edge_points], dim=0)

        return edge_points


if __name__ == "__main__":
    dimensions = [(0, 1), (0, 1), (0, 1)]

    bounding_box = BoundingBox(dimensions)
    edge_points = bounding_box.get_edge_points(3)

    inside_points = [(0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.7, 0.4, 0.1)]
    outside_points = [(0.0, 0.0, -0.1), (2.0, 1.0, 0.0), (2.0, 2.0, 2.0), (0.0, -0.1, 0.5), (1.1, 0.4, 0.1)]
    all_sets = [inside_points, outside_points]
    for current_set in all_sets:
        current_points = torch.as_tensor(current_set)
        is_inside = bounding_box.is_inside(current_points)
        print(is_inside.detach().cpu().numpy().tolist())

    pass


















