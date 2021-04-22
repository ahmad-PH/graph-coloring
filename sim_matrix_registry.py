
from __future__ import annotations
import os
from typing import Union
import torch

class SimMatrixRegistry:
    instance = None

    @staticmethod
    def get_instance() -> SimMatrixRegistry:
        if (SimMatrixRegistry.instance is None):
            SimMatrixRegistry.instance = SimMatrixRegistry()
        return SimMatrixRegistry.instance

    def register_similarity_matrix(self, matrix: torch.Tensor, name: str):
        if os.path.exists(self._get_matrix_path(name)):
            raise ValueError('similarity matrix with name {} already exists'.format(name))
        else:
            torch.save(matrix, self._get_matrix_path(name))
            self.existing_matrices.append(name)

    
    def get_similarity_matrix(self, name: str) -> Union[torch.Tensor, None]:
        if name in self.existing_matrices:
            return torch.load(self._get_matrix_path(name))
        else:
            return None

    working_dir_name = '.sim_matrix_registry'

    def __init__(self):
        self.working_dir_path = os.path.join(os.path.dirname(__file__), SimMatrixRegistry.working_dir_name)
        self._ensure_working_folder_exists()
        self.existing_matrices = [os.path.splitext(matrix_file)[0] for matrix_file in os.listdir(self.working_dir_path)]

    def _ensure_working_folder_exists(self):
        if not os.path.isdir(self.working_dir_path):
            os.mkdir(self.working_dir_path)

    def _get_matrix_path(self, matrix_name: str) -> str:
        return os.path.join(self.working_dir_path, '{}.pt'.format(matrix_name))


