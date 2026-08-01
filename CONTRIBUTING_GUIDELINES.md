# Contributing Guidelines

## Type hinting
Make sure every function argument, return value, and class attribute is type hinted. This makes debugging easier and works well with static type checkers like Ruff.

Example :-
Do this 
```python
func(num : int, config : data_config, learning_rate : float, train_mode : bool) -> None:
    ...
```
Instead of 
```python
func(num, config, learning_rate, train_mode):
    ...
```

## Docstring

Make sure to add docstrings to every function/class you define. The docstring should include the function arguments, brief description about what the function does, and what the function returns with type hinting.

Example :-
```python
class MLP(nn.Module):
    '''
    A class definition of a Multi - layer perceptron typically used in transformers
    Uses 2 linear layers which act as up_projection (d_model -> hidden) and down_projection
    (hidden -> d_model), with an activation function (ReLU) in between
    '''
    
    def __init__(self, config : model_config):
        '''
        Instantiates the class.
        Args :-
            config (model_config) : Dataclass with model hyperparameters
        '''
        super().__init__()
        self.up_proj = nn.Linear(config.d_model, config.hidden)
        self.down_proj = nn.Linear(config.hidden, config.d_model)

    def forward(self,x) -> Tensor:
        '''
        Calls the forward function on the input tensor (x)
        Args :-
            x (Tensor) : Input Tensor of shape (B, T, C)
        Returns :-  
            out (Tensor) : Output Tensor after processing with the same shape
        '''
        out = self.down_proj(F.relu(self.up_proj(x)))
        return out
```

## Modularity

Try to encapsulate most of the code you write into functions, so that they can be re-used in other parts of the codebase.
Eg:
  A function which pre-trains a model can also be used for post-training methods like SFT, DPO, GRPO, etc ...

## Unit Tests

Every new function or class should have a corresponding unit test in that directory's `test.py` file. Tests should verify the expected behavior, output shapes (if applicable), edge cases, and failure conditions.

Example:

```python
import unittest
import torch

from model import MLP
from config import model_config


class TestMLP(unittest.TestCase):
    """
    Unit tests for the MLP module.
    """

    def test_forward_output_shape(self) -> None:
        """
        Ensures that the MLP preserves the expected output shape.
        """
        config:model_config=model_config(
            d_model=128,
            hidden=256
        )

        model = MLP(config)
        x = torch.randn(32,128)

        out = model(x)

        self.assertEqual(out.shape,(32,128))
        self.assertFalse(torch.isnan(out).any())
```

## Pylint

Pylint is a static code analysis tool that catches bad coding practices and style issues.

It's not required, but it's a good idea to run it before opening a PR. If it points out something reasonable, try fixing it.

```bash
pylint .
```

