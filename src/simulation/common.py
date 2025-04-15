"""Common functions and classes for the package."""

import dataclasses
import getpass
import os
from collections.abc import Sequence
from pathlib import Path

SEED = 0


@dataclasses.dataclass(frozen=True, repr=True, eq=True, order=True)
class Conf:
    """
    Configuration for training and evaluation of models.

    :ivar model_name: name of the model
    :ivar dnn_layout: layout of the dynamic mlp
    :ivar snn_layout: layout of the static mlp
    :ivar dnn_act: activation function of the dynamic mlp
    :ivar snn_act: activation function of the static mlp
    :ivar lse_act: activation function to transform parameters of LSE layers
    :ivar use_lse: whether to use jax.nn.logsumexp in LSE layers
    :ivar final_act: activation function of the final layer
    :ivar schedule: learning rate schedule
    :ivar optim: optimizer
    :ivar max_lrate: maximum learning rate
    :ivar min_lrate: minimum learning rate
    :ivar opt_steps: itial number of steps in a learning rate schedule
    :ivar batch_size_train: batch size for training
    :ivar batch_size_val: batch size for validation
    :ivar loss: loss function
    :ivar data_mode: mode of the data
    :ivar l1_reg: L1 regularization strength
    :ivar n_epoch: number of epochs
    :ivar min_epoch: minimum number of epochs before early stopping
    :ivar clip: whether to clip model outputs and regression targets based
    on integration tolerances
    """

    model_name: str = ''
    dnn_layout: str = ''
    snn_layout: str = ''
    dnn_act: str = ''
    snn_act: str = ''
    lse_act: str = ''
    final_act: str = 'softplus'
    use_lse: bool = True
    optim: str = 'adam'
    max_lrate: float = 1e-2
    lrate_span: float = 1e1
    lrate_decay: float = 0.1
    warmup_fct: float = 0.1
    opt_steps: int = 1
    batch_size_train: int = 2048
    batch_size_val: int = 65392
    loss: str = 'l2'
    data_mode: str = 'static'
    l1_reg: float = 1e-5
    n_epoch: int = 2**10
    min_epoch: int = 100
    clip: bool = True
    patience: int = 15

    def __str__(self):
        """
        Return string representation of the configuration.

        Note: enables printing of Conf objects in fire
        """
        return super().__str__()


if os.getenv('CLUSTER_SSH_HOST'):
    from fabric import Connection
    from fabric.transfer import Transfer

    connection = Connection(
        host=os.getenv('CLUSTER_SSH_HOST'),
        user=getpass.getuser(),
    )
    transfer = Transfer(connection)
else:
    transfer = None


def load_remote_file(filename):
    """Open a file on the remote server."""
    if transfer is None:
        return
    filename_remote = str(filename).replace(
        str(get_base_dir()), os.getenv('CLUSTER_PATH')
    )
    transfer.get(filename_remote, filename)


def get_base_dir() -> Path:
    """Return the base directory of the project."""
    return Path(__file__).parent.parent


def get_model_dir(model_name: str):
    """
    Return the directory of the model with the given name.

    :param model_name: name of the model
    """
    return get_base_dir() / 'base_models' / model_name


@dataclasses.dataclass(frozen=True, eq=True, order=True)
class ModelInfo:
    """
    Information about a model.

    This information needs to be synchronized with the respective pysb model
    implementations.

    :ivar state_names: names of the states
    :ivar state_initials: names of the initial values of the states
    :ivar x0_cg: initial values of the coarse-grained states
    :ivar cg_par_names: names of the coarse-grained parameters
    :ivar fg_par_names: names of the fine-grained parameters
    :ivar kcat_names: names of the catalytic rates that are to be approximated
    """

    state_names: Sequence[str]
    state_initials: Sequence[str]
    x0_cg: Sequence[str]
    cg_par_names: Sequence[str]
    fg_par_names: Sequence[str]
    kcat_names: Sequence[str]


MODEL_INFO: dict[str, callable] = {
    'two_step_enzyme': lambda with_delay: ModelInfo(
        state_names=(
            'K(p=None)',
            "P(phospho='u', k=None)",
            "P(phospho='p', k=None)",
            *(
                [
                    'P_p_d',
                    'P_u_d',
                    'K_d',
                ]
                if with_delay
                else []
            ),
        ),
        state_initials=(
            'K0',
            'uP0',
            'pP0',
        ),
        x0_cg=(
            'pP0_preeq',
            'K0_preeq',
            *(
                [
                    'pP0_preeq',
                    'K0_preeq',
                ]
                if with_delay
                else []
            ),
        ),
        cg_par_names=(
            'K0',
            'tP',
            'kcat',
            'kinact',
            'krev',
        ),
        fg_par_names=(
            'koff_substrate',
            'kD_substrate',
            'kcat',
            'kinact',
            'krev',
        ),
        kcat_names=('kcat_cg',),
    ),
}
