# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from .fixmatch import FixMatch, FixMatch_lossnet
from .flexmatch import FlexMatch, FlexMatch_lossnet

from .pimodel import PiModel
from .meanteacher import MeanTeacher
from .pseudolabel import PseudoLabel, PseudoLabel_lossnet
from .uda import UDA, UDA_net, UDA_lossnet
from .mixmatch import MixMatch
from .vat import VAT
from .remixmatch import ReMixMatch
from .crmatch import CRMatch
from .dash import Dash
# from .mpl import MPL
from .fullysupervised import FullySupervised,FullySupervised_lossnet,FullySupervised_lossnet_mix
from .comatch import CoMatch
from .simmatch import SimMatch
from .adamatch import AdaMatch

# if any new alg., please append the dict
name2alg = {
    'fullysupervised': FullySupervised,
    'supervised': FullySupervised,
    'fullysupervised_lossnet': FullySupervised_lossnet,
    'fullysupervised_lossnet_mix': FullySupervised_lossnet_mix,
    'fixmatch': FixMatch,
    'fixmatch_lossnet':FixMatch_lossnet,
    'flexmatch': FlexMatch,
    'flexmatch_lossnet': FlexMatch_lossnet,
    'adamatch': AdaMatch,
    'pimodel': PiModel,
    'meanteacher': MeanTeacher,
    'pseudolabel': PseudoLabel,
    'pseudolabel_lossnet': PseudoLabel_lossnet,
    'uda': UDA,
    'uda_lossnet':UDA_lossnet,
    'vat': VAT,
    'mixmatch': MixMatch,
    'remixmatch': ReMixMatch,
    'crmatch': CRMatch,
    'comatch': CoMatch,
    'simmatch': SimMatch,
    'dash': Dash,
    # 'mpl': MPL
}

def get_algorithm(args, net_builder, tb_log, logger):

    try:
        alg = name2alg[args.algorithm](
            args=args,
            net_builder=net_builder,
            tb_log=tb_log,
            logger=logger
        )
        return alg
    except KeyError as e:
        print('keyerror:',args.algorithm)
        print(f'Unknown algorithm: {str(e)}')


