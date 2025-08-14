from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

from GBC.gyms.isaaclab_45.managers.ref_obs_term_cfg import ReferenceObservationCfg, ReferenceObservationGroupCfg, ReferenceObservationTermCfg


@configclass
class RLReferenceObservationGroupCfg(ReferenceObservationGroupCfg):
    trans = ReferenceObservationTermCfg(name="trans")


@configclass
class ManagerBasedRefRLEnvCfg(ManagerBasedRLEnvCfg):
    # ref_action: object = {
    #     "note": "Honestly, there isn't anything in this configuration at the moment. But this configuration must be non-empty in order to prepare terms"
    # }

    ref_observation = None