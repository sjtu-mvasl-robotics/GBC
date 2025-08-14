"""
    Physics Modifier Manager
    =========================
    This manager is responsible for applying external physics modifiers to the simulation, e.g. External forces, torques, random pushes, etc.
    This is used for curriculum learning for robot imitation tasks.
    All terms in physics modifier manager are activated every physics step.

"""
import time
import torch
from collections.abc import Sequence
import inspect
from prettytable import PrettyTable

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import ManagerBase, ManagerTermBase
from isaaclab.utils import modifiers
from .physics_modifier_cfg import PhysicsModifierTermCfg

class PhysicsModifierManager(ManagerBase):
    def __init__(self, cfg: object, env: ManagerBasedEnv):
        self._env = env
        self._term_names: list[str] = list()
        self._term_cfgs: list[PhysicsModifierTermCfg] = list()

        super().__init__(cfg, env)

    def __str__(self) -> str:
        msg = f"<PhysicsModifierManager> contains {len(self._term_names)} active terms.\n"
        # create table for term information
        table = PrettyTable()
        table.field_names = ["Term Name", "Description"]
        for term_name, term_cfg in zip(self._term_names, self._term_cfgs):
            table.add_row([term_name, term_cfg.description])
        msg += table.get_string()
        msg += "\n"
        return msg
    
    @property
    def active_terms(self) -> list[str]:
        return self._term_names
    
    def apply(self, env_ids: Sequence[int] | None = None):
        
        for term_name, term_cfg in zip(self._term_names, self._term_cfgs):
            func_params = dict(filter(lambda x: x[0] not in term_cfg.func._overrides, term_cfg.params.items()))
            term_cfg.func(self._env, env_ids, **func_params)
    
    def update(self, env_ids: Sequence[int] | None = None):
        """Update physics modifiers for the given environment ids.

        This function is called every observation step to determine whether to update physics modifiers parameters.
        
        """
        if env_ids is None:
            env_ids = slice(None)
        
        for name, term_cfg in zip(self._term_names, self._term_cfgs):
            new_kwargs = {
                "env": self._env,
                "env_ids": env_ids,
            }
            new_kwargs = {**new_kwargs, **term_cfg.params}
            term_cfg.func.update(**new_kwargs)
            
    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        for term_cfg in self._term_cfgs:
            term_cfg.func.reset(env_ids=env_ids)
        
        return {}
    
    def set_term_cfg(self, term_name: str, cfg: PhysicsModifierTermCfg):
        assert term_name in self._term_names, f"term_name {term_name} not found in active terms."
        self._term_cfgs[self._term_names.index(term_name)] = cfg
        
    def get_term_cfg(self, term_name: str) -> PhysicsModifierTermCfg:
        assert term_name in self._term_names, f"term_name {term_name} not found in active terms."
        return self._term_cfgs[self._term_names.index(term_name)]
    
    def _prepare_terms(self):
        if isinstance(self.cfg, dict):
            cfg_items = self.cfg.items()
        else:
            cfg_items = self.cfg.__dict__.items()
        for term_name, term_cfg in cfg_items:
            if not isinstance(term_cfg, PhysicsModifierTermCfg):
                raise TypeError(
                    f"Configuration for the term '{term_name}' is not of type PhysicsModifierTermCfg."
                    f" Received: '{type(term_cfg)}'."
                )
                
            self._resolve_common_term_cfg(term_name, term_cfg, min_argc=2)
            
            # check if the term function has attribute 'update'
            if not hasattr(term_cfg.func, "update"):
                raise AttributeError(
                    f"Physics modifier function '{term_name}' does not have 'update' attribute."
                    f"Decorate your function with @update(update_strategy) from ``physics_modifier_function_wrapper.py``."
                )
                
            self._term_names.append(term_name)
            self._term_cfgs.append(term_cfg)
            

    