import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from ogbench.impls.utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from ogbench.impls.utils.networks import GCActor, GCValue, LogParam
from ml_collections import FrozenConfigDict


class SACAgent(flax.struct.PyTreeNode):
    """Soft actor-critic (SAC) agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        """Compute the SAC critic loss."""
        next_dist = self.network.select('actor')(batch['next_observations'], batch['value_goals'])
        next_actions, next_log_probs = next_dist.sample_and_log_prob(seed=rng)

        next_qs = self.network.select('target_critic')(batch['next_observations'], batch['value_goals'], actions=next_actions)
        if self.config['min_q']:
            next_q = jnp.min(next_qs, axis=0)
        else:
            next_q = jnp.mean(next_qs, axis=0)

        target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_q
        target_q = target_q - self.config['discount'] * batch['masks'] * next_log_probs * self.network.select('alpha')()

        q = self.network.select('critic')(batch['observations'], batch['value_goals'], actions=batch['actions'], params=grad_params)
        critic_loss = jnp.square(q - target_q).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def actor_loss(self, batch, grad_params, rng):
        """Compute the SAC actor loss."""
        if self.config['actor_loss'] == 'original':
            # Actor loss.
            dist = self.network.select('actor')(batch['observations'], batch['actor_goals'], params=grad_params)
            actions, log_probs = dist.sample_and_log_prob(seed=rng)

            qs = self.network.select('critic')(batch['observations'], batch['actor_goals'], actions=actions)
            if self.config['min_q']:
                q = jnp.min(qs, axis=0)
            else:
                q = jnp.mean(qs, axis=0)

            actor_loss = (log_probs * self.network.select('alpha')() - q).mean()

            # Entropy loss.
            alpha = self.network.select('alpha')(params=grad_params)
            entropy = -jax.lax.stop_gradient(log_probs).mean()
            alpha_loss = (alpha * (entropy - self.config['target_entropy'])).mean()

            total_loss = actor_loss + alpha_loss

            if self.config['tanh_squash']:
                action_std = dist._distribution.stddev()
            else:
                action_std = dist.stddev().mean()

            return total_loss, {
                'total_loss': total_loss,
                'actor_loss': actor_loss,
                'alpha_loss': alpha_loss,
                'alpha': alpha,
                'entropy': -log_probs.mean(),
                'std': action_std.mean(),
            }
        elif self.config['actor_loss'] == 'originalbc':
            # Actor loss.
            dist = self.network.select('actor')(batch['observations'], batch['actor_goals'], params=grad_params)
            actions, log_probs = dist.sample_and_log_prob(seed=rng)

            qs = self.network.select('critic')(batch['observations'], batch['actor_goals'], actions=actions)
            if self.config['min_q']:
                q = jnp.min(qs, axis=0)
            else:
                q = jnp.mean(qs, axis=0)

            actor_loss = (log_probs * self.network.select('alpha')() - q).mean()

            # Entropy loss.
            alpha = self.network.select('alpha')(params=grad_params)
            entropy = -jax.lax.stop_gradient(log_probs).mean()
            alpha_loss = (alpha * (entropy - self.config['target_entropy'])).mean()


            log_prob = dist.log_prob(batch['actions'])
            bc_loss = -(self.config['alpha'] * log_prob).mean()

            total_loss = actor_loss + alpha_loss + bc_loss

            if self.config['tanh_squash']:
                action_std = dist._distribution.stddev()
            else:
                action_std = dist.stddev().mean()

            return total_loss, {
                'total_loss': total_loss,
                'actor_loss': actor_loss,
                'alpha_loss': alpha_loss,
                'bc_loss': bc_loss,
                'alpha': alpha,
                'entropy': -log_probs.mean(),
                'std': action_std.mean(),
            }

        elif self.config['actor_loss'] == 'ddpgbc':
            # DDPG+BC loss.
            assert not self.config['discrete']

            dist = self.network.select('actor')(batch['observations'], batch['actor_goals'], params=grad_params)
            actions, log_probs = dist.sample_and_log_prob(seed=rng)
            q_actions = jnp.clip(dist.sample(seed=rng), -1, 1)
            qs = self.network.select('critic')(batch['observations'], batch['actor_goals'], actions=q_actions)
            if self.config['min_q']:
                q = jnp.min(qs, axis=0)
            else:
                q = jnp.mean(qs, axis=0)
            # Normalize Q values by the absolute mean to make the loss scale invariant.
            q_loss = -q.mean() / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)
            log_prob = dist.log_prob(batch['actions'])

            bc_loss = -(self.config['alpha'] * log_prob).mean()

            actor_loss = q_loss + bc_loss

            return actor_loss, {
                'actor_loss': actor_loss,
                'q_loss': q_loss,
                'bc_loss': bc_loss,
                'q_mean': q.mean(),
                'q_abs_mean': jnp.abs(q).mean(),
                'bc_log_prob': log_prob.mean(),
                'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
            }
        else:
            raise ValueError(f'Unsupported actor loss: {self.config["actor_loss"]}')

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the actor."""
        dist = self.network.select('actor')(observations, goals, temperature=temperature)
        actions = dist.sample(seed=seed)
        actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = ex_observations

        if isinstance(config, FrozenConfigDict):
            config = config.as_configdict()

        action_dim = ex_actions.shape[-1]

        if config['target_entropy'] is None:
            config['target_entropy'] = -config['target_entropy_multiplier'] * action_dim

        # Define critic and actor networks.
        critic_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
        )


        actor_def = GCActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            state_dependent_std=False,
            const_std=config['const_std'],
        )

        # Define the dual alpha variable.
        alpha_def = LogParam()

        network_info = dict(
            critic=(critic_def, (ex_observations, ex_goals, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_goals, ex_actions)),
            actor=(actor_def, (ex_observations, ex_goals)),
            alpha=(alpha_def, ()),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_critic'] = params['modules_critic']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='sac',  # Agent name.
            lr=1e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512),  # Value network hidden dimensions.
            layer_norm=False,  # Whether to use layer normalization.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            target_entropy=ml_collections.config_dict.placeholder(float),  # Target entropy (None for automatic tuning).
            target_entropy_multiplier=0.5,  # Multiplier to dim(A) for target entropy.
            tanh_squash=True,  # Whether to squash actions with tanh.
            state_dependent_std=True,  # Whether to use state-dependent standard deviations for actor.
            actor_fc_scale=0.01,  # Final layer initialization scale for actor.
            min_q=True,  # Whether to use min Q (True) or mean Q (False).
            const_std=True,
            actor_loss='ddpgbc',
            alpha=0.3,
            discrete=False,  # Whether the action space is discrete.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            dataset_class='GCDataset',  # Dataset class name.
            value_p_curgoal=0.0,  # Probability of using the current state as the value goal.
            value_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.0,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            gc_negative=False,  # Unused (defined for compatibility with GCDataset).
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
        )
    )
    return config
